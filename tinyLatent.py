"""Neural image compression via latent texture + SIREN decoder.

Two-scale latent texture:
  - Lo-res grid  (ch_lo channels, coarser scale_lo)  — captures global structure
  - Hi-res grid  (ch_hi channels, finer  scale_hi)  — captures fine detail (optional)

Both grids are sampled with bilinear F.grid_sample and concatenated before the
SIREN decoder — directly inspired by RTXNTC's dual-resolution latent shape.

Quantization-Aware Training (QAT):
  When --qat_bits 8 is set, latent values are fake-quantized with a
  Straight-Through Estimator from step --qat_start onward.  The decoder
  learns to tolerate discrete values, so --save_latent emits real uint8 .npz
  (far more compressible by JPEG than fp16).

Architecture:
    (x, y) ─┬─► [grid_sample lo C_lo×H/s_lo×W/s_lo] ─┐
             └─► [grid_sample hi C_hi×H/s_hi×W/s_hi] ─┴─► cat → SirenLayer × depth
                                                              → Linear + Sigmoid → (r,g,b)

Orthogonal comparison:
    - Decoder is IDENTICAL to tinyMLP.py (same SirenLayer / hidden / depth / omega_0)
    - Only the encoder changes

Usage:
    python tinyLatent.py                              # single-scale (compat mode)
    python tinyLatent.py --ch_lo 16 --ch_hi 2        # two-scale (default two-scale config)
    python tinyLatent.py --ch_lo 16 --ch_hi 2 --qat_bits 8   # + QAT
    python tinyLatent.py --vis_latent                 # show latent grid in 3rd panel
    python tinyLatent.py --save_latent latent.npz     # save quantised artifact

Controls:
    ESC / Q  — quit
    Space    — pause / resume
    S        — save reconstruction snapshot
"""

import argparse
import pathlib
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

local_dir = pathlib.Path(__file__).parent.absolute()

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import sys
sys.path.insert(0, str(local_dir))
from tinyMLP import SirenLayer, render_full


# ---------------------------------------------------------------
# QAT helper: per-channel Straight-Through Estimator
# ---------------------------------------------------------------
def quantize_ste_perchannel(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Fake-quantize a (1, C, H, W) tensor per channel with STE gradient.

    Forward:  x → round to 2^bits levels per channel
    Backward: gradient passes straight through (d_out/d_in = 1)

    Args:
        x:    Latent tensor of shape (1, C, H, W)
        bits: Quantisation bit-depth (default 8 → uint8)
    Returns:
        Fake-quantised tensor with same shape, same dtype, but discrete values.
    """
    levels = float(2 ** bits - 1)                          # 255.0 for 8-bit
    # Per-channel min/max (detached — no gradient through the scale)
    x_flat  = x.detach().flatten(2)                        # (1, C, H*W)
    x_min   = x_flat.min(dim=-1).values[..., None, None]   # (1, C, 1, 1)
    x_max   = x_flat.max(dim=-1).values[..., None, None]
    scale   = (x_max - x_min).clamp(min=1e-6)
    # Normalise → quantise → de-normalise
    x_norm  = (x - x_min) / scale                          # [0, 1]
    x_q     = x_norm.mul(levels).round().div(levels)       # quantised in [0, 1]
    x_deq   = x_q * scale + x_min                          # back to original range
    # STE: use quantised value in forward, but gradient flows through x unchanged
    return x + (x_deq - x).detach()


def save_latent_uint8(latent_np: np.ndarray, path: str) -> int:
    """Save a (1, C, H, W) float32 latent as per-channel uint8 .npz.

    Returns the file size in bytes.
    """
    data = latent_np.squeeze(0)           # (C, H, W)
    C = data.shape[0]
    lo = data.reshape(C, -1).min(axis=1)  # (C,)
    hi = data.reshape(C, -1).max(axis=1)
    sc = np.maximum(hi - lo, 1e-6)
    norm = (data - lo[:, None, None]) / sc[:, None, None]
    u8   = (norm * 255.0).round().clip(0, 255).astype(np.uint8)
    np.savez_compressed(path, data=u8,
                        lo=lo.astype(np.float32),
                        hi=hi.astype(np.float32))
    import os
    return os.path.getsize(path if path.endswith('.npz') else path + '.npz')


# ---------------------------------------------------------------
# Two-Scale Latent Texture encoder
# ---------------------------------------------------------------
class LatentTexture(nn.Module):
    """Dense 2D learnable feature grid(s) sampled with bilinear interpolation.

    Two-scale mode (ch_hi > 0):
        Lo-res grid  captures global color / structure (large receptive field per cell)
        Hi-res grid  captures fine-grained edges and texture
        Features from both grids are concatenated → out_dim = ch_lo + ch_hi

    Single-scale mode (ch_hi = 0, default):
        Behaves exactly like the original LatentTexture — backward compatible.

    QAT mode (qat_bits > 0):
        From step qat_start onward the latent tensor is fake-quantised with STE
        before F.grid_sample, so the SIREN decoder learns to tolerate discrete
        integer values.  At save time the latent is stored as real uint8.

    Args:
        H, W:       Image height and width
        ch_lo:      Lo-res feature channels (default 32)
        scale_lo:   Lo-res spatial downscale (default 8 → 64×64 for 512px image)
        ch_hi:      Hi-res feature channels (0 = disabled, default)
        scale_hi:   Hi-res spatial downscale (default 4 → 128×128 for 512px image)
        qat_bits:   Quantisation bit depth (0 = disabled)
        qat_start:  Training step from which QAT is applied
    """
    def __init__(self, H: int, W: int,
                 ch_lo: int = 32, scale_lo: int = 8,
                 ch_hi:  int = 0,  scale_hi:  int = 4,
                 qat_bits: int = 0, qat_start: int = 500):
        super().__init__()
        self.ch_lo     = ch_lo
        self.ch_hi     = ch_hi
        self.qat_bits  = qat_bits
        self.qat_start = qat_start
        self.current_step = 0

        H_lo = max(H // scale_lo, 1)
        W_lo = max(W // scale_lo, 1)
        self.lo = nn.Parameter(torch.randn(1, ch_lo, H_lo, W_lo) * 0.01)

        if ch_hi > 0:
            H_hi = max(H // scale_hi, 1)
            W_hi = max(W // scale_hi, 1)
            self.hi = nn.Parameter(torch.randn(1, ch_hi, H_hi, W_hi) * 0.01)
        else:
            self.hi = None

        self.out_dim = ch_lo + max(ch_hi, 0)

    def set_step(self, step: int):
        self.current_step = step

    def _sample(self, grid: torch.Tensor, coords: torch.Tensor,
                apply_qat: bool) -> torch.Tensor:
        """Sample one grid at coords, optionally fake-quantising first."""
        g = quantize_ste_perchannel(grid, self.qat_bits) if apply_qat else grid
        q = coords.view(1, 1, -1, 2)
        feat = F.grid_sample(g, q, mode='bilinear',
                             padding_mode='border', align_corners=True)
        return feat.squeeze(0).squeeze(1).T    # (N, C)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) in [-1, 1]
        Returns:
            (N, out_dim) concatenated features
        """
        do_qat = (self.qat_bits > 0 and self.current_step >= self.qat_start)
        feat_lo = self._sample(self.lo, coords, do_qat)
        if self.hi is not None:
            feat_hi = self._sample(self.hi, coords, do_qat)
            return torch.cat([feat_lo, feat_hi], dim=-1)
        return feat_lo

    @property
    def lo_shape(self):
        return tuple(self.lo.shape[2:])

    @property
    def hi_shape(self):
        return tuple(self.hi.shape[2:]) if self.hi is not None else None

    @property
    def latent_params(self) -> int:
        n = self.lo.numel()
        if self.hi is not None:
            n += self.hi.numel()
        return n

    def to_rgb_preview(self) -> np.ndarray:
        """Return (H_lo, W_lo, 3) uint8 RGB preview of lo-res grid channels 0-2."""
        with torch.no_grad():
            g = self.lo[0, :3].cpu().numpy()   # (≤3, H, W)
            if g.shape[0] < 3:
                pad = np.zeros((3 - g.shape[0], *g.shape[1:]), dtype=g.dtype)
                g = np.concatenate([g, pad], axis=0)
            g = g.transpose(1, 2, 0)           # (H, W, 3)
            lo, hi = g.min(), g.max()
            g = (g - lo) / max(hi - lo, 1e-6)
        return (np.clip(g, 0, 1) * 255).astype(np.uint8)

    def size_bytes_fp32(self) -> int:
        return self.latent_params * 4

    def size_bytes_fp16(self) -> int:
        return self.latent_params * 2

    def size_bytes_uint8(self) -> int:
        return self.latent_params


# ---------------------------------------------------------------
# Simple MLP layer with a choice of activation (GELU / SiLU / ReLU)
# ---------------------------------------------------------------
_ACTIVATIONS = {
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'relu': nn.ReLU,
}


def _make_mlp_layer(in_dim: int, out_dim: int, act: str) -> nn.Sequential:
    """Linear → BatchNorm-free residual-friendly dense layer."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), _ACTIVATIONS[act]())


# ---------------------------------------------------------------
# Full model: LatentTexture → MLP decoder → RGB
# ---------------------------------------------------------------
class LatentImageMLP(nn.Module):
    """Coordinate → RGB model: two-scale latent encoder + MLP decoder.

    Decoder activation is configurable:
      - 'siren'  — SirenLayer (legacy, kept for orthogonal encoder comparison)
      - 'gelu'   — GELU MLP  (recommended for latent route; smoother, faster)
      - 'silu'   — SiLU MLP  (similar to GELU, slightly sharper gradients)
      - 'relu'   — ReLU MLP

    Note on activation choice: despite bilinear grid_sample producing spatially
    smooth features, SIREN outperforms GELU/SiLU by ~5 dB on the same budget.
    Sinusoidal activations provide a much richer non-linear basis for the 18D→RGB
    mapping, compensating for the latent's lack of multi-scale structure.
    Use --activation gelu/silu for experiments or speed-sensitive deployments.
    """
    def __init__(self, H: int, W: int,
                 ch_lo: int = 32, scale_lo: int = 8,
                 ch_hi:  int = 0,  scale_hi:  int = 4,
                 hidden: int = 64, depth: int = 2,
                 omega_0: float = 30.0,
                 activation: str = 'siren',
                 qat_bits: int = 0, qat_start: int = 500):
        super().__init__()
        self.activation = activation.lower()
        self.encoder = LatentTexture(H, W,
                                     ch_lo=ch_lo, scale_lo=scale_lo,
                                     ch_hi=ch_hi,  scale_hi=scale_hi,
                                     qat_bits=qat_bits, qat_start=qat_start)
        in_dim = self.encoder.out_dim

        if self.activation == 'siren':
            layers = [SirenLayer(in_dim, hidden, is_first=True, omega_0=omega_0)]
            for _ in range(depth - 1):
                layers.append(SirenLayer(hidden, hidden, is_first=False, omega_0=omega_0))
            self.net = nn.Sequential(*layers)
        else:
            if self.activation not in _ACTIVATIONS:
                raise ValueError(f"Unknown activation '{activation}'. "
                                 f"Choose from: siren, {', '.join(_ACTIVATIONS)}")
            layers = [_make_mlp_layer(in_dim, hidden, self.activation)]
            for _ in range(depth - 1):
                layers.append(_make_mlp_layer(hidden, hidden, self.activation))
            self.net = nn.Sequential(*layers)

        self.head = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(self.encoder(coords)))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def size_bytes(self) -> int:
        return self.param_count * 4


# ---------------------------------------------------------------
# JPEG compression estimate for latent grids
# ---------------------------------------------------------------
def estimate_jpeg_size(latent_np: np.ndarray, quality: int = 85) -> int:
    """Encode each 3-channel slice of latent as JPEG; return total bytes."""
    from PIL import Image
    import io
    flat = latent_np.squeeze(0)   # (C, H, W)
    total = 0
    for i in range(0, flat.shape[0], 3):
        chunk = flat[i:i + 3]
        if chunk.shape[0] < 3:
            pad = np.zeros((3 - chunk.shape[0], *chunk.shape[1:]), dtype=flat.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)
        rgb = chunk.transpose(1, 2, 0)
        lo, hi = rgb.min(), rgb.max()
        u8 = ((rgb - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, format='JPEG', quality=quality)
        total += buf.tell()
    return total


def estimate_jpeg_size_uint8(latent_np: np.ndarray, quality: int = 85) -> int:
    """Same as estimate_jpeg_size but input is already uint8 (QAT-quantised)."""
    from PIL import Image
    import io
    flat = latent_np.squeeze(0) if latent_np.ndim == 4 else latent_np  # (C,H,W)
    total = 0
    for i in range(0, flat.shape[0], 3):
        chunk = flat[i:i + 3]
        if chunk.shape[0] < 3:
            pad = np.zeros((3 - chunk.shape[0], *chunk.shape[1:]), dtype=np.uint8)
            chunk = np.concatenate([chunk, pad], axis=0)
        buf = io.BytesIO()
        Image.fromarray(chunk.transpose(1, 2, 0)).save(buf, format='JPEG', quality=quality)
        total += buf.tell()
    return total


# ---------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------
def make_display_cv2(orig_bgr, recon_bgr, status_line1, status_line2,
                     latent_preview=None):
    H, W = orig_bgr.shape[:2]
    sep = np.full((H, 12, 3), 30, dtype=np.uint8)
    panels = [orig_bgr, sep, recon_bgr]

    if latent_preview is not None:
        lat_bgr = cv2.cvtColor(latent_preview, cv2.COLOR_RGB2BGR)
        lat_bgr = cv2.resize(lat_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
        panels += [sep, lat_bgr]

    canvas = np.hstack(panels)
    W_c = canvas.shape[1]

    header = np.full((32, W_c, 3), 30, dtype=np.uint8)
    cv2.putText(header, "Original",
                (W // 2 - 40, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, "Reconstruction",
                (W + 12 + W // 2 - 65, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 200), 1, cv2.LINE_AA)
    if latent_preview is not None:
        cv2.putText(header, "Latent lo-res (ch 0-2)",
                    (2 * W + 24 + W // 2 - 95, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1, cv2.LINE_AA)

    bar = np.full((52, W_c, 3), 30, dtype=np.uint8)
    cv2.putText(bar, status_line1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, status_line2, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    return np.vstack([header, canvas, bar])


def make_display_mpl(fig, axes, img_np, recon_np, latent_preview, title):
    n = len(axes)
    axes[0].clear(); axes[0].imshow(img_np)
    axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].clear(); axes[1].imshow(np.clip(recon_np, 0, 1))
    axes[1].set_title("Reconstruction", color='teal'); axes[1].axis('off')
    if n > 2 and latent_preview is not None:
        axes[2].clear(); axes[2].imshow(latent_preview)
        axes[2].set_title("Latent lo ch 0-2", color='orange'); axes[2].axis('off')
    fig.suptitle(title, fontsize=9, fontfamily='monospace')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="tinyLatent — two-scale latent texture + SIREN neural compression")
    parser.add_argument("-i", "--input", default=str(local_dir / "sample.png"))

    # ── Lo-res grid ──────────────────────────────────────────────────────────
    parser.add_argument("--scale_lo", "--scale", type=int, default=8,
                        dest="scale_lo",
                        help="Lo-res grid downscale factor (default: 8)")
    parser.add_argument("--ch_lo", "--channels", type=int, default=32,
                        dest="ch_lo",
                        help="Lo-res feature channels (default: 32)")

    # ── Hi-res grid (optional) ───────────────────────────────────────────────
    parser.add_argument("--scale_hi", type=int, default=4,
                        help="Hi-res grid downscale factor (default: 4)")
    parser.add_argument("--ch_hi", type=int, default=0,
                        help="Hi-res feature channels; 0 = disabled (default: 0)")

    # ── QAT ──────────────────────────────────────────────────────────────────
    parser.add_argument("--qat_bits", type=int, default=0,
                        help="Latent quantisation bit-depth; 0 = disabled (default: 0)")
    parser.add_argument("--qat_start", type=int, default=500,
                        help="Step from which QAT is applied (default: 500)")

    # ── Decoder ───────────────────────────────────────────────────────────────
    parser.add_argument("--hidden",     type=int,   default=64)
    parser.add_argument("--depth",      type=int,   default=2)
    parser.add_argument("--omega_0",    type=float, default=30.0,
                        help="SIREN omega_0 (only used when --activation siren)")
    parser.add_argument("--activation", type=str,   default='siren',
                        choices=['siren', 'gelu', 'silu', 'relu'],
                        help="Decoder activation (default: siren). "
                             "SIREN outperforms GELU/SiLU by ~5 dB on the latent route — "
                             "sinusoidal activations provide a richer non-linear basis "
                             "for the 18D→RGB mapping even with smooth latent inputs.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--lr",    type=float, default=3e-3)
    parser.add_argument("--iters", type=int,   default=5000)
    parser.add_argument("--batch", type=int,   default=2 ** 16)

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--save",        type=str, default=None,
                        help="Save full model weights (.pth)")
    parser.add_argument("--save_latent", type=str, default=None,
                        help="Save latent as fp16 .npy (no QAT) or uint8 .npz (QAT)")
    parser.add_argument("--vis_latent", action="store_true",
                        help="Show lo-res latent grid as 3rd display panel")
    args = parser.parse_args()

    # ── Load image ────────────────────────────────────────────────────────────
    from PIL import Image as PILImage
    img_pil = PILImage.open(args.input).convert('RGB')
    img_np  = np.array(img_pil, dtype=np.float32) / 255.0
    H, W, _ = img_np.shape
    img_bytes = H * W * 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentImageMLP(
        H=H, W=W,
        ch_lo=args.ch_lo, scale_lo=args.scale_lo,
        ch_hi=args.ch_hi,  scale_hi=args.scale_hi,
        hidden=args.hidden, depth=args.depth, omega_0=args.omega_0,
        activation=args.activation,
        qat_bits=args.qat_bits, qat_start=args.qat_start,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters)

    # ── Stats ─────────────────────────────────────────────────────────────────
    enc  = model.encoder
    lo_p = enc.lo.numel()
    hi_p = enc.hi.numel() if enc.hi is not None else 0
    dec_p = model.param_count - enc.latent_params
    ratio = img_bytes / model.size_bytes

    lo_sh = enc.lo_shape
    hi_sh = enc.hi_shape

    print(f"Image          : {W}×{H}  ({img_bytes:,} bytes)")
    print(f"Lo-res grid    : {args.ch_lo}×{lo_sh[1]}×{lo_sh[0]}  "
          f"({lo_p:,} params, {lo_p*4//1024} KB fp32)")
    if hi_p:
        print(f"Hi-res grid    : {args.ch_hi}×{hi_sh[1]}×{hi_sh[0]}  "
              f"({hi_p:,} params, {hi_p*4//1024} KB fp32)")
    act_label = args.activation.upper()
    print(f"{act_label} decoder   : {dec_p:,} params  ({dec_p*4//1024} KB)")
    print(f"Total params   : {model.param_count:,}  "
          f"({model.size_bytes//1024} KB fp32)")
    print(f"Compression    : {ratio:.2f}x  (raw fp32)")

    qat_label = (f"QAT int{args.qat_bits} from step {args.qat_start}"
                 if args.qat_bits else "QAT disabled")
    lo_np = enc.lo.detach().cpu().numpy()
    jpeg_fp32 = estimate_jpeg_size(lo_np)
    decoder_fp16_bytes = dec_p * 2
    two_stage = img_bytes / (jpeg_fp32 + decoder_fp16_bytes)
    print(f"Two-stage est. : ~{two_stage:.1f}x  (lo-grid JPEG Q=85 + fp16 decoder)")
    if args.qat_bits:
        lo_u8 = (lo_np - lo_np.min(axis=(2, 3), keepdims=True))
        lo_u8 = lo_u8 / np.maximum(lo_u8.max(axis=(2, 3), keepdims=True), 1e-6)
        jpeg_u8 = estimate_jpeg_size_uint8((lo_u8 * 255).astype(np.uint8))
        two_stage_u8 = img_bytes / (jpeg_u8 + decoder_fp16_bytes)
        print(f"Two-stage uint8: ~{two_stage_u8:.1f}x  (lo-grid uint8 JPEG Q=85 + fp16 decoder)")
    print(f"Device         : {device}")
    print(f"{qat_label}")
    print(f"\nTraining {args.iters} steps...  (Q/ESC quit, Space pause, S save)\n")

    # ── Display setup ─────────────────────────────────────────────────────────
    n_panels = 3 if args.vis_latent else 2
    if HAS_CV2:
        win = f"tinyLatent — Two-Scale Latent + {args.activation.upper()}"
        win_w = min(W * n_panels + 12 * (n_panels - 1), 1800)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, win_w, min(H + 84, 900))
        orig_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        plt.subplots_adjust(top=0.88)

    # ── Training loop ─────────────────────────────────────────────────────────
    paused    = False
    t0        = time.time()
    psnr      = 0.0
    recon_np  = np.zeros_like(img_np)
    step      = 0

    while step < args.iters:
        if HAS_CV2:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord(' '):
                paused = not paused
                if paused:
                    print("\n  [PAUSED — press Space to resume]", end="", flush=True)
            if key == ord('s'):
                path = f"tinyLatent_step{step}.png"
                cv2.imwrite(path, cv2.cvtColor(
                    (np.clip(recon_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR))
                print(f"\n  Saved: {path}", end="", flush=True)

        if paused:
            time.sleep(0.05)
            continue

        step += 1
        model.encoder.set_step(step)

        # ── Train step ────────────────────────────────────────────────────────
        model.train()
        idx  = torch.randint(0, N, (min(args.batch, N),), device=device)
        pred = model(all_coords[idx])
        loss = ((pred - all_colors[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── Display / log update ─────────────────────────────────────────────
        if (step % 50 == 0) or (step <= 10) or (step == args.iters):
            recon_np = render_full(model, all_coords, H, W)
            mse  = np.mean((img_np - recon_np) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            elapsed = time.time() - t0

            qat_active = args.qat_bits > 0 and step >= args.qat_start
            qat_str = f"QAT int{args.qat_bits}" if qat_active else ""

            line1 = (f"Step {step}/{args.iters}  |  "
                     f"Loss {loss.item():.5f}  |  "
                     f"PSNR {psnr:.2f} dB  |  "
                     f"Ratio {ratio:.1f}x  |  {elapsed:.1f}s"
                     + (f"  |  {qat_str}" if qat_str else ""))

            lo_dims = f"Lo {args.ch_lo}×{lo_sh[1]}×{lo_sh[0]} {lo_p*4//1024}KB"
            hi_dims = (f"  Hi {args.ch_hi}×{hi_sh[1]}×{hi_sh[0]} {hi_p*4//1024}KB"
                       if hi_p else "")
            line2 = (lo_dims + hi_dims +
                     f"  |  Dec {dec_p*4//1024}KB"
                     f"  |  LR {scheduler.get_last_lr()[0]:.1e}"
                     + (f"  |  {qat_str}" if qat_str else ""))

            print(f"\r{line1}", end="", flush=True)
            latent_preview = (model.encoder.to_rgb_preview()
                              if args.vis_latent else None)

            if HAS_CV2:
                recon_bgr = cv2.cvtColor(
                    (np.clip(recon_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                cv2.imshow(win, make_display_cv2(
                    orig_bgr, recon_bgr, line1, line2, latent_preview))
            else:
                make_display_mpl(fig, axes, img_np, recon_np,
                                 latent_preview, line1)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s — Final PSNR: {psnr:.2f} dB")

    if args.save:
        torch.save(model.state_dict(), args.save)
        import os
        sz = os.path.getsize(args.save)
        print(f"Model saved : {args.save}  ({sz:,} bytes, {img_bytes/sz:.1f}x)")

    if args.save_latent:
        if args.qat_bits:
            # Save as per-channel uint8 .npz
            lo_np_final = enc.lo.detach().cpu().numpy()
            sz = save_latent_uint8(lo_np_final, args.save_latent)
            jpeg_sz = estimate_jpeg_size_uint8(
                ((lo_np_final - lo_np_final.min(axis=(2, 3), keepdims=True)) /
                 np.maximum(lo_np_final.max(axis=(2, 3), keepdims=True) -
                            lo_np_final.min(axis=(2, 3), keepdims=True), 1e-6)
                 * 255).astype(np.uint8))
            print(f"Latent saved (uint8): {args.save_latent}  "
                  f"({sz:,} bytes, {img_bytes/sz:.1f}x)")
            print(f"JPEG of uint8 latent: {jpeg_sz:,} bytes  "
                  f"({img_bytes/jpeg_sz:.1f}x vs raw, Q=85)")
        else:
            lo_np_final = enc.lo.detach().cpu().numpy().astype(np.float16)
            np.save(args.save_latent, lo_np_final)
            import os
            sz = os.path.getsize(args.save_latent)
            jpeg_sz = estimate_jpeg_size(enc.lo.detach().cpu().numpy())
            print(f"Latent saved (fp16) : {args.save_latent}  "
                  f"({sz:,} bytes, {img_bytes/sz:.1f}x)")
            print(f"JPEG of fp32 latent : {jpeg_sz:,} bytes  "
                  f"({img_bytes/jpeg_sz:.1f}x vs raw, Q=85)")

    if HAS_CV2:
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
