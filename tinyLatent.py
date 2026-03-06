"""Neural image compression via latent texture + SIREN decoder.

Learns a dense 2D feature grid (the "latent texture") at reduced resolution,
sampled with bilinear interpolation, then decoded by a SIREN MLP to RGB.

  Compressed artifact = the latent grid weights (a small 2D float array)
                        + the tiny SIREN decoder weights
  Two-stage potential  = latent grid can be further compressed with JPEG/PNG

Architecture:
    (x, y) → [F.grid_sample on C×(H/s)×(W/s) grid] → C features
           → [SirenLayer × depth] → Linear + Sigmoid → (r, g, b)

Orthogonal comparison with tinyMLP.py (Hash Encoding):
    - Decoder is IDENTICAL (same SirenLayer / hidden / depth / omega_0)
    - Encoder swapped: dense latent grid vs sparse hash tables
    - out_dim = channels = 32 in both default configs → same decoder input

Usage:
    python tinyLatent.py                             # compress sample.png
    python tinyLatent.py -i photo.jpg                # custom input
    python tinyLatent.py --scale 4 --channels 16     # finer grid, fewer channels
    python tinyLatent.py --vis_latent                # show latent evolution panel
    python tinyLatent.py --save_latent latent.npy    # save latent grid as float16

Controls:
    ESC / Q  — quit
    Space    — pause / resume training
    S        — save current reconstruction as PNG
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

# Import the shared SIREN layer and render helper from tinyMLP
import sys
sys.path.insert(0, str(local_dir))
from tinyMLP import SirenLayer, render_full


# ---------------------------------------------------------------
# Latent Texture encoder
# ---------------------------------------------------------------
class LatentTexture(nn.Module):
    """Dense 2D learnable feature grid, sampled with bilinear interpolation.

    Unlike sparse hash tables, every parameter occupies an explicit spatial
    position in a regular grid.  This means:
      - Spatial coherence is enforced by construction.
      - The grid can be saved as a standard float16 array and further
        compressed with image codecs (JPEG, PNG, etc.).
      - Coarser grids trade reconstruction quality for compression ratio.

    Args:
        H, W:      Image height and width (used to size the latent grid)
        scale:     Spatial downscale factor — grid size = (H//scale, W//scale)
        channels:  Number of feature channels C (= out_dim fed to decoder)
    """
    def __init__(self, H, W, scale=8, channels=32):
        super().__init__()
        self.scale = scale
        self.channels = channels
        H_lat = max(H // scale, 1)
        W_lat = max(W // scale, 1)
        # Small random init: values near 0 so SIREN starts from sin(~0)
        self.latent = nn.Parameter(
            torch.randn(1, channels, H_lat, W_lat) * 0.01
        )
        self.out_dim = channels

    @property
    def grid_shape(self):
        return tuple(self.latent.shape[2:])

    def forward(self, coords):
        """
        Args:
            coords: (N, 2) in [-1, 1]
        Returns:
            (N, channels) bilinearly interpolated features
        """
        # grid_sample expects (N, H_out, W_out, 2); we use H_out=1, W_out=N
        grid = coords.view(1, 1, -1, 2)          # (1, 1, N, 2)
        feat = F.grid_sample(
            self.latent, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )                                          # (1, C, 1, N)
        return feat.squeeze(0).squeeze(1).T        # (N, C)

    def to_rgb_preview(self):
        """Return a (H_lat, W_lat, 3) uint8 image of the first 3 channels."""
        with torch.no_grad():
            grid = self.latent[0, :3].cpu().numpy()   # (3, H, W)
            grid = grid.transpose(1, 2, 0)             # (H, W, 3)
            lo, hi = grid.min(), grid.max()
            if hi > lo:
                grid = (grid - lo) / (hi - lo)
            else:
                grid = np.zeros_like(grid)
        return (np.clip(grid, 0, 1) * 255).astype(np.uint8)

    def size_bytes(self):
        return self.latent.numel() * 4

    def size_bytes_fp16(self):
        return self.latent.numel() * 2


# ---------------------------------------------------------------
# Full model: LatentTexture → SIREN decoder → RGB
# ---------------------------------------------------------------
class LatentImageMLP(nn.Module):
    """Coordinate-to-color MLP with latent texture encoder and SIREN decoder.

    The decoder is structurally identical to ImageMLP in tinyMLP.py so that
    encoder quality can be compared directly (orthogonal comparison).

    Args:
        H, W:      Image size (determines latent grid resolution)
        scale:     Latent grid downscale factor (default 8)
        channels:  Latent feature channels — also sets decoder input dim (default 32)
        hidden:    SIREN hidden width (default 64)
        depth:     SIREN hidden layer count (default 2)
        omega_0:   SIREN frequency multiplier (default 30)
    """
    def __init__(self, H, W, scale=8, channels=32,
                 hidden=64, depth=2, omega_0=30.0):
        super().__init__()
        self.encoder = LatentTexture(H, W, scale, channels)
        in_dim = self.encoder.out_dim

        layers = [SirenLayer(in_dim, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(depth - 1):
            layers.append(SirenLayer(hidden, hidden, is_first=False, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, coords):
        return self.head(self.net(self.encoder(coords)))

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_bytes(self):
        return self.param_count * 4


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
    W_canvas = canvas.shape[1]

    header = np.full((32, W_canvas, 3), 30, dtype=np.uint8)
    cv2.putText(header, "Original", (W // 2 - 40, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, "Reconstruction", (W + 12 + W // 2 - 65, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 200), 1, cv2.LINE_AA)
    if latent_preview is not None:
        cv2.putText(header, "Latent (ch 0-2)", (2 * W + 24 + W // 2 - 72, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1, cv2.LINE_AA)

    bar = np.full((52, W_canvas, 3), 30, dtype=np.uint8)
    cv2.putText(bar, status_line1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, status_line2, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    return np.vstack([header, canvas, bar])


def make_display_mpl(fig, axes, img_np, recon_np, latent_preview, title):
    n = len(axes)
    axes[0].clear(); axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].clear(); axes[1].imshow(np.clip(recon_np, 0, 1))
    axes[1].set_title("Reconstruction", color='teal'); axes[1].axis('off')
    if n > 2 and latent_preview is not None:
        axes[2].clear(); axes[2].imshow(latent_preview)
        axes[2].set_title("Latent ch 0-2", color='orange'); axes[2].axis('off')
    fig.suptitle(title, fontsize=9, fontfamily='monospace')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ---------------------------------------------------------------
# Estimate compressed latent size using JPEG
# ---------------------------------------------------------------
def estimate_jpeg_size(latent_np_fp32, quality=85):
    """Normalise latent to [0,255] uint8 and measure JPEG byte size."""
    from PIL import Image
    import io
    flat = latent_np_fp32.squeeze(0)          # (C, H, W)
    pages = []
    for i in range(0, flat.shape[0], 3):
        chunk = flat[i:i + 3]
        if chunk.shape[0] < 3:
            pad = np.zeros((3 - chunk.shape[0], *chunk.shape[1:]), dtype=flat.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)
        rgb = chunk.transpose(1, 2, 0)
        lo, hi = rgb.min(), rgb.max()
        rgb_u8 = ((rgb - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb_u8).save(buf, format='JPEG', quality=quality)
        pages.append(buf.tell())
    return sum(pages)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="tinyLatent — latent texture neural image compression")
    parser.add_argument("-i", "--input", default=str(local_dir / "sample.png"))
    # Latent grid
    parser.add_argument("--scale", type=int, default=8,
                        help="spatial downscale factor for latent grid (default: 8)")
    parser.add_argument("--channels", type=int, default=32,
                        help="feature channels in latent grid (default: 32)")
    # SIREN decoder
    parser.add_argument("--hidden", type=int, default=64,
                        help="SIREN hidden width (default: 64)")
    parser.add_argument("--depth", type=int, default=2,
                        help="SIREN hidden layers (default: 2)")
    parser.add_argument("--omega_0", type=float, default=30.0)
    # Training
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Adam learning rate (default: 3e-3)")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2 ** 16)
    # Output
    parser.add_argument("--save", type=str, default=None,
                        help="save full model weights (.pth)")
    parser.add_argument("--save_latent", type=str, default=None,
                        help="save latent grid as float16 .npy (the compressed artifact)")
    parser.add_argument("--vis_latent", action="store_true",
                        help="show latent grid evolution as 3rd display panel")
    args = parser.parse_args()

    # --- Load image ---
    from PIL import Image as PILImage
    img_pil = PILImage.open(args.input).convert('RGB')
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    H, W, _ = img_np.shape
    img_bytes = H * W * 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Coordinate grid ---
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    # --- Model ---
    model = LatentImageMLP(
        H=H, W=W, scale=args.scale, channels=args.channels,
        hidden=args.hidden, depth=args.depth, omega_0=args.omega_0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

    H_lat, W_lat = model.encoder.grid_shape
    latent_params = model.encoder.latent.numel()
    decoder_params = model.param_count - latent_params
    ratio = img_bytes / model.size_bytes
    jpeg_est = estimate_jpeg_size(
        model.encoder.latent.detach().cpu().numpy())
    two_stage_ratio = img_bytes / (jpeg_est + decoder_params * 2)

    print(f"Image          : {W}x{H}  ({img_bytes:,} bytes)")
    print(f"Latent grid    : {args.channels}×{W_lat}×{H_lat}  "
          f"({latent_params:,} params, {latent_params*4//1024} KB fp32, "
          f"{latent_params*2//1024} KB fp16)")
    print(f"SIREN decoder  : {decoder_params:,} params  ({decoder_params*4//1024} KB)")
    print(f"Total params   : {model.param_count:,}  ({model.size_bytes//1024} KB)")
    print(f"Compression    : {ratio:.2f}x  (raw fp32)")
    print(f"Two-stage est. : ~{two_stage_ratio:.1f}x  "
          f"(latent JPEG Q=85 + fp16 decoder)")
    print(f"Device         : {device}")
    print(f"\nTraining {args.iters} steps...  (Q/ESC to quit, Space to pause)\n")

    # --- Display setup ---
    n_panels = 3 if args.vis_latent else 2
    if HAS_CV2:
        win = "tinyLatent — Latent Texture + SIREN"
        win_w = min(W * n_panels + 12 * (n_panels - 1), 1800)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, win_w, min(H + 84, 900))
        orig_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        plt.subplots_adjust(top=0.88)

    # --- Training loop ---
    paused = False
    t0 = time.time()
    psnr = 0.0
    recon_np = np.zeros_like(img_np)
    step = 0

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

        # --- Train step ---
        model.train()
        idx = torch.randint(0, N, (min(args.batch, N),), device=device)
        pred = model(all_coords[idx])
        loss = ((pred - all_colors[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Display update ---
        if (step % 50 == 0) or (step <= 10) or (step == args.iters):
            recon_np = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon_np) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            elapsed = time.time() - t0

            line1 = (f"Step {step}/{args.iters}  |  "
                     f"Loss {loss.item():.5f}  |  "
                     f"PSNR {psnr:.2f} dB  |  "
                     f"Ratio {ratio:.1f}x  |  {elapsed:.1f}s")
            line2 = (f"Latent {args.channels}×{W_lat}×{H_lat}  "
                     f"{latent_params*4//1024}KB  |  "
                     f"Decoder {decoder_params*4//1024}KB  |  "
                     f"LR {scheduler.get_last_lr()[0]:.1e}")

            print(f"\r{line1}", end="", flush=True)
            latent_preview = model.encoder.to_rgb_preview() if args.vis_latent else None

            if HAS_CV2:
                recon_bgr = cv2.cvtColor(
                    (np.clip(recon_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                cv2.imshow(win, make_display_cv2(orig_bgr, recon_bgr, line1, line2,
                                                  latent_preview))
            else:
                make_display_mpl(fig, axes, img_np, recon_np, latent_preview, line1)

    # --- Done ---
    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s — Final PSNR: {psnr:.2f} dB")

    if args.save:
        torch.save(model.state_dict(), args.save)
        import os
        sz = os.path.getsize(args.save)
        print(f"Model saved : {args.save}  ({sz:,} bytes, {img_bytes/sz:.1f}x)")

    if args.save_latent:
        lat_fp16 = model.encoder.latent.detach().cpu().numpy().astype(np.float16)
        np.save(args.save_latent, lat_fp16)
        import os
        sz = os.path.getsize(args.save_latent)
        print(f"Latent saved: {args.save_latent}  ({sz:,} bytes fp16, "
              f"{img_bytes/sz:.1f}x vs raw image)")
        jpeg_final = estimate_jpeg_size(
            model.encoder.latent.detach().cpu().numpy())
        print(f"JPEG estimate of latent: {jpeg_final:,} bytes  "
              f"({img_bytes/jpeg_final:.1f}x vs raw, Q=85)")

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
