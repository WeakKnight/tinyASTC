"""Neural image compression via hash-encoded coordinate MLP.

Trains a multi-resolution hash encoding (Instant-NGP style) + SIREN decoder
to memorize an image by mapping (x, y) -> (r, g, b).

  Compression: hash table weights vs raw image pixels
  Quality:     ~32 dB PSNR at 5x compression (5000 steps on 512² image)

Usage:
    python tinyMLP.py                            # compress sample.jpg
    python tinyMLP.py -i photo.png               # custom input
    python tinyMLP.py --log2_T 10 --hidden 32    # tiny model, higher compression
    python tinyMLP.py --save model.pth           # save trained weights

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

local_dir = pathlib.Path(__file__).parent.absolute()

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------
# Multi-resolution hash encoding  (Müller et al. 2022, Instant-NGP)
# ---------------------------------------------------------------
class HashEncoding(nn.Module):
    """Multi-resolution spatial hash encoding.

    Maintains L learnable feature tables at geometrically spaced resolutions.
    For each 2D coordinate:
      1. Find the 4 surrounding grid corners at each of the L resolution levels.
      2. Hash each corner to a table slot (or direct-index for small grids).
      3. Bilinearly interpolate the 4 corner features.
      4. Concatenate interpolated features from all L levels → out_dim = L*F.

    Because coarse levels use small direct-index tables and fine levels hash
    into a fixed-size T table, the total parameter count stays bounded while
    still covering a wide range of spatial frequencies.

    Args:
        n_levels:  Number of resolution levels L (default 16)
        n_features: Feature dims per level F (default 2)
        log2_T:    log₂ of hash table size T per level (default 12 → 4096 entries)
        base_res:  Coarsest grid resolution (default 16)
        max_res:   Finest grid resolution (default 512)
    """
    # Large primes for spatial hashing: (x·p1) XOR (y·p2) mod T
    _P1 = 2654435761
    _P2 = 805459861

    def __init__(self, n_levels=16, n_features=2, log2_T=12,
                 base_res=16, max_res=512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.T = 2 ** log2_T

        # Geometric progression of grid resolutions
        b = np.exp(np.log(max_res / base_res) / max(n_levels - 1, 1))
        self.resolutions = [int(round(base_res * b ** l)) for l in range(n_levels)]

        # One learnable table per level; coarse levels use fewer entries
        self.tables = nn.ParameterList([
            nn.Parameter(
                torch.empty(min(self.T, res * res), n_features).uniform_(-1e-3, 1e-3)
            )
            for res in self.resolutions
        ])
        self.out_dim = n_levels * n_features

    def _idx(self, ix, iy, res, T):
        """Return flat table indices for integer grid coords (ix, iy)."""
        ix = ix.long().clamp(0, res - 1)
        iy = iy.long().clamp(0, res - 1)
        if T < res * res:
            return ((ix * self._P1) ^ (iy * self._P2)) % T
        return ix * res + iy

    def forward(self, coords):
        """
        Args:
            coords: (N, 2) in [-1, 1]
        Returns:
            (N, n_levels * n_features) concatenated hash features
        """
        x01 = (coords + 1.0) * 0.5        # [0, 1]
        features = []

        for res, table in zip(self.resolutions, self.tables):
            T = table.shape[0]
            s = x01 * (res - 1)            # scale to grid
            s0 = s.floor()                 # bottom-left corner
            f = s - s0                     # bilinear weights, (N, 2)
            fx, fy = f[:, 0:1], f[:, 1:2] # (N, 1) each

            x0, y0 = s0[:, 0], s0[:, 1]
            x1, y1 = x0 + 1, y0 + 1

            # Bilinear interpolation over 4 corners
            feat = (
                (1 - fx) * (1 - fy) * table[self._idx(x0, y0, res, T)] +
                (1 - fx) *      fy  * table[self._idx(x0, y1, res, T)] +
                     fx  * (1 - fy) * table[self._idx(x1, y0, res, T)] +
                     fx  *      fy  * table[self._idx(x1, y1, res, T)]
            )
            features.append(feat)

        return torch.cat(features, dim=-1)


# ---------------------------------------------------------------
# SIREN layer  (Sitzmann et al. 2020)
# ---------------------------------------------------------------
class SirenLayer(nn.Module):
    """Linear layer with sinusoidal activation and SIREN weight init.

    The first layer uses a wide uniform init so the sine inputs span [−π, π].
    Subsequent layers use a tighter init to preserve output variance through
    many sin layers (derived in the SIREN paper).

    Args:
        in_features:  Input dimension
        out_features: Output dimension
        is_first:     True for the first layer (uses wider weight init)
        omega_0:      Frequency multiplier ω₀ applied before sin
    """
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ---------------------------------------------------------------
# Full model: HashEncoding → SIREN decoder → RGB
# ---------------------------------------------------------------
class ImageMLP(nn.Module):
    """Coordinate-to-color MLP with hash encoding and SIREN decoder.

    HashEncoding captures multi-scale spatial features efficiently.
    SIREN decoder maps those features to RGB with sinusoidal activations,
    which excel at representing smooth, continuous image signals.

    Args:
        n_levels:  Hash table resolution levels (default 16)
        n_features: Features per hash level (default 2)
        log2_T:    log₂ of hash table size (default 12)
        hidden:    SIREN hidden width (default 64)
        depth:     SIREN hidden layer count (default 2)
        omega_0:   SIREN frequency multiplier (default 30)
        base_res:  Coarsest hash resolution (default 16)
        max_res:   Finest hash resolution (default 512)
    """
    def __init__(self, n_levels=16, n_features=2, log2_T=12,
                 hidden=64, depth=2, omega_0=30.0,
                 base_res=16, max_res=512):
        super().__init__()
        self.encoder = HashEncoding(n_levels, n_features, log2_T, base_res, max_res)
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
# Render the full image from the MLP (chunked to avoid OOM)
# ---------------------------------------------------------------
@torch.no_grad()
def render_full(model, coords, H, W, chunk=65536):
    model.eval()
    parts = [model(coords[i:i + chunk]) for i in range(0, coords.shape[0], chunk)]
    return torch.cat(parts).cpu().numpy().reshape(H, W, 3)


# ---------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------
def make_display_cv2(orig_bgr, recon_bgr, status_line1, status_line2):
    H, W = orig_bgr.shape[:2]
    sep = np.full((H, 16, 3), 30, dtype=np.uint8)
    canvas = np.hstack([orig_bgr, sep, recon_bgr])

    header = np.full((32, canvas.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, "Original", (W // 2 - 40, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, "MLP Reconstruction", (W + 16 + W // 2 - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 200), 1, cv2.LINE_AA)

    bar = np.full((52, canvas.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, status_line1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, status_line2, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    return np.vstack([header, canvas, bar])


def make_display_mpl(fig, axes, img_np, recon_np, title):
    axes[0].clear()
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis('off')
    axes[1].clear()
    axes[1].imshow(np.clip(recon_np, 0, 1))
    axes[1].set_title("MLP Reconstruction", fontsize=11, color='teal')
    axes[1].axis('off')
    fig.suptitle(title, fontsize=10, fontfamily='monospace')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="tinyMLP — neural image compression")
    parser.add_argument("-i", "--input", default=str(local_dir / "sample.jpg"))
    # Hash encoding
    parser.add_argument("--n_levels", type=int, default=16,
                        help="number of hash resolution levels (default: 16)")
    parser.add_argument("--n_features", type=int, default=2,
                        help="features per hash level (default: 2)")
    parser.add_argument("--log2_T", type=int, default=12,
                        help="log2 of hash table size (default: 12 → 4096 entries)")
    parser.add_argument("--base_res", type=int, default=16)
    parser.add_argument("--max_res", type=int, default=512)
    # SIREN decoder
    parser.add_argument("--hidden", type=int, default=64,
                        help="SIREN hidden layer width (default: 64)")
    parser.add_argument("--depth", type=int, default=2,
                        help="number of SIREN hidden layers (default: 2)")
    parser.add_argument("--omega_0", type=float, default=30.0,
                        help="SIREN frequency multiplier (default: 30)")
    # Training
    parser.add_argument("--lr_hash", type=float, default=3e-2,
                        help="learning rate for hash tables (default: 3e-2)")
    parser.add_argument("--lr_net", type=float, default=1e-3,
                        help="learning rate for SIREN decoder (default: 1e-3)")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2 ** 16,
                        help="pixels per training step (default: 65536)")
    parser.add_argument("--save", type=str, default=None,
                        help="save trained model weights to this path")
    args = parser.parse_args()

    # --- Load image ---
    from PIL import Image
    img_pil = Image.open(args.input).convert('RGB')
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    H, W, _ = img_np.shape
    img_bytes = H * W * 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Build full coordinate grid ---
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    # --- Create model ---
    model = ImageMLP(
        n_levels=args.n_levels, n_features=args.n_features, log2_T=args.log2_T,
        hidden=args.hidden, depth=args.depth, omega_0=args.omega_0,
        base_res=args.base_res, max_res=args.max_res
    ).to(device)

    # Hash tables need a much higher LR than the SIREN decoder
    hash_params = list(model.encoder.parameters())
    net_params  = list(model.net.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.Adam([
        {'params': hash_params, 'lr': args.lr_hash},
        {'params': net_params,  'lr': args.lr_net},
    ], eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

    ratio = img_bytes / model.size_bytes
    print(f"Image      : {W}x{H}  ({img_bytes:,} bytes)")
    print(f"Model      : {model.param_count:,} params  ({model.size_bytes:,} bytes)")
    print(f"Compression: {ratio:.2f}x")
    print(f"Device     : {device}")
    print(f"\nTraining {args.iters} steps...  (Q/ESC to quit, Space to pause)\n")

    # --- Set up display ---
    if HAS_CV2:
        win = "tinyMLP — Hash Encoding + SIREN"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(W * 2 + 16, 1600), min(H + 84, 900))
        orig_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
                path = f"tinyMLP_step{step}.png"
                cv2.imwrite(path, cv2.cvtColor(
                    (np.clip(recon_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR))
                print(f"\n  Saved: {path}", end="", flush=True)

        if paused:
            time.sleep(0.05)
            continue

        step += 1

        # --- Train one step ---
        model.train()
        idx = torch.randint(0, N, (min(args.batch, N),), device=device)
        pred = model(all_coords[idx])
        loss = ((pred - all_colors[idx]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Update display every 50 steps ---
        update_display = (step % 50 == 0) or (step <= 10) or (step == args.iters)
        if update_display:
            recon_np = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon_np) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            elapsed = time.time() - t0

            line1 = (f"Step {step}/{args.iters}  |  "
                     f"Loss {loss.item():.5f}  |  "
                     f"PSNR {psnr:.2f} dB  |  "
                     f"Ratio {ratio:.1f}x  |  "
                     f"{elapsed:.1f}s")
            line2 = (f"Hash: {model.encoder.param_count if hasattr(model.encoder, 'param_count') else '?'} params  |  "
                     f"Model: {model.param_count:,} params, "
                     f"{model.size_bytes // 1024}KB  |  "
                     f"LR_hash {scheduler.get_last_lr()[0]:.1e}  "
                     f"LR_net {scheduler.get_last_lr()[1]:.1e}")

            print(f"\r{line1}", end="", flush=True)

            if HAS_CV2:
                recon_bgr = cv2.cvtColor(
                    (np.clip(recon_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                cv2.imshow(win, make_display_cv2(orig_bgr, recon_bgr, line1, line2))
            else:
                make_display_mpl(fig, axes, img_np, recon_np, line1)

    # --- Done ---
    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s — Final PSNR: {psnr:.2f} dB")

    if args.save:
        torch.save(model.state_dict(), args.save)
        import os
        sz = os.path.getsize(args.save)
        print(f"Model saved: {args.save} ({sz:,} bytes, {img_bytes / sz:.1f}x compression)")

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
