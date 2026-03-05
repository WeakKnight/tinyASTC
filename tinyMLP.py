"""Neural image compression via coordinate-based MLP.

Trains a small MLP to memorize an image by mapping (x, y) -> (r, g, b).
The "compressed" representation is the network weights — typically 3-10x
smaller than the raw pixel data.

Usage:
    python tinyMLP.py                           # compress sample.jpg
    python tinyMLP.py -i photo.png              # custom input
    python tinyMLP.py --hidden 128 --freq 6     # smaller model, higher compression
    python tinyMLP.py --save model.pth          # save trained weights

Controls:
    ESC / Q  — quit
    S        — save current reconstruction as PNG
    Space    — pause / resume training
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
# Fourier positional encoding — lets the MLP learn high frequencies
# ---------------------------------------------------------------
class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_frequencies).float() * torch.pi
        self.register_buffer('freqs', freqs)
        self.out_dim = 2 + 2 * 2 * num_frequencies

    def forward(self, x):
        encoded = [x]
        for f in self.freqs:
            encoded.append(torch.sin(f * x))
            encoded.append(torch.cos(f * x))
        return torch.cat(encoded, dim=-1)


# ---------------------------------------------------------------
# The MLP: Fourier features -> hidden layers -> RGB
# ---------------------------------------------------------------
class ImageMLP(nn.Module):
    def __init__(self, num_freq=10, hidden=256, depth=4):
        super().__init__()
        self.encoder = FourierFeatures(num_freq)
        layers = []
        dim_in = self.encoder.out_dim
        for i in range(depth):
            layers += [nn.Linear(dim_in, hidden), nn.GELU()]
            dim_in = hidden
        layers += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(self.encoder(coords))

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_bytes(self):
        return self.param_count * 4


# ---------------------------------------------------------------
# Render the full image from the MLP (in chunks to avoid OOM)
# ---------------------------------------------------------------
@torch.no_grad()
def render_full(model, coords, H, W, chunk=65536):
    parts = [model(coords[i:i+chunk]) for i in range(0, coords.shape[0], chunk)]
    return torch.cat(parts).cpu().numpy().reshape(H, W, 3)


# ---------------------------------------------------------------
# Interactive display helpers
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
    parser.add_argument("-L", "--freq", type=int, default=10,
                        help="number of Fourier frequency bands (default: 10)")
    parser.add_argument("--hidden", type=int, default=128,
                        help="hidden layer width (default: 128)")
    parser.add_argument("--depth", type=int, default=3,
                        help="number of hidden layers (default: 3)")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=2**16,
                        help="pixels per training step")
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

    # --- Build coordinate grid [-1, 1] ---
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    # --- Create model ---
    model = ImageMLP(args.freq, args.hidden, args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

    ratio = img_bytes / model.size_bytes
    print(f"Image      : {W}x{H}  ({img_bytes:,} bytes)")
    print(f"Model      : {model.param_count:,} params  ({model.size_bytes:,} bytes)")
    print(f"Compression: {ratio:.2f}x")
    print(f"Device     : {device}")
    print(f"\nTraining {args.iters} steps...  (Q/ESC to quit, Space to pause)\n")

    # --- Set up display ---
    if HAS_CV2:
        win = "tinyMLP - Neural Image Compression"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(W * 2 + 16, 1600),
                         min(H + 84, 900))
        orig_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(top=0.88)

    # --- Training loop ---
    paused = False
    quit_flag = False
    t0 = time.time()
    psnr = 0.0
    recon_np = np.zeros_like(img_np)
    step = 0

    while step < args.iters:
        # Handle input
        if HAS_CV2:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                quit_flag = True
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

        # --- Update display ---
        update_display = (step % 50 == 0) or (step <= 10) or (step == args.iters)
        if update_display:
            model.eval()
            recon_np = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon_np) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            elapsed = time.time() - t0

            line1 = (f"Step {step}/{args.iters}  |  "
                     f"Loss {loss.item():.5f}  |  "
                     f"PSNR {psnr:.2f} dB  |  "
                     f"Ratio {ratio:.1f}x  |  "
                     f"{elapsed:.1f}s")
            line2 = (f"Model: {model.param_count:,} params, "
                     f"{model.size_bytes//1024}KB  |  "
                     f"Image: {W}x{H}, {img_bytes//1024}KB  |  "
                     f"LR {scheduler.get_last_lr()[0]:.1e}")

            print(f"\r{line1}", end="", flush=True)

            if HAS_CV2:
                recon_bgr = cv2.cvtColor(
                    (np.clip(recon_np, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                display = make_display_cv2(orig_bgr, recon_bgr, line1, line2)
                cv2.imshow(win, display)
            else:
                make_display_mpl(fig, axes, img_np, recon_np, line1)

    # --- Done ---
    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s — Final PSNR: {psnr:.2f} dB")

    if args.save:
        torch.save(model.state_dict(), args.save)
        import os
        sz = os.path.getsize(args.save)
        print(f"Model saved: {args.save} ({sz:,} bytes, {img_bytes/sz:.1f}x compression)")

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
