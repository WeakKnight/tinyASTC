"""Generate MLP architecture diagram and training comparison figures.

Usage: python generate_mlp_figures.py
Requires: torch, matplotlib, Pillow, numpy
"""
import sys
import pathlib
import time
import numpy as np

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from PIL import Image

local_dir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(local_dir))
from tinyMLP import FourierFeatures, ImageMLP, render_full

import os
os.makedirs('images', exist_ok=True)


# ====================================================================
# Figure 1: MLP Architecture Diagram
# ====================================================================
def fig_mlp_architecture():
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='#1A1A2E')
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.3, 5.3)
    ax.axis('off')
    ax.set_facecolor('#1A1A2E')

    # ---- color palette ----
    C_INPUT   = '#4FC3F7'  # light blue
    C_FOURIER = '#FF8A65'  # orange
    C_HIDDEN  = '#81C784'  # green
    C_OUTPUT  = '#CE93D8'  # purple
    C_ARROW   = '#90A4AE'  # grey
    C_TEXT    = '#ECEFF1'

    def draw_node(ax, cx, cy, r, color, label, sublabel=None):
        circ = plt.Circle((cx, cy), r, color=color, zorder=5, alpha=0.92)
        ax.add_patch(circ)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', zorder=6)
        if sublabel:
            ax.text(cx, cy - r - 0.22, sublabel, ha='center', va='top',
                    fontsize=7, color=color, zorder=6, alpha=0.85)

    def draw_box(ax, x, y, w, h, color, title, lines, fontsize=8):
        rect = patches.FancyBboxPatch((x, y), w, h,
                                       boxstyle='round,pad=0.12',
                                       facecolor=color + '22',
                                       edgecolor=color, linewidth=2, zorder=4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top',
                fontsize=fontsize, fontweight='bold', color=color, zorder=5)
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h - 0.52 - i*0.3, line,
                    ha='center', va='top', fontsize=7,
                    color=C_TEXT, zorder=5, alpha=0.85)

    def arrow(ax, x0, x1, y=2.6, color=C_ARROW):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='->', lw=1.8,
                                    color=color, mutation_scale=14),
                    zorder=7)

    # ---- Input nodes (x, y) ----
    draw_box(ax, 0.0, 1.7, 1.2, 1.8, C_INPUT, 'Input',
             ['(x, y)', 'pixel coords', '2D → [−1, 1]'])
    arrow(ax, 1.2, 1.7)

    # ---- Fourier encoding ----
    draw_box(ax, 1.7, 0.3, 2.8, 4.4, C_FOURIER, 'Fourier Encoding',
             ['x, y',
              'sin(π·x), cos(π·x)',
              'sin(2π·x), cos(2π·x)',
              '  ⋮',
              'sin(2^9π·x), cos(2^9π·x)',
              '─────────────',
              '42 features'],
             fontsize=8.5)
    arrow(ax, 4.5, 5.0)

    # ---- Hidden layers ----
    layer_configs = [
        (5.0, '#81C784', 'Linear\n+ GELU', '42 → h'),
        (7.5, '#A5D6A7', 'Linear\n+ GELU', 'h → h'),
        (10.0, '#C8E6C9', 'Linear\n+ GELU', 'h → h'),
    ]
    for lx, lc, lt, ls in layer_configs:
        draw_box(ax, lx, 1.0, 2.0, 3.2, lc, lt, [ls, '', 'h = 64/128/256'])
        arrow(ax, lx + 2.0, lx + 2.5)

    # ---- Output ----
    draw_box(ax, 12.5, 1.4, 2.0, 2.4, C_OUTPUT, 'Linear\n+ Sigmoid',
             ['h → 3', 'R, G, B', '[0, 1]'])
    arrow(ax, 14.5, 15.0)

    # ---- Output pixel ----
    draw_box(ax, 15.0, 1.7, 1.0, 1.8, C_OUTPUT, 'RGB',
             ['pixel', 'color'])

    # ---- Loss annotation ----
    ax.annotate('MSE Loss\nvs target pixel',
                xy=(13.5, 1.4), xytext=(13.5, 0.1),
                ha='center', fontsize=8, color='#EF9A9A',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#EF9A9A'),
                zorder=8)

    # ---- Depth brace ----
    ax.annotate('', xy=(12.3, 0.7), xytext=(4.8, 0.7),
                arrowprops=dict(arrowstyle='<->', lw=1.5,
                                color='#81C784', mutation_scale=12))
    ax.text(8.55, 0.42, 'depth hidden layers  (default: 3)',
            ha='center', fontsize=8, color='#81C784', fontstyle='italic')

    # ---- Title ----
    fig.suptitle('tinyMLP Architecture:  (x, y)  →  Fourier Features  →  MLP  →  (r, g, b)',
                 fontsize=13, fontweight='bold', color=C_TEXT, y=0.98)

    plt.savefig('images/fig_mlp_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='#1A1A2E', edgecolor='none')
    plt.close()
    print("Generated: images/fig_mlp_architecture.png")


# ====================================================================
# Training helper — headless, returns snapshots
# ====================================================================
def train_snapshots(img_np, hidden, depth, num_freq=10,
                    total_steps=2000, snapshot_steps=(500, 1000, 2000),
                    lr=3e-3, batch=2**16):
    H, W, _ = img_np.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    model = ImageMLP(num_freq, hidden, depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    snapshots = {}
    psnr_curve = []
    snap_set = set(snapshot_steps)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        idx = torch.randint(0, N, (min(batch, N),), device=device)
        pred = model(all_coords[idx])
        loss = ((pred - all_colors[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step in snap_set:
            model.eval()
            recon = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            snapshots[step] = (np.clip(recon, 0, 1), psnr)
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{total_steps}  PSNR {psnr:.2f} dB  ({elapsed:.1f}s)")

        if step % 200 == 0:
            model.eval()
            recon = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon) ** 2)
            psnr_curve.append((step, 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))))

    img_bytes = H * W * 3
    ratio = img_bytes / model.size_bytes
    return snapshots, psnr_curve, model.param_count, ratio


# ====================================================================
# Figure 2: Training progression (steps × architectures comparison)
# ====================================================================
def fig_mlp_comparison(img_np):
    CONFIGS = [
        {'label': 'Tiny\nhidden=64, depth=2',  'hidden': 64,  'depth': 2},
        {'label': 'Default\nhidden=128, depth=3', 'hidden': 128, 'depth': 3},
        {'label': 'Large\nhidden=256, depth=4', 'hidden': 256, 'depth': 4},
    ]
    STEPS = [500, 1000, 2000]

    all_snaps = []
    all_psnr_curves = []
    all_params = []
    all_ratios = []

    for cfg in CONFIGS:
        print(f"\nTraining {cfg['label'].replace(chr(10), ' ')} ...")
        snaps, curve, params, ratio = train_snapshots(
            img_np, cfg['hidden'], cfg['depth'],
            total_steps=2000, snapshot_steps=STEPS
        )
        all_snaps.append(snaps)
        all_psnr_curves.append(curve)
        all_params.append(params)
        all_ratios.append(ratio)

    n_rows = len(CONFIGS)
    n_cols = len(STEPS) + 1  # +1 for original
    fig = plt.figure(figsize=(18, 11), facecolor='white')
    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig,
                           wspace=0.04, hspace=0.35,
                           height_ratios=[0.06] + [1] * n_rows)

    # ---- Column headers ----
    col_titles = ['Original'] + [f'{s} steps' for s in STEPS]
    for j, t in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.axis('off')
        ax.text(0.5, 0.5, t, ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='#212121' if j == 0 else '#1565C0',
                transform=ax.transAxes)

    # ---- Image grid ----
    for i, (cfg, snaps, params, ratio) in enumerate(
            zip(CONFIGS, all_snaps, all_params, all_ratios)):

        # Original column
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(img_np)
        ax.axis('off')
        row_label = cfg['label'].replace('\n', '  ') + f'\n{params:,} params  ·  {ratio:.1f}x'
        ax.text(0.5, -0.04, row_label, ha='center', va='top',
                fontsize=8, color='#333333', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                          edgecolor='#BDBDBD', alpha=0.9))

        for j, step in enumerate(STEPS):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            recon, psnr = snaps[step]
            ax.imshow(recon)
            ax.axis('off')
            ax.set_title(f'PSNR {psnr:.1f} dB', fontsize=9,
                         color='#2E7D32' if psnr > 28 else '#E65100', pad=3)

    fig.suptitle('tinyMLP: Quality vs Steps vs Model Size', fontsize=15,
                 fontweight='bold', y=0.995)
    plt.savefig('images/fig_mlp_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("\nGenerated: images/fig_mlp_comparison.png")

    # ---- PSNR curve figure ----
    fig2, ax2 = plt.subplots(figsize=(9, 5), facecolor='white')
    colors = ['#E53935', '#1E88E5', '#43A047']
    for cfg, curve, ratio, color in zip(CONFIGS, all_psnr_curves, all_ratios, colors):
        steps_x = [s for s, _ in curve]
        psnrs_y = [p for _, p in curve]
        label = cfg['label'].replace('\n', '  ') + f'  ({ratio:.1f}x)'
        ax2.plot(steps_x, psnrs_y, '-o', color=color, label=label,
                 linewidth=2, markersize=4)

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('PSNR vs Training Steps (512×512 image)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    fig2.savefig('images/fig_mlp_psnr_curve.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_mlp_psnr_curve.png")


# ====================================================================
# Run all
# ====================================================================
if __name__ == '__main__':
    print("=== Generating MLP architecture diagram ===")
    fig_mlp_architecture()

    print("\n=== Training models for comparison figures ===")
    img_np = np.array(Image.open(local_dir / 'sample.jpg').convert('RGB'),
                      dtype=np.float32) / 255.0
    fig_mlp_comparison(img_np)

    print("\nAll MLP figures generated in images/")
