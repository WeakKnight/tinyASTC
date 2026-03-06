"""Generate MLP architecture diagram and training comparison figures.

Usage:
  python generate_mlp_figures.py              # everything
  python generate_mlp_figures.py --arch-only  # architecture diagram only
  python generate_mlp_figures.py --hash-only  # hash comparison figures only
  python generate_mlp_figures.py --latent-only # latent vs hash figures only

Requires: torch, matplotlib, Pillow, numpy
"""
import argparse
import sys
import pathlib
import time
import numpy as np

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from PIL import Image

local_dir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(local_dir))
from tinyMLP import HashEncoding, SirenLayer, ImageMLP, render_full
from tinyLatent import LatentTexture, LatentImageMLP

import os
os.makedirs('images', exist_ok=True)


# ====================================================================
# Figure 1: Architecture Diagram
# ====================================================================
def fig_mlp_architecture():
    fig, ax = plt.subplots(figsize=(18, 7), facecolor='#12121E')
    ax.set_xlim(0, 18)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    ax.set_facecolor('#12121E')

    BG   = '#12121E'
    C_IN   = '#4FC3F7'
    C_HASH = '#FF8A65'
    C_FEAT = '#FFD54F'
    C_SIREN= '#81C784'
    C_OUT  = '#CE93D8'
    C_ARR  = '#78909C'
    C_TXT  = '#ECEFF1'
    C_DIM  = '#90A4AE'

    def box(cx, cy, w, h, fc, ec, lw=1.8, alpha=0.18, radius=0.18):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle=f'round,pad={radius}',
            facecolor=fc, edgecolor=ec,
            linewidth=lw, alpha=alpha, zorder=4)
        ax.add_patch(rect)
        rect2 = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle=f'round,pad={radius}',
            facecolor='none', edgecolor=ec,
            linewidth=lw, zorder=5)
        ax.add_patch(rect2)

    def txt(x, y, s, color=C_TXT, fs=9, fw='normal', ha='center', va='center', **kw):
        ax.text(x, y, s, color=color, fontsize=fs, fontweight=fw,
                ha=ha, va=va, zorder=6, **kw)

    def arrow(x0, y0, x1, y1, color=C_ARR, lw=1.8):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', lw=lw,
                                    color=color, mutation_scale=14), zorder=7)

    # ── Input ──────────────────────────────────────────────────────────────
    box(1.1, 3.5, 1.6, 1.8, C_IN, C_IN)
    txt(1.1, 4.0, 'Input', color=C_IN, fs=11, fw='bold')
    txt(1.1, 3.5, '(x, y)', color=C_TXT, fs=10, fontfamily='monospace')
    txt(1.1, 3.0, 'coords ∈ [−1, 1]', color=C_DIM, fs=8)

    arrow(1.9, 3.5, 2.8, 3.5)

    # ── Multi-res grids ────────────────────────────────────────────────────
    box(4.3, 3.5, 2.6, 6.4, C_HASH, C_HASH, alpha=0.10)
    txt(4.3, 6.5, 'Multi-Resolution Hash Encoding', color=C_HASH, fs=10.5, fw='bold')
    txt(4.3, 6.1, '16 levels,  base=16px → max=512px', color=C_DIM, fs=8)

    grid_defs = [
        (3.0, 5.0, 4,  0.18, 'Coarse  (16×16)'),
        (3.0, 3.5, 8,  0.10, 'Medium  (64×64)'),
        (3.0, 2.0, 14, 0.07, 'Fine   (512×512)'),
    ]
    for gx0, gy0, n, cs, label in grid_defs:
        for i in range(n + 1):
            ax.plot([gx0, gx0 + n*cs], [gy0 + i*cs, gy0 + i*cs],
                    color=C_HASH, lw=0.4, alpha=0.5, zorder=3)
            ax.plot([gx0 + i*cs, gx0 + i*cs], [gy0, gy0 + n*cs],
                    color=C_HASH, lw=0.4, alpha=0.5, zorder=3)
        # Highlight one query cell
        qx, qy = n // 2, n // 2
        hilite = mpatches.Rectangle(
            (gx0 + qx*cs, gy0 + qy*cs), 2*cs, 2*cs,
            facecolor=C_HASH, alpha=0.4, zorder=4)
        ax.add_patch(hilite)
        # Dots at 4 corners
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            ax.plot(gx0 + (qx+dx)*cs, gy0 + (qy+dy)*cs, 'o',
                    color=C_FEAT, ms=4.5, zorder=6)
        txt(5.6, gy0 + n*cs/2, label, color=C_HASH, fs=7.5, ha='left')

    # bilinear label
    txt(4.3, 1.3, 'bilinear interp + hash lookup per level', color=C_DIM, fs=7.5, fontstyle='italic')

    arrow(5.6, 3.5, 6.5, 3.5)

    # ── Concatenated features ──────────────────────────────────────────────
    box(7.3, 3.5, 1.2, 4.8, C_FEAT, C_FEAT, alpha=0.12)
    txt(7.3, 6.1, 'Hash', color=C_FEAT, fs=10, fw='bold')
    txt(7.3, 5.75, 'Features', color=C_FEAT, fs=10, fw='bold')
    # Colored level stripes
    n_shown = 8
    stripe_h = 3.6 / n_shown
    for i in range(n_shown):
        alpha = 0.15 + 0.55 * i / n_shown
        stripe = mpatches.FancyBboxPatch(
            (6.75, 5.4 - (i+1) * stripe_h), 1.1, stripe_h * 0.9,
            boxstyle='round,pad=0.01',
            facecolor=C_FEAT, alpha=alpha, zorder=5)
        ax.add_patch(stripe)
        if i in (0, n_shown - 1):
            lbl = 'level 0\n(F=2)' if i == 0 else 'level 15\n(F=2)'
            txt(7.3, 5.4 - (i + 0.5) * stripe_h, lbl, color=BG, fs=6.5, fw='bold')
        elif i == n_shown // 2:
            txt(7.3, 5.4 - (i + 0.5) * stripe_h, '⋮', color=BG, fs=10, fw='bold')
    txt(7.3, 1.3, '32 dims\n(16 × 2)', color=C_FEAT, fs=8.5)

    arrow(7.9, 3.5, 8.8, 3.5)

    # ── SIREN layers ───────────────────────────────────────────────────────
    siren_layers = [
        (9.8,  'SIREN Layer 1', '32 → 64\nsin(ω₀ · Wx)',  True),
        (11.5, 'SIREN Layer 2', '64 → 64\nsin(ω₀ · Wx)', False),
    ]
    for lx, title, subtitle, is_first in siren_layers:
        box(lx, 3.5, 1.6, 3.0, C_SIREN, C_SIREN, alpha=0.12 if not is_first else 0.18)
        txt(lx, 4.4, title, color=C_SIREN, fs=9.5, fw='bold')
        txt(lx, 3.8, subtitle.split('\n')[0], color=C_TXT, fs=8.5, fontfamily='monospace')
        txt(lx, 3.2, subtitle.split('\n')[1], color=C_SIREN, fs=8, fontstyle='italic')
        if is_first:
            txt(lx, 2.7, 'ω₀ = 30', color=C_DIM, fs=7.5)
        arrow(lx + 0.8, 3.5, lx + 1.7, 3.5)

    # ── Output head ───────────────────────────────────────────────────────
    box(13.6, 3.5, 1.6, 2.4, C_OUT, C_OUT, alpha=0.14)
    txt(13.6, 4.3, 'Linear', color=C_OUT, fs=10, fw='bold')
    txt(13.6, 3.8, '+ Sigmoid', color=C_OUT, fs=10, fw='bold')
    txt(13.6, 3.2, '64 → 3', color=C_TXT, fs=8.5, fontfamily='monospace')
    txt(13.6, 2.7, 'output ∈ [0, 1]', color=C_DIM, fs=7.5)

    arrow(14.4, 3.5, 15.3, 3.5)

    # ── Output pixel ──────────────────────────────────────────────────────
    box(15.9, 3.5, 1.0, 2.2, C_OUT, C_OUT)
    txt(15.9, 3.9, 'RGB', color=C_OUT, fs=12, fw='bold')
    txt(15.9, 3.35, 'pixel\ncolor', color=C_TXT, fs=8.5)

    # ── MSE loss annotation ───────────────────────────────────────────────
    ax.annotate('MSE Loss vs\ntarget pixel',
                xy=(15.9, 2.4), xytext=(15.9, 1.0),
                ha='center', fontsize=8.5, color='#EF9A9A', fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.8, color='#EF9A9A'), zorder=8)

    # ── Depth brace ───────────────────────────────────────────────────────
    ax.annotate('', xy=(12.3, 0.5), xytext=(8.8, 0.5),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=C_SIREN, mutation_scale=12))
    txt(10.55, 0.15, 'depth SIREN layers  (default: 2,  hidden: 64)', color=C_SIREN, fs=8, fontstyle='italic')

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        'tinyMLP Architecture:  (x, y)  →  Hash Encoding  →  SIREN  →  (r, g, b)',
        fontsize=13, fontweight='bold', color=C_TXT, y=0.97)

    plt.savefig('images/fig_mlp_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='#12121E', edgecolor='none')
    plt.close()
    print("Generated: images/fig_mlp_architecture.png")


# ====================================================================
# Training helper — headless, returns snapshots + PSNR curve
# ====================================================================
def train_snapshots(img_np, model_kwargs,
                    total_steps=5000, snapshot_steps=(500, 2000, 5000),
                    lr_hash=3e-2, lr_net=1e-3, batch=2**16):
    H, W, _ = img_np.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    model = ImageMLP(**model_kwargs).to(device)

    hash_params = list(model.encoder.parameters())
    net_params  = list(model.net.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.Adam([
        {'params': hash_params, 'lr': lr_hash},
        {'params': net_params,  'lr': lr_net},
    ], eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    snapshots = {}
    psnr_curve = []
    snap_set = set(snapshot_steps)
    curve_interval = max(total_steps // 20, 50)

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

        if step in snap_set or step % curve_interval == 0:
            recon = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            psnr_curve.append((step, psnr))
            if step in snap_set:
                snapshots[step] = (np.clip(recon, 0, 1), psnr)
                print(f"  step {step:5d}/{total_steps}  PSNR {psnr:.2f} dB  ({time.time()-t0:.1f}s)")

    img_bytes = H * W * 3
    return snapshots, psnr_curve, model.param_count, img_bytes / model.size_bytes


# ====================================================================
# Figure 2: Quality grid (architectures × steps)
# ====================================================================
def fig_mlp_comparison(img_np):
    CONFIGS = [
        {
            'label': 'Tiny\n(log2_T=10, h=32, d=1)',
            'model_kwargs': dict(n_levels=8,  n_features=2, log2_T=10,
                                 hidden=32,  depth=1, base_res=8, max_res=256),
        },
        {
            'label': 'Default\n(log2_T=12, h=64, d=2)',
            'model_kwargs': dict(n_levels=16, n_features=2, log2_T=12,
                                 hidden=64,  depth=2),
        },
        {
            'label': 'Large\n(log2_T=14, h=128, d=3)',
            'model_kwargs': dict(n_levels=16, n_features=2, log2_T=14,
                                 hidden=128, depth=3),
        },
    ]
    STEPS = [500, 2000, 5000]

    all_snaps, all_curves, all_params, all_ratios = [], [], [], []
    for cfg in CONFIGS:
        print(f"\nTraining  {cfg['label'].replace(chr(10), ' ')} ...")
        snaps, curve, params, ratio = train_snapshots(
            img_np, cfg['model_kwargs'], total_steps=5000, snapshot_steps=STEPS)
        all_snaps.append(snaps)
        all_curves.append(curve)
        all_params.append(params)
        all_ratios.append(ratio)

    # ── Grid figure ──────────────────────────────────────────────────────
    n_rows, n_cols = len(CONFIGS), len(STEPS) + 1
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig,
                           wspace=0.04, hspace=0.38,
                           height_ratios=[0.06] + [1] * n_rows)

    # Column headers
    for j, t in enumerate(['Original'] + [f'{s} steps' for s in STEPS]):
        ax = fig.add_subplot(gs[0, j])
        ax.axis('off')
        ax.text(0.5, 0.5, t, ha='center', va='center', fontsize=12, fontweight='bold',
                color='#212121' if j == 0 else '#1565C0', transform=ax.transAxes)

    # Image grid
    for i, (cfg, snaps, params, ratio) in enumerate(
            zip(CONFIGS, all_snaps, all_params, all_ratios)):

        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(img_np)
        ax.axis('off')
        label = cfg['label'].replace('\n', '  ') + f'\n{params:,} params  ·  {ratio:.1f}x'
        ax.text(0.5, -0.04, label, ha='center', va='top', fontsize=8,
                color='#333333', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                          edgecolor='#BDBDBD', alpha=0.9))

        for j, step in enumerate(STEPS):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            recon, psnr = snaps[step]
            ax.imshow(recon)
            ax.axis('off')
            color = '#1B5E20' if psnr >= 33 else '#2E7D32' if psnr >= 28 else '#E65100'
            ax.set_title(f'PSNR {psnr:.1f} dB', fontsize=9.5, color=color, pad=3)

    fig.suptitle('tinyMLP (Hash Encoding + SIREN): Quality vs Steps vs Model Size',
                 fontsize=14, fontweight='bold', y=0.997)
    plt.savefig('images/fig_mlp_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("\nGenerated: images/fig_mlp_comparison.png")

    # ── PSNR curve figure ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5.5), facecolor='white')
    colors = ['#E53935', '#1E88E5', '#43A047']
    for cfg, curve, ratio, color in zip(CONFIGS, all_curves, all_ratios, colors):
        xs = [s for s, _ in curve]
        ys = [p for _, p in curve]
        label = cfg['label'].replace('\n', '  ') + f'  ({ratio:.1f}x)'
        ax2.plot(xs, ys, '-', color=color, label=label, linewidth=2.2)
        for step in STEPS:
            sx = [s for s, p in curve if s == step]
            sy = [p for s, p in curve if s == step]
            if sx:
                ax2.plot(sx, sy, 'o', color=color, markersize=7,
                         markeredgecolor='white', markeredgewidth=1, zorder=5)
                ax2.annotate(f'{sy[0]:.1f}', (sx[0], sy[0]),
                             textcoords='offset points', xytext=(4, 5),
                             fontsize=8, color=color, fontweight='bold')

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Hash Encoding + SIREN: PSNR vs Training Steps',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.92)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    fig2.tight_layout()
    fig2.savefig('images/fig_mlp_psnr_curve.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_mlp_psnr_curve.png")

    return all_configs_results(CONFIGS, all_snaps, all_params, all_ratios, STEPS)


def all_configs_results(configs, all_snaps, all_params, all_ratios, steps):
    rows = []
    for cfg, snaps, params, ratio in zip(configs, all_snaps, all_params, all_ratios):
        row = {'label': cfg['label'].replace('\n', '  '), 'params': params, 'ratio': ratio}
        for s in steps:
            row[f'psnr_{s}'] = snaps[s][1]
        rows.append(row)
    return rows


# ====================================================================
# Latent training helper — mirrors train_snapshots but uses LatentImageMLP
# ====================================================================
def train_latent_snapshots(img_np, H, W, scale, channels,
                           hidden=64, depth=2, omega_0=30.0,
                           total_steps=5000, snapshot_steps=(500, 2000, 5000),
                           lr=3e-3, batch=2**16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H),
                             torch.linspace(-1, 1, W), indexing='ij')
    all_coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2).to(device)
    all_colors  = torch.from_numpy(img_np).reshape(-1, 3).to(device)
    N = all_coords.shape[0]

    model = LatentImageMLP(H=H, W=W, scale=scale, channels=channels,
                           hidden=hidden, depth=depth, omega_0=omega_0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    snapshots = {}
    psnr_curve = []
    snap_set = set(snapshot_steps)
    curve_interval = max(total_steps // 20, 50)

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

        if step in snap_set or step % curve_interval == 0:
            recon = render_full(model, all_coords, H, W)
            mse = np.mean((img_np - recon) ** 2)
            psnr = 20 * np.log10(1.0 / max(np.sqrt(mse), 1e-10))
            psnr_curve.append((step, psnr))
            if step in snap_set:
                lat_preview = model.encoder.to_rgb_preview()
                snapshots[step] = (np.clip(recon, 0, 1), psnr, lat_preview)
                print(f"  step {step:5d}/{total_steps}  PSNR {psnr:.2f} dB  ({time.time()-t0:.1f}s)")

    img_bytes = H * W * 3
    lat_params = model.encoder.latent.numel()
    net_params  = model.param_count - lat_params
    latent_np   = model.encoder.latent.detach().cpu().numpy()
    return snapshots, psnr_curve, model.param_count, img_bytes / model.size_bytes, latent_np, lat_params, net_params


# ====================================================================
# Figure: Latent vs Hash orthogonal comparison
# ====================================================================
def fig_latent_vs_hash(img_np):
    """Train matched-budget Hash and Latent models; produce comparison grid + curves."""
    H, W, _ = img_np.shape
    img_bytes = H * W * 3
    STEPS = [500, 2000, 5000]

    # ── Default-budget Hash config ────────────────────────────────────────
    hash_cfg_default = dict(n_levels=16, n_features=2, log2_T=12,
                            hidden=64, depth=2)
    # ── Matched Latent config (scale=8, channels=32 → ~same KB as Hash Default) ──
    latent_scale_default, latent_ch_default = 8, 32

    print("\n[Hash Default]  Training ...")
    h_snaps, h_curve, h_params, h_ratio = train_snapshots(
        img_np, hash_cfg_default, total_steps=5000, snapshot_steps=STEPS)

    print("\n[Latent Default]  Training ...")
    l_snaps, l_curve, l_params, l_ratio, latent_np, lat_p, net_p = train_latent_snapshots(
        img_np, H, W,
        scale=latent_scale_default, channels=latent_ch_default,
        total_steps=5000, snapshot_steps=STEPS)

    # ── Grid figure (2 rows × 4 cols) ───────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), facecolor='white')
    plt.subplots_adjust(wspace=0.04, hspace=0.30)

    row_labels = [
        f'Hash Encoding\n{h_params:,} params  {h_ratio:.1f}x',
        f'Latent Texture\n{l_params:,} params  {l_ratio:.1f}x',
    ]
    col_labels = ['Original'] + [f'{s} steps' for s in STEPS]
    col_colors = ['#333333', '#1565C0', '#1565C0', '#1565C0']

    for j, (lbl, col) in enumerate(zip(col_labels, col_colors)):
        ax = axes[0, j]
        ax.set_title(lbl, fontsize=11.5, fontweight='bold', color=col, pad=6)

    for i, (snaps, row_lbl) in enumerate(
            [(h_snaps, row_labels[0]), (l_snaps, row_labels[1])]):
        # Original column
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        axes[i, 0].text(-0.04, 0.5, row_lbl,
                        transform=axes[i, 0].transAxes,
                        ha='right', va='center', fontsize=9, color='#333333',
                        rotation=0, wrap=True,
                        bbox=dict(boxstyle='round,pad=0.35', fc='#F5F5F5',
                                  ec='#BDBDBD', alpha=0.9))
        for j, step in enumerate(STEPS):
            ax = axes[i, j + 1]
            recon, psnr = snaps[step][0], snaps[step][1]
            ax.imshow(recon)
            ax.axis('off')
            color = '#1B5E20' if psnr >= 33 else '#2E7D32' if psnr >= 28 else '#E65100'
            ax.set_title(f'PSNR {psnr:.1f} dB', fontsize=9.5, color=color, pad=3)

    fig.suptitle('Hash Encoding vs Latent Texture — Matched Parameter Budget  (~same KB)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.savefig('images/fig_latent_vs_hash_grid.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nGenerated: images/fig_latent_vs_hash_grid.png")

    # ── Curve figure ─────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5.5), facecolor='white')
    curve_data = [
        (h_curve, '#1E88E5', f'Hash Default  ({h_params:,} params, {h_ratio:.1f}x)'),
        (l_curve, '#E53935', f'Latent Default ({l_params:,} params, {l_ratio:.1f}x)'),
    ]
    for curve, color, label in curve_data:
        xs = [s for s, _ in curve]
        ys = [p for _, p in curve]
        ax2.plot(xs, ys, '-', color=color, label=label, linewidth=2.2)
        for step in STEPS:
            sy = [p for s, p in curve if s == step]
            if sy:
                ax2.plot([step], sy, 'o', color=color, markersize=7,
                         markeredgecolor='white', markeredgewidth=1, zorder=5)
                ax2.annotate(f'{sy[0]:.1f}',  (step, sy[0]),
                             textcoords='offset points', xytext=(4, 5),
                             fontsize=8.5, color=color, fontweight='bold')

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Hash Encoding vs Latent Texture: PSNR vs Training Steps\n'
                  '(same SIREN decoder, matched parameter budget)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.92)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    fig2.tight_layout()
    fig2.savefig('images/fig_latent_vs_hash_curves.png', dpi=150,
                 bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: images/fig_latent_vs_hash_curves.png")

    # Collect PSNR summary dict for README
    results = {
        'hash':   {'params': h_params, 'ratio': h_ratio,
                   **{f'psnr_{s}': h_snaps[s][1] for s in STEPS}},
        'latent': {'params': l_params, 'ratio': l_ratio,
                   **{f'psnr_{s}': l_snaps[s][1] for s in STEPS}},
    }

    # Also generate latent visualization using the final-step latent preview
    fig_latent_visualization(img_np, latent_np,
                             l_snaps[STEPS[-1]][0], l_snaps[STEPS[-1]][2])
    return results


# ====================================================================
# Figure: Latent visualization (4-panel)
# ====================================================================
def fig_latent_visualization(img_np, latent_np, recon_np, latent_preview_np):
    """4-panel figure: original crop | latent RGB preview | reconstruction | error."""
    H, W, _ = img_np.shape

    # Central crop
    cy, cx = H // 2, W // 2
    h_crop = min(H, 256)
    w_crop = min(W, 256)
    orig_crop   = img_np[cy-h_crop//2:cy+h_crop//2, cx-w_crop//2:cx+w_crop//2]
    recon_crop  = recon_np[cy-h_crop//2:cy+h_crop//2, cx-w_crop//2:cx+w_crop//2]

    # Per-pixel error heatmap
    err = np.mean((orig_crop - np.clip(recon_crop, 0, 1)) ** 2, axis=-1)
    err_norm = err / (err.max() + 1e-8)

    # Latent grid RGB preview (resize to crop size)
    lat_img = Image.fromarray(latent_preview_np)
    lat_img = lat_img.resize((w_crop, h_crop), Image.NEAREST)
    lat_arr = np.array(lat_img)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='white')
    panels = [
        (orig_crop,              'Original Crop',        None,       'gray'),
        (lat_arr,                'Latent Grid  (ch 0–2)', None,      '#E65100'),
        (np.clip(recon_crop, 0, 1), 'Reconstruction',   None,       'teal'),
        (err_norm,               'Error Heatmap',        'inferno',  '#B71C1C'),
    ]
    for ax, (arr, title, cmap, tcolor) in zip(axes, panels):
        if cmap:
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=12, fontweight='bold', color=tcolor, pad=5)
        ax.axis('off')

    fig.suptitle('Latent Texture Internals — what the model "memorized"',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/fig_latent_visualization.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: images/fig_latent_visualization.png")


# ====================================================================
# Run all
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch-only',   action='store_true')
    parser.add_argument('--hash-only',   action='store_true')
    parser.add_argument('--latent-only', action='store_true')
    args = parser.parse_args()

    run_arch   = not (args.hash_only or args.latent_only)
    run_hash   = not (args.arch_only or args.latent_only)
    run_latent = not (args.arch_only or args.hash_only)

    # Load image — use lossless PNG for clean PSNR numbers
    img_path = local_dir / 'sample.png'
    if not img_path.exists():
        img_path = local_dir / 'sample.jpg'
    img_np = np.array(Image.open(img_path).convert('RGB'),
                      dtype=np.float32) / 255.0
    print(f"Image: {img_path.name}  {img_np.shape[1]}×{img_np.shape[0]}")

    if run_arch:
        print("\n=== Generating architecture diagram ===")
        fig_mlp_architecture()

    hash_results = None
    if run_hash:
        print("\n=== Training Hash models for comparison figures ===")
        hash_results = fig_mlp_comparison(img_np)
        print("\n\n=== Hash Summary ===")
        for r in hash_results:
            print(f"{r['label']:50s}  "
                  f"{r['params']:>8,} params  "
                  f"{r['ratio']:5.1f}x  "
                  + "  ".join(f"@{s}: {r[f'psnr_{s}']:.1f}dB" for s in (500, 2000, 5000)))

    if run_latent:
        print("\n=== Training Latent vs Hash comparison ===")
        latent_results = fig_latent_vs_hash(img_np)
        print("\n\n=== Latent vs Hash Summary ===")
        for method, r in latent_results.items():
            print(f"{method:10s}  "
                  f"{r['params']:>8,} params  "
                  f"{r['ratio']:5.1f}x  "
                  + "  ".join(f"@{s}: {r[f'psnr_{s}']:.1f}dB" for s in (500, 2000, 5000)))

    print("\nAll figures generated in images/")
