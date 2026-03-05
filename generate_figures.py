"""Generate educational figures for the tinyBC README.

Usage: python generate_figures.py
Requires: matplotlib, Pillow, numpy
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os

os.makedirs('images', exist_ok=True)

original = np.array(Image.open('sample.jpg').convert('RGB')).astype(np.float64) / 255.0
simplex = np.array(Image.open('simplex.png').convert('RGB')).astype(np.float64) / 255.0
nosimplex = np.array(Image.open('nosimplex.png').convert('RGB')).astype(np.float64) / 255.0


def pca_endpoints(block_pixels):
    """Compute PCA-based endpoint colors for a block of pixels."""
    mean = block_pixels.mean(axis=0)
    centered = block_pixels - mean
    cov = centered.T @ centered
    _, vecs = np.linalg.eigh(cov)
    direction = vecs[:, -1]
    projs = centered @ direction
    ep0 = np.clip(mean + direction * projs.min(), 0, 1)
    ep1 = np.clip(mean + direction * projs.max(), 0, 1)
    weights = (projs - projs.min()) / max(projs.max() - projs.min(), 1e-10)
    return ep0, ep1, weights


# ============================================================
# Figure 1: Block Compression - The Core Idea
# ============================================================
def fig_block_concept():
    fig = plt.figure(figsize=(16, 5.5), facecolor='white')
    gs = gridspec.GridSpec(2, 7, figure=fig, wspace=0.35, hspace=0.5,
                           width_ratios=[2.8, 0.25, 1.1, 0.25, 1.1, 1.1, 1.4])

    crop_y, crop_x, crop_size = 180, 130, 80
    crop = original[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    bx, by = 28, 32

    # --- Panel 0: Image with 4x4 grid ---
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(crop)
    for i in range(0, crop_size + 1, 4):
        ax.axhline(y=i - 0.5, color='white', linewidth=0.3, alpha=0.4)
        ax.axvline(x=i - 0.5, color='white', linewidth=0.3, alpha=0.4)
    rect = patches.Rectangle((bx - 0.5, by - 0.5), 4, 4,
                               linewidth=3, edgecolor='#FFD700', facecolor='none')
    ax.add_patch(rect)
    ax.set_title('Image divided into\n4x4 pixel blocks', fontsize=11, fontweight='bold')
    ax.axis('off')

    # --- Arrow ---
    ax_a1 = fig.add_subplot(gs[:, 1])
    ax_a1.axis('off')
    ax_a1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#757575'),
                   xycoords='axes fraction')

    # --- Panel 1: Zoomed 4x4 block ---
    ax = fig.add_subplot(gs[:, 2])
    block = crop[by:by+4, bx:bx+4]
    ax.imshow(block, interpolation='nearest')
    for i in range(5):
        ax.axhline(y=i - 0.5, color='white', linewidth=1.5)
        ax.axvline(x=i - 0.5, color='white', linewidth=1.5)
    ax.set_title('One 4x4 block\n(16 pixels)', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Arrow ---
    ax_a2 = fig.add_subplot(gs[:, 3])
    ax_a2.axis('off')
    ax_a2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#757575'),
                   xycoords='axes fraction')
    ax_a2.text(0.5, 0.35, 'encode', fontsize=7, ha='center', va='center',
               rotation=0, color='#757575', fontstyle='italic',
               transform=ax_a2.transAxes)

    # --- Panel 2: Endpoints ---
    block_flat = block.reshape(-1, 3)
    ep0, ep1, weights_flat = pca_endpoints(block_flat)
    weights_grid = weights_flat.reshape(4, 4)

    ax_ep = fig.add_subplot(gs[0, 4])
    gradient = np.zeros((1, 32, 3))
    for t_idx in range(32):
        t_val = t_idx / 31.0
        gradient[0, t_idx] = (1 - t_val) * ep0 + t_val * ep1
    ax_ep.imshow(gradient, interpolation='nearest', aspect='auto')
    ax_ep.plot(0, 0, 's', markersize=12, color=ep0, markeredgecolor='white', markeredgewidth=2)
    ax_ep.plot(31, 0, 's', markersize=12, color=ep1, markeredgecolor='white', markeredgewidth=2)
    ax_ep.set_title('2 endpoint colors', fontsize=10, fontweight='bold')
    ax_ep.text(0, 0, ' A', fontsize=8, fontweight='bold', color='white', va='center')
    ax_ep.text(29, 0, 'B ', fontsize=8, fontweight='bold', color='white', va='center', ha='right')
    ax_ep.set_xticks([])
    ax_ep.set_yticks([])

    # --- Panel 3: Weight grid ---
    ax_w = fig.add_subplot(gs[1, 4])
    ax_w.imshow(weights_grid, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
    for i in range(4):
        for j in range(4):
            val = weights_grid[i, j]
            ax_w.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=7, color='white' if 0.3 < val < 0.7 else 'black',
                      fontweight='bold')
    for i in range(5):
        ax_w.axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        ax_w.axvline(x=i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    ax_w.set_title('16 weights', fontsize=10, fontweight='bold')
    ax_w.set_xticks([])
    ax_w.set_yticks([])

    # --- Panel 4: Reconstructed block ---
    ax_r = fig.add_subplot(gs[0, 5])
    reconstructed = np.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            w = weights_grid[i, j]
            reconstructed[i, j] = (1 - w) * ep0 + w * ep1
    ax_r.imshow(np.clip(reconstructed, 0, 1), interpolation='nearest')
    for i in range(5):
        ax_r.axhline(y=i - 0.5, color='white', linewidth=1.5)
        ax_r.axvline(x=i - 0.5, color='white', linewidth=1.5)
    ax_r.set_title('Decoded block', fontsize=10, fontweight='bold')
    ax_r.set_xticks([])
    ax_r.set_yticks([])

    # --- Formula ---
    ax_f = fig.add_subplot(gs[1, 5])
    ax_f.axis('off')
    ax_f.text(0.5, 0.7, 'pixel = lerp(A, B, w)', fontsize=10,
              ha='center', va='center', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7', edgecolor='#FBC02D', lw=1.5))
    ax_f.text(0.5, 0.2, '128 bits / block\n= 4:1 compression', fontsize=9,
              ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1976D2', lw=1.5))

    # --- Sidebar: bit budget ---
    ax_info = fig.add_subplot(gs[:, 6])
    ax_info.axis('off')
    info_text = (
        "Bit budget per block\n"
        "================\n"
        "Endpoint A    ~28 bit\n"
        "Endpoint B    ~28 bit\n"
        "16 weights    ~64 bit\n"
        "Mode + misc    ~8 bit\n"
        "----------------\n"
        "Total         128 bit\n"
        "\n"
        "Uncompressed:\n"
        "  16 x 32 = 512 bit\n"
        "\n"
        "Savings: 75%"
    )
    ax_info.text(0.08, 0.5, info_text, fontsize=9, fontfamily='monospace',
                 va='center', ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA', edgecolor='#BDBDBD'))

    fig.suptitle('Block Compression at a Glance', fontsize=15, fontweight='bold', y=1.02)
    plt.savefig('images/fig_block_concept.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_block_concept.png")


# ============================================================
# Figure 2: Color Line in RGB Space
# ============================================================
def fig_color_line():
    fig = plt.figure(figsize=(8, 6.5), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    by, bx = 200, 150
    block = original[by:by+4, bx:bx+4]
    pixels = block.reshape(-1, 3)

    ep0, ep1, weights_flat = pca_endpoints(pixels)

    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2],
               c=pixels, s=100, edgecolors='black', linewidth=0.8, zorder=5,
               depthshade=False)

    t = np.linspace(-0.1, 1.1, 60)
    line = ep0[np.newaxis, :] + t[:, np.newaxis] * (ep1 - ep0)[np.newaxis, :]
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r-', linewidth=2.5,
            label='Color line', zorder=3)

    ax.scatter(*ep0, c=[ep0], s=200, marker='D', edgecolors='black',
               linewidth=1.5, zorder=10, label='Endpoint A')
    ax.scatter(*ep1, c=[ep1], s=200, marker='D', edgecolors='black',
               linewidth=1.5, zorder=10, label='Endpoint B')

    ep_diff = ep1 - ep0
    ep_len_sq = np.dot(ep_diff, ep_diff)
    for p in pixels:
        proj_t = np.clip(np.dot(p - ep0, ep_diff) / max(ep_len_sq, 1e-10), 0, 1)
        proj_point = ep0 + proj_t * ep_diff
        ax.plot([p[0], proj_point[0]], [p[1], proj_point[1]], [p[2], proj_point[2]],
                'k--', alpha=0.35, linewidth=0.8)

    ax.set_xlabel('Red', fontsize=11, labelpad=8)
    ax.set_ylabel('Green', fontsize=11, labelpad=8)
    ax.set_zlabel('Blue', fontsize=11, labelpad=8)
    ax.set_title('Block compression = fitting a line\nthrough pixel colors in RGB space',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=25, azim=135)

    plt.savefig('images/fig_color_line.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_color_line.png")


# ============================================================
# Figure 3: Algorithm Pipeline
# ============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 4), facecolor='white')
    ax.set_xlim(-0.3, 15.3)
    ax.set_ylim(-0.8, 4.2)
    ax.axis('off')

    box_defs = [
        (0.2,  0.6, 2.3, 2.2, 'Input\nTexture',
         '#E3F2FD', '#1565C0'),
        (3.3,  0.6, 2.3, 2.2, 'Split into\n4x4 Blocks',
         '#E8F5E9', '#2E7D32'),
        (6.4,  0.6, 2.5, 2.2, 'PCA\nInitial Guess',
         '#FFF3E0', '#E65100'),
        (9.7,  0.6, 2.8, 2.2, 'Nelder-Mead\nSimplex Opt.',
         '#FCE4EC', '#C62828'),
        (13.0, 0.6, 2.0, 2.2, 'Decoded\nOutput',
         '#E8EAF6', '#283593'),
    ]

    for x, y, w, h, text, fc, ec in box_defs:
        rect = patches.FancyBboxPatch((x, y), w, h,
                                       boxstyle='round,pad=0.18',
                                       facecolor=fc, edgecolor=ec, linewidth=2.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=ec)

    arrow_pairs = [(2.5, 3.3), (5.6, 6.4), (8.9, 9.7), (12.5, 13.0)]
    for x0, x1 in arrow_pairs:
        ax.annotate('', xy=(x1, 1.7), xytext=(x0, 1.7),
                    arrowprops=dict(arrowstyle='->', lw=2.2, color='#616161'))

    ax.annotate('loss > threshold?', xy=(9.7, 2.5), xytext=(8.2, 3.7),
                fontsize=9, ha='center', fontstyle='italic', color='#C62828',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#C62828'))

    ax.annotate('good enough: skip!', xy=(13.0, 1.1), xytext=(8.5, -0.4),
                fontsize=9, ha='center', fontstyle='italic', color='#2E7D32',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2E7D32',
                                connectionstyle='arc3,rad=-0.15'))

    fig.suptitle('tinyBC Encoding Pipeline', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig('images/fig_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_pipeline.png")


# ============================================================
# Figure 4: Quality Comparison
# ============================================================
def fig_comparison():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.08, hspace=0.25,
                           height_ratios=[1, 1])

    # Crop a detail-rich region (headlights / grille area)
    cy, cx, cs = 230, 120, 140
    orig_crop = original[cy:cy+cs, cx:cx+cs]
    nosimp_crop = nosimplex[cy:cy+cs, cx:cx+cs]
    simp_crop = simplex[cy:cy+cs, cx:cx+cs]

    mse_nosimp = np.mean((original - nosimplex) ** 2)
    mse_simp = np.mean((original - simplex) ** 2)
    psnr_nosimp = 20 * np.log10(1.0 / np.sqrt(mse_nosimp)) if mse_nosimp > 0 else 99
    psnr_simp = 20 * np.log10(1.0 / np.sqrt(mse_simp)) if mse_simp > 0 else 99

    titles_row1 = [
        'Original',
        f'PCA Only  (PSNR {psnr_nosimp:.2f} dB)',
        f'PCA + Nelder-Mead  (PSNR {psnr_simp:.2f} dB)',
    ]
    crops_row1 = [orig_crop, nosimp_crop, simp_crop]
    for i, (c, t) in enumerate(zip(crops_row1, titles_row1)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(c)
        ax.set_title(t, fontsize=11, fontweight='bold')
        ax.axis('off')

    # Row 2: amplified error heatmaps
    error_nosimp = np.sqrt(np.mean((orig_crop - nosimp_crop) ** 2, axis=2))
    error_simp = np.sqrt(np.mean((orig_crop - simp_crop) ** 2, axis=2))
    vmax = max(error_nosimp.max(), error_simp.max(), 0.01)

    ax0 = fig.add_subplot(gs[1, 0])
    ax0.axis('off')
    ax0.text(0.5, 0.55, 'Error Maps', fontsize=16, ha='center', va='center',
             transform=ax0.transAxes, fontweight='bold', color='#424242')
    ax0.text(0.5, 0.35, '(brighter = more error)', fontsize=10,
             ha='center', va='center', transform=ax0.transAxes,
             fontstyle='italic', color='#757575')

    ax1 = fig.add_subplot(gs[1, 1])
    im1 = ax1.imshow(error_nosimp, cmap='inferno', vmin=0, vmax=vmax)
    ax1.set_title(f'PCA Only Error (max {error_nosimp.max():.4f})', fontsize=10)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 2])
    im2 = ax2.imshow(error_simp, cmap='inferno', vmin=0, vmax=vmax)
    ax2.set_title(f'PCA + NM Error (max {error_simp.max():.4f})', fontsize=10)
    ax2.axis('off')

    fig.suptitle('Compression Quality: PCA vs PCA + Nelder-Mead Simplex',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('images/fig_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_comparison.png")


# ============================================================
# Figure 5: Nelder-Mead Simplex Intuition (2D analogy)
# ============================================================
def fig_simplex_intuition():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 5 * (y - x ** 2) ** 2

    xg = np.linspace(-1.5, 2, 200)
    yg = np.linspace(-0.5, 3, 200)
    X, Y = np.meshgrid(xg, yg)
    Z = rosenbrock(X, Y)

    triangles_sequence = [
        np.array([[-1.0, 2.5], [1.5, 2.8], [0.0, 0.5]]),
        np.array([[0.0, 0.5], [1.5, 2.8], [0.8, 1.5]]),
        np.array([[0.8, 1.5], [1.5, 2.8], [1.2, 1.0]]),
        np.array([[0.8, 1.5], [1.2, 1.0], [1.05, 0.95]]),
    ]
    step_labels = ['Initial simplex', 'Reflection', 'Expansion', 'Contraction']
    colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5']

    for idx, (ax, tri, label, c) in enumerate(zip(axes, triangles_sequence, step_labels, colors)):
        ax.contourf(X, Y, np.log1p(Z), levels=30, cmap='YlOrBr_r', alpha=0.6)
        ax.contour(X, Y, np.log1p(Z), levels=15, colors='#8D6E63', linewidths=0.4, alpha=0.5)

        triangle = plt.Polygon(tri, fill=True, facecolor=c, alpha=0.25,
                                edgecolor=c, linewidth=2.5)
        ax.add_patch(triangle)
        ax.plot(*tri.T, 'o', color=c, markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax.plot(tri[-1, 0], tri[-1, 1], 'o', color=c, markersize=8,
                markeredgecolor='black', markeredgewidth=1)

        ax.plot(1, 1, '*', color='gold', markersize=15, markeredgecolor='black',
                markeredgewidth=1, zorder=10)
        ax.set_title(f'Step {idx+1}: {label}', fontsize=10, fontweight='bold')
        ax.set_xlim(-1.5, 2)
        ax.set_ylim(-0.5, 3)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Nelder-Mead Simplex: Walking a triangle downhill to find the optimum',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.savefig('images/fig_simplex_intuition.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: images/fig_simplex_intuition.png")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    fig_block_concept()
    fig_color_line()
    fig_pipeline()
    fig_comparison()
    fig_simplex_intuition()
    print("\nAll figures generated in images/")
