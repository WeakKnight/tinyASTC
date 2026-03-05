# tinyBC

**A minimal GPU texture block compressor in ~450 lines of code.**

tinyBC implements BC7 Mode 6 texture compression entirely on the GPU using a [Slang](https://shader-slang.com/) compute shader + a Python driver via [SlangPy](https://shader-slang.com/slang-python/). It compresses a texture in real-time, achieving ~40 dB PSNR, and fits in two small files you can read in one sitting.

<p align="center">
  <img src="images/fig_comparison.png" alt="Compression quality comparison" width="100%"/>
</p>

---

## Table of Contents

- [Quick Start](#quick-start)
- [What is Block Compression?](#what-is-block-compression)
  - [Why Block Compression Exists](#why-block-compression-exists)
  - [The Core Idea](#the-core-idea-two-colors-and-a-recipe)
  - [Geometric Insight](#geometric-insight-a-line-in-color-space)
- [How tinyBC Works](#how-tinybc-works)
  - [Step 1: PCA Initial Guess](#step-1-pca-initial-guess)
  - [Step 2: Nelder-Mead Refinement](#step-2-nelder-mead-refinement)
  - [The Pipeline](#the-full-pipeline)
- [Results](#results)
- [Code Walkthrough](#code-walkthrough)
- [License](#license)

---

## Quick Start

**Prerequisites:** Python 3.10+, a GPU with Vulkan support, and [SlangPy](https://shader-slang.com/slang-python/) installed.

```bash
pip install slangpy sgl numpy
```

**Run the compressor:**

```bash
python tinybc.py                          # compress sample.jpg, print PSNR
python tinybc.py -i photo.png -o out.png  # custom input, save decoded output
python tinybc.py -b                       # benchmark mode (1000 iterations)
```

That's it. Two files, zero build system.

---

## What is Block Compression?

### Why Block Compression Exists

When a GPU renders a 3D scene, it reads texture data _millions_ of times per frame. Unlike JPEG or PNG, the GPU can't afford to decompress an entire image first — it needs **random access** to any pixel at any time.

This rules out most image compression formats. JPEG uses variable-length coding that requires sequential decoding. PNG uses an LZ-based stream. Neither lets you jump to pixel (423, 871) without decoding everything before it.

**Block compression** solves this by dividing the image into small, independent tiles — typically **4×4 pixels** — each compressed to a **fixed-size** bit string (128 bits for BC7). The GPU can decode any block in O(1) without touching any other block.

| Property | JPEG/PNG | Block Compression |
|---|---|---|
| Random access | No | **Yes** |
| Decode unit | Entire image | Single 4×4 block |
| GPU-friendly | No (CPU decode) | **Yes** (hardware decoder) |
| Compression ratio | Very high | Moderate (~4:1) |
| Use case | Storage, web | **Real-time rendering** |

### The Core Idea: Two Colors and a Recipe

Every 4×4 block is encoded as:

1. **Two endpoint colors** (Color A and Color B)
2. **16 weights** — one per pixel, each saying "how much of A vs B"

To decode a pixel, just interpolate: `pixel = lerp(A, B, weight)`.

<p align="center">
  <img src="images/fig_block_concept.png" alt="Block compression concept" width="100%"/>
</p>

The total cost: ~28 bits for each endpoint + ~64 bits for 16 weights + a few mode bits = **128 bits per block**, or **0.5 bytes per pixel**. Uncompressed RGBA would cost 16 × 32 = 512 bits — a 4× saving.

> **Key insight:** We're betting that within any tiny 4×4 region of an image, all the colors can be _reasonably approximated_ as a blend of just two colors. For natural images, this bet pays off surprisingly well.

### Geometric Insight: A Line in Color Space

Here's another way to think about it. Each pixel is a point in RGB color space (a 3D cube). Block compression finds the **best-fit line** through these 16 points, then projects each pixel onto that line.

<p align="center">
  <img src="images/fig_color_line.png" alt="Color line in RGB space" width="55%"/>
</p>

The two endpoints define where the line starts and ends. The per-pixel weight records where each pixel falls along this line. All the compression error comes from the **perpendicular distance** between each pixel and the line — the off-axis detail that gets lost.

---

## How tinyBC Works

### Step 1: PCA Initial Guess

Finding the "best" two endpoints is an optimization problem. A brute-force search over all possible RGBA endpoint pairs would be astronomically expensive (each endpoint lives in a continuous 4D space).

tinyBC starts with a fast, classic trick: **Principal Component Analysis (PCA)**.

1. Compute the **mean color** of the 16 pixels.
2. Find the **dominant direction** of color variation (the axis of greatest spread).
3. Project all pixels onto this axis.
4. The two extremes become the initial endpoint colors.

This is cheap — just a few dot products — and gives a surprisingly good initial answer. For many blocks, it's already good enough (loss below a threshold), and we skip straight to output.

### Step 2: Nelder-Mead Refinement

For harder blocks (strong color variation, edges, mixed content), the PCA solution can be noticeably off. tinyBC then applies **Nelder-Mead simplex optimization** to refine the endpoints.

<p align="center">
  <img src="images/fig_simplex_intuition.png" alt="Nelder-Mead simplex intuition" width="100%"/>
</p>

Nelder-Mead is a derivative-free optimizer. It works by maintaining a **simplex** (a geometric shape with N+1 vertices in N dimensions). Since our search space is 8-dimensional (4 components × 2 endpoints), the simplex has 9 vertices. At each iteration, it:

| Operation | What happens |
|---|---|
| **Reflection** | Mirror the worst vertex through the centroid of the rest — try the "opposite direction" |
| **Expansion** | If reflection found a great point, push even further in that direction |
| **Contraction** | If reflection didn't help, pull the worst vertex closer to the centroid |
| **Shrink** | If nothing works, shrink the entire simplex toward the best vertex |

After up to 64 iterations, the simplex converges to a local (often global) minimum. The best vertex gives our refined endpoints.

> **Why Nelder-Mead?** It's derivative-free (our loss landscape is non-smooth due to weight quantization), simple to implement in a shader, and converges quickly in low dimensions. Perfect for a GPU compute kernel where each thread independently optimizes one 4×4 block.

### The Full Pipeline

<p align="center">
  <img src="images/fig_pipeline.png" alt="Encoding pipeline" width="100%"/>
</p>

1. **Load** the input texture onto the GPU.
2. **Dispatch** one compute thread per 4×4 block.
3. Each thread runs **PCA** to get initial endpoints.
4. If loss > threshold (0.004), run **Nelder-Mead** (64 iterations max).
5. Compute final per-pixel weights and write the decoded block to the output texture.
6. **Compute PSNR** on the CPU by comparing input vs decoded.

---

## Results

| Metric | PCA Only | PCA + Nelder-Mead |
|---|---|---|
| **PSNR** | 38.46 dB | **40.05 dB** |
| Max per-pixel error | 0.1897 | **0.1601** |

The Nelder-Mead refinement adds ~1.6 dB PSNR — a meaningful improvement, especially on blocks with complex color distributions (edges, specular highlights, mixed materials).

<p align="center">
  <img src="images/fig_comparison.png" alt="Quality comparison with error maps" width="100%"/>
</p>

The error maps (bottom row) use the `inferno` colormap — brighter means more error. Notice how the Nelder-Mead version has fewer bright spots, especially around the grille bars and headlight edges where color variation is highest.

---

## Code Walkthrough

The entire project is two files:

### `tinybc.slang` — The GPU Compressor (~379 lines)

| Section | Lines | What it does |
|---|---|---|
| `compute_unorm_end_point_and_unorm_weight` | 53–149 | PCA-based initial endpoint + weight estimation |
| `compute_weights` | 152–172 | Given endpoints, project all 16 pixels onto the endpoint line to get weights |
| `compute_loss` | 175–188 | MSE between original pixels and their endpoint-interpolated reconstructions |
| `sort_simplex` | 191–206 | Bubble sort the 9 simplex vertices by loss (ascending) |
| `compute_centroid` | 209–217 | Mean of the 8 best vertices (excluding the worst) |
| `encoder` | 219–378 | Main entry point: load block → PCA → (optional) Nelder-Mead → write output |

### `tinybc.py` — The Python Driver (~72 lines)

Loads the input texture via `sgl.TextureLoader`, creates an output texture, dispatches the `encoder` kernel over all 4×4 tiles, and computes PSNR.

```
tinyASTC/
├── tinybc.slang       # GPU compute shader (the compressor)
├── tinybc.py          # Python driver script
├── sample.jpg         # Test input image
├── generate_figures.py # Generate README figures
├── images/            # Generated figures
└── LICENSE            # MIT
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
