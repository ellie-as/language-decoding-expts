#!/usr/bin/env python3
"""Save an mp4 of cached per-TR flatmap images over time.

Reads a (T, H, W) image stack and a (H, W) pixel_mask written by
``train_flatmap_image_autoencoder.py`` and renders it as a video with a
matplotlib animation. Pixels outside the cortical sheet stay transparent
(NaN-masked).

Example:
    python 2D_representation_expt/make_flatmap_video.py \\
        --images 2D_representation_expt/results/flatmap_image_autoencoder/\\
flatmap_image_cache/S1__<xfm>__thick__h192__train_images.npy \\
        --pixel-mask 2D_representation_expt/results/flatmap_image_autoencoder/\\
flatmap_image_cache/S1__<xfm>__thick__h192__pixel_mask.npy \\
        --out s1_train_flatmap.mp4 \\
        --fps 5 --start-tr 0 --end-tr 600
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="Path to (T, H, W) .npy from the AE cache.")
    p.add_argument("--pixel-mask", required=True, help="Path to (H, W) bool .npy pixel mask.")
    p.add_argument("--out", required=True, help="Output .mp4 path.")
    p.add_argument("--start-tr", type=int, default=0)
    p.add_argument("--end-tr", type=int, default=None, help="Exclusive end TR (default: full length).")
    p.add_argument("--stride", type=int, default=1, help="Use every Nth TR.")
    p.add_argument("--fps", type=float, default=5.0, help="Output video frame rate. fMRI TR ~2s, so fps=0.5 is real time.")
    p.add_argument("--vmax-quantile", type=float, default=0.99, help="Symmetric vmin/vmax computed from this absolute-value quantile across selected frames.")
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--writer", default="ffmpeg", choices=["ffmpeg", "pillow"], help="pillow falls back to .gif if ffmpeg is unavailable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images = np.load(args.images, mmap_mode="r")
    pixel_mask = np.load(args.pixel_mask).astype(bool)
    assert images.ndim == 3, f"expected (T, H, W), got {images.shape}"
    assert images.shape[1:] == pixel_mask.shape, f"image shape {images.shape[1:]} vs mask shape {pixel_mask.shape}"

    end = int(args.end_tr) if args.end_tr is not None else int(images.shape[0])
    start = max(0, int(args.start_tr))
    stride = max(1, int(args.stride))
    frame_indices = np.arange(start, end, stride, dtype=int)
    print(f"Rendering {len(frame_indices)} frames from images of shape {images.shape}")

    sample = images[frame_indices[:: max(1, len(frame_indices) // 200)]]
    sample_masked = sample[:, pixel_mask]
    vmax = float(np.nanquantile(np.abs(sample_masked), float(args.vmax_quantile))) or 1.0
    print(f"Symmetric color limits: ±{vmax:.4f}")

    fig, ax = plt.subplots(figsize=(pixel_mask.shape[1] / 100.0, pixel_mask.shape[0] / 100.0))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)

    def to_display(idx: int) -> np.ndarray:
        frame = images[idx].astype(np.float32, copy=True)
        frame[~pixel_mask] = np.nan
        return frame

    im = ax.imshow(to_display(frame_indices[0]), cmap=args.cmap, vmin=-vmax, vmax=vmax, origin="lower", interpolation="nearest")
    title = ax.set_title(f"TR {frame_indices[0]}")

    def update(k: int):
        idx = int(frame_indices[k])
        im.set_data(to_display(idx))
        title.set_text(f"TR {idx}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=1000.0 / float(args.fps), blit=False)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.writer == "ffmpeg":
        writer = animation.FFMpegWriter(fps=float(args.fps), bitrate=2400)
    else:
        if out_path.suffix.lower() == ".mp4":
            out_path = out_path.with_suffix(".gif")
            print(f"Pillow writer cannot make mp4; saving to {out_path}")
        writer = animation.PillowWriter(fps=float(args.fps))
    print(f"Writing {out_path} ...")
    anim.save(str(out_path), writer=writer, dpi=int(args.dpi))
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
