"""inkcut: background removal for drawings, scans and photos on flat backgrounds.

  inkcut INPUT [INPUT ...] [-o OUT_DIR] [--gpu] [--mask FILE] [--no-clean]
         [--plain] [--save-mask] [--preview] [--res 2048] [--model-dir DIR]
         [--set KEY=VALUE ...]

INPUT is an image file or a folder of images. Writes <name>_cut.png (RGBA) next to
each input, or into OUT_DIR.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

from . import cleanup

EXTS = (".png", ".webp", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def open_rgb(path):
    """RGB; an input that already has transparency is flattened onto white first."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return im.convert("RGB")


def checker(w, h, size=16):
    y, x = np.mgrid[0:h, 0:w]
    board = ((x // size + y // size) % 2).astype(np.float32)
    return np.dstack([150 + 60 * board] * 3)


def preview(src, plain, cut):
    """Side-by-side: original | model cut | inkcut, the cuts on dark grey above a checkerboard."""
    h, w = src.shape[:2]
    tiles = [src.astype(np.uint8)]
    for rgba in (plain, cut):
        a = rgba[..., 3:4].astype(np.float32) / 255
        dark = rgba[..., :3] * a + 40 * (1 - a)
        board = rgba[..., :3] * a + checker(w, h) * (1 - a)
        tiles.append(np.concatenate([dark[: h // 2], board[h // 2:]]).clip(0, 255).astype(np.uint8))
    gap = np.full((h, 8, 3), 255, np.uint8)
    return Image.fromarray(np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], axis=1))


def gather(inputs):
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths += [p for p in sorted(glob.glob(os.path.join(inp, "*"))) if p.lower().endswith(EXTS)]
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            sys.exit(f"inkcut: not a file or folder: {inp}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inkcut", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", help="output folder (default: next to each input)")
    ap.add_argument("--gpu", action="store_true", help="run the model on CUDA if available")
    ap.add_argument("--mask", help="use this alpha mask (grayscale image) instead of running the model; one input only")
    ap.add_argument("--no-clean", action="store_true", help="plain model cut, no cleanup")
    ap.add_argument("--plain", action="store_true", help="also save the plain model cut as <name>_plain.png")
    ap.add_argument("--save-mask", action="store_true", help="also save the model mask as <name>_mask.png")
    ap.add_argument("--preview", action="store_true", help="also save <name>_preview.png: original | model cut | inkcut")
    ap.add_argument("--res", type=int, default=2048, help="model input size (default 2048)")
    ap.add_argument("--model-dir", help="folder with birefnet.py, BiRefNet_config.py and the weights (default: download)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override a cleanup setting")
    args = ap.parse_args(argv)

    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in cleanup.P:
            sys.exit(f"inkcut: unknown setting {k}")
        cleanup.P[k] = float(v) if isinstance(cleanup.P[k], float) else int(float(v))

    paths = gather(args.inputs)
    if args.mask and len(paths) != 1:
        sys.exit("inkcut: --mask takes exactly one input image")

    model = device = None
    if not args.mask:
        import torch
        from .model import load_model
        device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 8)
        t0 = time.time()
        model = load_model(device, args.model_dir)
        print(f"BiRefNet-HR on {device}, loaded in {time.time() - t0:.0f}s, {len(paths)} image(s)")

    for path in paths:
        t1 = time.time()
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = args.out or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        img = open_rgb(path)
        src = np.asarray(img).astype(np.float32)
        if args.mask:
            a0 = np.asarray(Image.open(args.mask).convert("L").resize(img.size)).astype(np.float32) / 255
        else:
            from .model import predict
            a0 = predict(model, img, args.res, device)

        plain = np.dstack([src, a0 * 255]).round().astype(np.uint8)
        rgb, a, info = (None, a0, "cleanup off") if args.no_clean else cleanup.clean(src, a0)
        cut = plain if rgb is None else np.dstack([rgb, a * 255]).round().astype(np.uint8)

        out_path = os.path.join(out_dir, stem + "_cut.png")
        Image.fromarray(cut, "RGBA").save(out_path)
        if args.plain:
            Image.fromarray(plain, "RGBA").save(os.path.join(out_dir, stem + "_plain.png"))
        if args.save_mask:
            Image.fromarray((a0 * 255).round().astype(np.uint8)).save(os.path.join(out_dir, stem + "_mask.png"))
        if args.preview:
            preview(src, plain, cut).save(os.path.join(out_dir, stem + "_preview.png"))
        print(f"{out_path}  ({time.time() - t1:.0f}s)  {info}")


if __name__ == "__main__":
    main()
