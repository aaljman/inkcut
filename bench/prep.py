"""Benchmark prep: AIM-500 subset composited onto a flat background.

Usage: python prep.py <aim_root> <bench_dir> <n> <seed> <r,g,b>
       (the README's numbers: n=100, seed=0, 255,255,255)
Picks n images (fixed seed), estimates each foreground with pymatting from the original photo
and its ground-truth alpha, composites onto the flat colour, and saves
bench_dir/src/<id>.png, bench_dir/gt/<id>_alpha.png, bench_dir/gt/<id>_fg.png.
"""
import glob
import os
import random
import sys

import numpy as np
from PIL import Image
from pymatting import estimate_foreground_ml

root, bdir, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
bgc = np.array([float(v) for v in sys.argv[5].split(",")]) / 255
masks = sorted(glob.glob(os.path.join(root, "mask", "*.png")))
ids = [os.path.splitext(os.path.basename(m))[0] for m in masks]
orig = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(root, "original", "*"))}
ids = [i for i in ids if i in orig]
random.Random(seed).shuffle(ids)
ids = sorted(ids[:n])
os.makedirs(os.path.join(bdir, "src"), exist_ok=True)
os.makedirs(os.path.join(bdir, "gt"), exist_ok=True)
for k, i in enumerate(ids):
    img = np.asarray(Image.open(orig[i]).convert("RGB")).astype(np.float64) / 255
    a = np.asarray(Image.open(os.path.join(root, "mask", i + ".png")).convert("L")).astype(np.float64) / 255
    if a.shape != img.shape[:2]:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]))) / 255
    fg = estimate_foreground_ml(img, a)
    comp = fg * a[..., None] + bgc * (1 - a[..., None])
    Image.fromarray((comp * 255).round().clip(0, 255).astype(np.uint8)).save(os.path.join(bdir, "src", i + ".png"))
    Image.fromarray((a * 255).round().astype(np.uint8)).save(os.path.join(bdir, "gt", i + "_alpha.png"))
    Image.fromarray((fg * 255).round().clip(0, 255).astype(np.uint8)).save(os.path.join(bdir, "gt", i + "_fg.png"))
    print(f"{k + 1}/{len(ids)} {i} {img.shape[1]}x{img.shape[0]}", flush=True)
