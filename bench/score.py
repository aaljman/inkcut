"""Benchmark metrics against AIM-500 ground truth.

Usage: python score.py <bench_dir> <results_dir> [out.csv]
Per method (file suffix in results_dir): alpha SAD (/1000), MSE (x1e3), Grad (/1000, Gaussian
derivative sigma 1.4, as in the Composition-1k protocol), and re-composite error: the cut-out and
the ground truth (fg, alpha) each composited over dark grey (40,40,40); mean absolute error in
8-bit levels over the edge region (ground-truth 0 < alpha < 1, grown by 10 px).
"""
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

bdir, rdir = sys.argv[1:3]
out_csv = sys.argv[3] if len(sys.argv) > 3 else None
METHODS = [("rembg (U2-Net)", "__rembg_u2net.png"), ("rembg (IS-Net)", "__rembg_isnet.png"),
           ("InSPyReNet", "__inspyrenet.png"), ("BiRefNet-HR", "_plain.png"),
           ("BiRefNet-HR + pymatting", "__birefnet_pymatting.png"), ("inkcut", "_cut.png")]
DARK = np.array([40.0, 40.0, 40.0])


def grad_mag(a, sigma=1.4):
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1))
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0))
    return np.sqrt(gx ** 2 + gy ** 2)


ids = sorted(os.path.basename(p)[:-10] for p in glob.glob(os.path.join(bdir, "gt", "*_alpha.png")))
rows = {m: [] for m, _ in METHODS}
for i in ids:
    ag = np.asarray(Image.open(os.path.join(bdir, "gt", i + "_alpha.png"))).astype(np.float64) / 255
    fg = np.asarray(Image.open(os.path.join(bdir, "gt", i + "_fg.png")).convert("RGB")).astype(np.float64)
    gt_comp = fg * ag[..., None] + DARK * (1 - ag[..., None])
    region = ndimage.binary_dilation((ag > 0.01) & (ag < 0.99), iterations=10)
    g_gt = grad_mag(ag)
    for m, suf in METHODS:
        p = os.path.join(rdir, i + suf)
        if not os.path.exists(p):
            continue
        im = np.asarray(Image.open(p).convert("RGBA")).astype(np.float64)
        if im.shape[:2] != ag.shape:
            im = np.asarray(Image.fromarray(im.astype(np.uint8), "RGBA").resize((ag.shape[1], ag.shape[0]), Image.BILINEAR)).astype(np.float64)
        a = im[..., 3] / 255
        comp = im[..., :3] * a[..., None] + DARK * (1 - a[..., None])
        sad = np.abs(a - ag).sum() / 1000
        mse = ((a - ag) ** 2).mean() * 1000
        grad = ((grad_mag(a) - g_gt) ** 2).sum() / 1000
        rec = np.abs(comp - gt_comp)[region].mean() if region.any() else 0.0
        rows[m].append((i, sad, mse, grad, rec))

print(f"{len(ids)} images\n{'method':26s} {'SAD':>8s} {'MSE':>8s} {'Grad':>8s} {'recomp':>8s}  (lower is better)")
summary = []
for m, _ in METHODS:
    r = np.array([x[1:] for x in rows[m]]) if rows[m] else np.zeros((0, 4))
    if len(r):
        mean = r.mean(axis=0)
        summary.append((m, *mean, len(r)))
        print(f"{m:26s} {mean[0]:8.2f} {mean[1]:8.2f} {mean[2]:8.2f} {mean[3]:8.2f}  n={len(r)}")
if out_csv:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "SAD", "MSE_x1e3", "Grad", "recomposite_MAE", "n"])
        w.writerows(summary)
