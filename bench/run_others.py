"""Run competing background removers on images; inkcut and BiRefNet outputs must already exist.

Usage: python run_others.py <img_dir> <out_dir>
Needs: pip install rembg onnxruntime transparent-background pymatting
For each <name> in img_dir expects out_dir/<name>_mask.png (BiRefNet-HR) and writes
out_dir/<name>__<method>.png (RGBA) for: rembg_u2net, rembg_isnet, inspyrenet, birefnet_pymatting.
"""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

img_dir, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
paths = [p for p in sorted(glob.glob(os.path.join(img_dir, "*"))) if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

from rembg import new_session, remove  # noqa: E402

sessions = {"rembg_u2net": new_session("u2net"), "rembg_isnet": new_session("isnet-general-use")}
from transparent_background import Remover  # noqa: E402

remover = Remover(mode="base", device="cpu")
from pymatting import estimate_foreground_ml  # noqa: E402

for p in paths:
    name = os.path.splitext(os.path.basename(p))[0]
    img = Image.open(p).convert("RGB")
    for key, sess in sessions.items():
        t = time.time()
        remove(img, session=sess).save(os.path.join(out_dir, f"{name}__{key}.png"))
        print(f"{name} {key} {time.time() - t:.0f}s", flush=True)
    t = time.time()
    remover.process(img, type="rgba").save(os.path.join(out_dir, f"{name}__inspyrenet.png"))
    print(f"{name} inspyrenet {time.time() - t:.0f}s", flush=True)
    mask_path = os.path.join(out_dir, f"{name}_mask.png")
    if os.path.exists(mask_path):
        t = time.time()
        a = np.asarray(Image.open(mask_path)).astype(np.float64) / 255
        fg = estimate_foreground_ml(np.asarray(img).astype(np.float64) / 255, a)
        rgba = np.dstack([fg * 255, a * 255]).clip(0, 255).round().astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(out_dir, f"{name}__birefnet_pymatting.png"))
        print(f"{name} birefnet+pymatting {time.time() - t:.0f}s", flush=True)
