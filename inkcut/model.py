"""BiRefNet-HR (ZhengPeng7/BiRefNet_HR, MIT) loader and prediction."""
import importlib.util
import os
import sys
import types

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

HF_REPO = "ZhengPeng7/BiRefNet_HR"
CODE_FILES = ("birefnet.py", "BiRefNet_config.py")
WEIGHT_NAMES = ("model.safetensors", "BiRefNet-HR.safetensors")


def model_files(model_dir=None):
    """(code dir, weights path) from a local folder, else downloaded once from Hugging Face."""
    if model_dir:
        for weights in WEIGHT_NAMES:
            if all(os.path.isfile(os.path.join(model_dir, f)) for f in CODE_FILES + (weights,)):
                return model_dir, os.path.join(model_dir, weights)
        raise FileNotFoundError(f"{model_dir} needs {', '.join(CODE_FILES)} and one of {', '.join(WEIGHT_NAMES)}")
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(HF_REPO, f) for f in CODE_FILES + ("model.safetensors",)]
    return os.path.dirname(paths[0]), paths[2]


def load_model(device="cpu", model_dir=None):
    code_dir, weights = model_files(model_dir)
    cfg_spec = importlib.util.spec_from_file_location("BiRefNet_config", os.path.join(code_dir, "BiRefNet_config.py"))
    cfg = importlib.util.module_from_spec(cfg_spec)
    sys.modules["BiRefNet_config"] = cfg
    cfg_spec.loader.exec_module(cfg)
    # The model file uses a package-relative import; load it from text with that made absolute.
    with open(os.path.join(code_dir, "birefnet.py"), encoding="utf-8") as f:
        code = f.read().replace("from .BiRefNet_config", "from BiRefNet_config")
    net = types.ModuleType("birefnet")
    sys.modules["birefnet"] = net
    exec(compile(code, os.path.join(code_dir, "birefnet.py"), "exec"), net.__dict__)

    model = net.BiRefNet(cfg.BiRefNetConfig())
    model.load_state_dict({k: v.float() for k, v in load_file(weights).items()})
    model.eval()
    return model.half().to(device) if device == "cuda" else model.float()


def predict(model, img, res=2048, device="cpu"):
    """Alpha mask 0..1 at the image's size."""
    tf = transforms.Compose([
        transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0)
    x = x.to(device).half() if device == "cuda" else x
    with torch.inference_mode():
        pred = model(x)[-1].sigmoid()[0].squeeze().float().cpu()
    mask = transforms.ToPILImage()(pred).resize(img.size, Image.BICUBIC)
    return np.asarray(mask).astype(np.float32) / 255.0
