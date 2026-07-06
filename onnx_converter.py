"""Export the trained crack-segmentation models to ONNX and verify they match PyTorch."""
import sys
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path

# The ONNX exporter prints unicode (e.g. checkmarks); force UTF-8 so it
# doesn't crash on consoles with a non-UTF-8 codepage (e.g. Windows cp1251).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import onnx
import onnxruntime as ort

import model as _model

# The UNet was saved as a full object while its classes lived in __main__
# (Colab notebook / train.py's __main__). Re-expose them so torch.load can
# resolve the pickled class references.
for _name in ["UNet", "UNetBlock", "DownSampling", "UpSampling", "FinalConv"]:
    setattr(sys.modules["__main__"], _name, getattr(_model, _name))


# ----------------------- CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent

MODELS = {
    "unet": PROJECT_ROOT / "saved_models_unet" / "crack_unet_comparison_best_model.pt",
    "deeplab": PROJECT_ROOT / "saved_models_deeplab" / "crack_deeplab_best_model.pt",
}
ONNX_DIR = PROJECT_ROOT / "onnx_models"

IM_H, IM_W = 256, 256
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
OPSET = 17
TEST_DIR = PROJECT_ROOT / "datasetlar" / "cracks" / "test_img"   # for the sanity check
# ------------------------------------------------------


def get_transform():
    return A.Compose([
        A.Resize(IM_H, IM_W),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], is_check_shapes=False)


def sample_input():
    """Real test image if available, otherwise random noise -> (1, 3, IM_H, IM_W) tensor."""
    imgs = sorted(TEST_DIR.glob("*.*")) if TEST_DIR.exists() else []
    if imgs:
        np_im = np.array(Image.open(imgs[0]).convert("RGB"))
        return get_transform()(image=np_im)["image"].unsqueeze(0)
    return torch.randn(1, 3, IM_H, IM_W)


def convert_one(name, weights):
    if not weights.exists():
        print(f"[skip] {name}: weights not found at {weights}")
        return

    onnx_path = ONNX_DIR / f"crack_{name}.onnx"

    # 1) Load the trained PyTorch model (full object, CPU is enough for export)
    model = torch.load(weights, map_location="cpu", weights_only=False)
    model.eval()

    # 2) Export to ONNX (dynamic batch + spatial axes -> any size works at inference)
    dummy_input = torch.randn(1, 3, IM_H, IM_W)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        dynamo=False,   # classic exporter (respects opset & dynamic_axes directly)
    )
    print(f"Exported ONNX -> {onnx_path}")

    # 3) Validate the ONNX graph structure
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[{name}] ONNX model structure is valid.")

    # 4) Verify PyTorch vs ONNX Runtime on a real image
    x = sample_input()
    with torch.no_grad():
        torch_out = model(x).numpy()

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: x.numpy()})[0]

    # Compare both the raw logits and the final predicted masks
    max_diff = np.abs(torch_out - onnx_out).max()
    torch_mask = np.argmax(torch_out, axis=1)
    onnx_mask = np.argmax(onnx_out, axis=1)
    mask_agree = (torch_mask == onnx_mask).mean() * 100

    print(f"[{name}] Max logit difference PyTorch vs ONNX: {max_diff:.2e}")
    print(f"[{name}] Predicted masks agree on {mask_agree:.2f}% of pixels")
    print(f"[{name}] PASSED ✓\n" if max_diff < 1e-3 else f"[{name}] WARNING: outputs differ more than expected.\n")


def main():
    ONNX_DIR.mkdir(exist_ok=True)
    for name, weights in MODELS.items():
        convert_one(name, weights)


if __name__ == "__main__":
    main()
