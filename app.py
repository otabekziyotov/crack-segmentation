import sys
import numpy as np
import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path

import model as _model

# The UNet was saved as a full object (torch.save(model, ...)) while its classes
# lived in __main__ (Colab notebook / train.py's __main__). Re-expose them in
# __main__ so torch.load can resolve the pickled class references.
for _name in ["UNet", "UNetBlock", "DownSampling", "UpSampling", "FinalConv"]:
    setattr(sys.modules["__main__"], _name, getattr(_model, _name))


# ----------------------- CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Trained models are saved as full objects via torch.save(model, ...) in train.py
MODELS = {
    "UNet": PROJECT_ROOT / "saved_models_unet" / "crack_unet_comparison_best_model.pt",
    "DeepLabV3Plus": PROJECT_ROOT / "saved_models_deeplab" / "crack_deeplab_best_model.pt",
}

IM_H, IM_W = 256, 256
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prefer the committed demo_samples/ (works on Streamlit Cloud); fall back to the
# full dataset locally. Each folder holds images + matching masks (by filename stem).
if (PROJECT_ROOT / "demo_samples" / "images").exists():
    TEST_DIR = PROJECT_ROOT / "demo_samples" / "images"   # sample images
    LAB_DIR  = PROJECT_ROOT / "demo_samples" / "masks"    # ground-truth masks
else:
    TEST_DIR = PROJECT_ROOT / "datasetlar" / "cracks" / "test_img"
    LAB_DIR  = PROJECT_ROOT / "datasetlar" / "cracks" / "test_lab"

N_SAMPLES = 6                                             # thumbnails in the sidebar
# ------------------------------------------------------


@st.cache_resource
def load_model(model_path):
    """Load a full trained model object (cached across reruns)."""
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    return model.to(DEVICE).eval()


def get_transform():
    return A.Compose([
        A.Resize(IM_H, IM_W),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], is_check_shapes=False)


def list_samples(n):
    """Return up to n sample image paths from the test set."""
    if not TEST_DIR.exists():
        return []
    return sorted(TEST_DIR.glob("*.*"))[:n]


def get_gt_mask(im_path):
    """Load the ground-truth mask for a test image (matched by filename stem)."""
    gt_path = LAB_DIR / f"{Path(im_path).stem}.png"
    if not gt_path.exists():
        return None
    return Image.open(gt_path).convert("L").resize((IM_W, IM_H))


def predict(model, image):
    """Run inference and return the predicted binary mask (H, W) at IM_H x IM_W."""
    np_im = np.array(image.convert("RGB"))
    x = get_transform()(image=np_im)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        mask = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return mask


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay the predicted crack mask (in `color`) on the original image."""
    base = np.array(image.convert("RGB").resize((IM_W, IM_H))).astype(np.float32)
    colored = np.zeros_like(base)
    colored[mask == 1] = color
    blended = np.where(mask[..., None] == 1, (1 - alpha) * base + alpha * colored, base)
    return blended.astype(np.uint8)


def mask_metrics(pred, gt, eps=1e-10):
    """Segmentation scores for the CRACK class (pred/gt are 0/1 arrays)."""
    pred_c, gt_c = (pred == 1), (gt == 1)
    inter = np.logical_and(pred_c, gt_c).sum()
    union = np.logical_or(pred_c, gt_c).sum()

    iou = inter / (union + eps)
    dice = 2 * inter / (pred_c.sum() + gt_c.sum() + eps)
    pa = (pred == gt).mean()
    # If the image truly has no crack and the model predicted none -> perfect
    if union == 0:
        iou = dice = 1.0
    return {"iou": float(iou), "dice": float(dice), "pa": float(pa)}


def match_map(pred, gt):
    """Colour each pixel: green = correct crack (TP), red = missed (FN), blue = false alarm (FP)."""
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    pred_c, gt_c = (pred == 1), (gt == 1)
    out[np.logical_and(pred_c, gt_c)]  = (0, 200, 0)     # TP - correctly found crack
    out[np.logical_and(~pred_c, gt_c)] = (220, 0, 0)     # FN - missed crack
    out[np.logical_and(pred_c, ~gt_c)] = (0, 90, 255)    # FP - false alarm
    return out


def verdict(iou):
    """Human-readable quality label from the IoU score."""
    if iou >= 0.75: return "Excellent ✅", "#22c55e"
    if iou >= 0.5:  return "Good 👍", "#84cc16"
    if iou >= 0.3:  return "Fair ⚠️", "#f59e0b"
    return "Poor ❌", "#ef4444"


# --------------------------- UI ---------------------------
st.set_page_config(page_title="Crack Segmentation", page_icon="🧱")
st.title("🧱 Crack Segmentation")
st.write("Upload a surface image and the model will segment the **cracks** pixel by pixel.")

# ----- Sidebar: model choice + sample images -----
st.sidebar.header("⚙️ Options")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()), index=0)
weights = MODELS[model_name]

if not weights.exists():
    st.error(f"Trained weights not found at `{weights}`.\n\n"
             "Run `python main.py` (or `python train.py`) first to train and save the models.")
    st.stop()

model = load_model(str(weights))

if "sample_path" not in st.session_state:
    st.session_state.sample_path = None

st.sidebar.header("🖼️ Test set samples")
st.sidebar.caption("Click a sample to run it through the model.")
samples = list_samples(N_SAMPLES)
if not samples:
    st.sidebar.write("No samples found. Run `python downloader.py` to fetch the dataset.")
for p in samples:
    st.sidebar.image(str(p), use_container_width=True)
    if st.sidebar.button(f"Use {p.name}", key=str(p)):
        st.session_state.sample_path = str(p)

# ----- Main: pick input (uploaded file OR selected sample) -----
file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

image, source, gt_mask = None, "", None
if file is not None:
    image = Image.open(file).convert("RGB")
    source = "Uploaded image"
    st.session_state.sample_path = None          # an upload overrides the sample
elif st.session_state.sample_path:
    image = Image.open(st.session_state.sample_path).convert("RGB")
    source = f"Sample: {Path(st.session_state.sample_path).name}"
    gt_mask = get_gt_mask(st.session_state.sample_path)   # ground truth (samples only)

if image is not None:
    mask = predict(model, image)
    crack_ratio = float(mask.mean()) * 100      # % of pixels predicted as crack

    if gt_mask is not None:
        # ----- We have ground truth -> score how well the model did -----
        gt = (np.array(gt_mask) > 127).astype(np.uint8)   # 0/1 mask at IM_H x IM_W
        scores = mask_metrics(mask, gt)
        label, color = verdict(scores["iou"])

        st.markdown(
            f"### {model_name} result — "
            f"<span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("IoU (overlap)", f"{scores['iou'] * 100:.1f}%")
        m2.metric("Dice / F1", f"{scores['dice'] * 100:.1f}%")
        m3.metric("Pixel accuracy", f"{scores['pa'] * 100:.1f}%")
        st.caption("IoU = how much the predicted crack overlaps the true crack (higher is better).")

        c1, c2, c3, c4 = st.columns(4)
        c1.image(image, caption=source, use_container_width=True)
        c2.image(gt_mask, caption="Ground-truth mask", use_container_width=True)
        c3.image(mask * 255, caption="Predicted mask", use_container_width=True, clamp=True)
        c4.image(match_map(mask, gt), caption="Match map", use_container_width=True)
        st.caption("🟢 correctly found crack  ·  🔴 missed crack  ·  🔵 false alarm")
    else:
        # ----- Uploaded image: no ground truth to score against -----
        st.subheader(f"{model_name} — crack covers {crack_ratio:.2f}% of the image")
        st.caption("No ground-truth mask for uploaded images, so accuracy can't be measured — "
                   "pick a test sample from the sidebar to see IoU/Dice scores.")

        c1, c2, c3 = st.columns(3)
        c1.image(image, caption=source, use_container_width=True)
        c2.image(mask * 255, caption="Predicted mask", use_container_width=True, clamp=True)
        c3.image(overlay_mask(image, mask), caption="Overlay", use_container_width=True)
else:
    st.info("⬆️ Upload an image or pick a sample from the sidebar.")
