import torch
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from downloader import DatasetDownloader
from custom_dataset import get_dls
from model import UNet
from vis import DataVisualizer
from train import Trainer
from plot import plot_comparison_metrics
from infer import compare_inference_results


# ----------------------- CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DS_NOMI   = "cracks"
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]
IM_H      = 256
IM_W      = 256
BS        = 8
EPOCHS    = 30
LR        = 3e-4
PATIENCE  = 5

UNET_SAVE_DIR    = PROJECT_ROOT / "saved_models_unet"
DEEPLAB_SAVE_DIR = PROJECT_ROOT / "saved_models_deeplab"
UNET_PREFIX      = "crack_unet_comparison"
DEEPLAB_PREFIX   = "crack_deeplab"

# All result images (for the GitHub README) are saved here
RESULTS_DIR = PROJECT_ROOT / "results"
# ------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) Download dataset (idempotent)
    root = DatasetDownloader(save_dir=PROJECT_ROOT / "datasetlar").download(ds_nomi=DS_NOMI)

    # 2) Transforms + DataLoaders
    trans = A.Compose([
        A.Resize(IM_H, IM_W),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)
    tr_dl, val_dl, test_dl, n_cls = get_dls(root=root, transformations=trans, bs=BS)

    # 3) Dataset visualization
    DataVisualizer(mean=MEAN, std=STD, save_dir=RESULTS_DIR).visualize_dataset(val_dl.dataset, num_images=20)

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4) Train UNet (from scratch)
    unet = UNet(in_chs=3, out_chs=64, n_cls=n_cls, up_method="tr_conv")
    unet_history = Trainer(
        model=unet, tr_dl=tr_dl, val_dl=val_dl, loss_fn=loss_fn,
        optimizer=torch.optim.Adam(params=unet.parameters(), lr=LR),
        device=device, n_cls=n_cls, save_path=str(UNET_SAVE_DIR),
        early_stop_threshold=PATIENCE
    ).run(epochs=EPOCHS, save_prefix=UNET_PREFIX)

    # 5) Train DeepLabV3Plus (library model)
    deeplab = smp.DeepLabV3Plus(classes=n_cls)
    deeplab_history = Trainer(
        model=deeplab, tr_dl=tr_dl, val_dl=val_dl, loss_fn=loss_fn,
        optimizer=torch.optim.Adam(params=deeplab.parameters(), lr=LR),
        device=device, n_cls=n_cls, save_path=str(DEEPLAB_SAVE_DIR),
        early_stop_threshold=PATIENCE
    ).run(epochs=EPOCHS, save_prefix=DEEPLAB_PREFIX)

    # 6) Learning-curve comparison
    plot_comparison_metrics(unet_history, deeplab_history, "UNet", "DeepLabV3Plus", save_dir=RESULTS_DIR)

    # 7) Visual inference comparison (UNet vs DeepLabV3Plus)
    compare_inference_results(
        test_dataloader=test_dl,
        unet_model_path=str(UNET_SAVE_DIR / f"{UNET_PREFIX}_best_model.pt"),
        deeplab_model_path=str(DEEPLAB_SAVE_DIR / f"{DEEPLAB_PREFIX}_best_model.pt"),
        device=device,
        n_images=5,
        save_dir=RESULTS_DIR
    )

    print(f"\nBarcha bosqichlar tugadi! Natijalar saqlandi -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
