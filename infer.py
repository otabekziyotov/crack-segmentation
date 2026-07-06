import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as tfs

import model as _model

# The UNet was saved as a full object while its classes lived in __main__
# (Colab notebook / train.py's __main__). Re-expose them so torch.load can
# resolve the pickled class references when loading the saved .pt files.
for _name in ["UNet", "UNetBlock", "DownSampling", "UpSampling", "FinalConv"]:
    setattr(sys.modules["__main__"], _name, getattr(_model, _name))


def compare_inference_results(test_dataloader, unet_model_path, deeplab_model_path, device,
                              n_images=5, save_dir=None):
    # Load models
    unet_model = torch.load(unet_model_path, map_location=device, weights_only=False)
    unet_model.to(device)
    unet_model.eval()

    deeplab_model = torch.load(deeplab_model_path, map_location=device, weights_only=False)
    deeplab_model.to(device)
    deeplab_model.eval()

    # Inverse transform for plotting
    # These mean and std values should match those used during training/normalization
    invTrans = tfs.Compose([
        tfs.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        tfs.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])

    def tn_2_np_local(t, is_rgb=True):
        if is_rgb:
            t_denorm = invTrans(t)
            np_array = (t_denorm * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            # For masks, they are typically 0 or 1, and multiplied by 255 for visualization
            np_array = (t * 255).detach().cpu().numpy().astype(np.uint8)
        return np_array

    fig, axes = plt.subplots(n_images, 4, figsize=(20, n_images * 5))
    # Ensure axes is always a 2D array for consistent indexing, even if n_images=1
    if n_images == 1:
        axes = np.array([axes])

    image_idx = 0
    for i, (im, gt) in enumerate(test_dataloader):
        if image_idx >= n_images:
            break

        im_device = im.to(device)

        # UNet prediction
        with torch.no_grad():
            unet_pred_logits = unet_model(im_device)
            unet_pred = torch.argmax(unet_pred_logits, dim=1)

        # DeepLabV3Plus prediction
        with torch.no_grad():
            deeplab_pred_logits = deeplab_model(im_device)
            deeplab_pred = torch.argmax(deeplab_pred_logits, dim=1)

        # Plot original image
        axes[image_idx, 0].imshow(tn_2_np_local(im.squeeze(0)), cmap="gray")
        axes[image_idx, 0].set_title("Original Image")
        axes[image_idx, 0].axis("off")

        # Plot ground truth
        axes[image_idx, 1].imshow(tn_2_np_local(gt.squeeze(0), is_rgb=False), cmap="gray")
        axes[image_idx, 1].set_title("Ground Truth")
        axes[image_idx, 1].axis("off")

        # Plot UNet prediction
        axes[image_idx, 2].imshow(tn_2_np_local(unet_pred.squeeze(0), is_rgb=False), cmap="gray")
        axes[image_idx, 2].set_title("UNet Prediction")
        axes[image_idx, 2].axis("off")

        # Plot DeepLabV3Plus prediction
        axes[image_idx, 3].imshow(tn_2_np_local(deeplab_pred.squeeze(0), is_rgb=False), cmap="gray")
        axes[image_idx, 3].set_title("DeepLabV3Plus Prediction")
        axes[image_idx, 3].axis("off")

        image_idx += 1

    plt.tight_layout()

    # If save_dir is set -> save to file, otherwise -> show on screen
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "inference_comparison.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saqlandi -> {path}")
    else:
        plt.show()


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from pathlib import Path
    from custom_dataset import get_dls

    PROJECT_ROOT = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = PROJECT_ROOT / "datasetlar" / "cracks"

    mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
    trans = A.Compose([
        A.Resize(im_h, im_w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)
    tr_dl, val_dl, test_dl, n_cls = get_dls(root=root, transformations=trans, bs=8)

    unet_model_path = PROJECT_ROOT / "saved_models_unet" / "crack_unet_comparison_best_model.pt"
    deeplab_model_path = PROJECT_ROOT / "saved_models_deeplab" / "crack_deeplab_best_model.pt"

    compare_inference_results(
        test_dataloader=test_dl,
        unet_model_path=str(unet_model_path),
        deeplab_model_path=str(deeplab_model_path),
        device=device,
        n_images=5,
        save_dir=PROJECT_ROOT / "results"
    )
