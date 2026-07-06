import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as tfs


class DataVisualizer:
    def __init__(self, mean, std, save_dir=None):
        self.mean = mean
        self.std = std
        self.inv_transform = self._get_inverse_transform()

        # If save_dir is given -> save images to file, otherwise show them on screen
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def _get_inverse_transform(self):
        return tfs.Compose([
            tfs.Normalize(mean=[0., 0., 0.], std=[1 / s for s in self.std]),
            tfs.Normalize(mean=[-m for m in self.mean], std=[1., 1., 1.])
        ])

    def tensor_to_numpy(self, tensor):
        is_rgb = len(tensor) == 3
        if is_rgb:
            tensor = self.inv_transform(tensor)
            np_array = (tensor * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            np_array = (tensor * 255).detach().cpu().numpy().astype(np.uint8)
        return np_array

    def _plot(self, rows, cols, count, image, is_gt=False, title="Original Image"):
        plt.subplot(rows, cols, count)  # e.g. 12: (3, 4, 1~12)
        plt.imshow(
            self.tensor_to_numpy(image.squeeze(0).float()) if is_gt else self.tensor_to_numpy(image.squeeze(0)),
            cmap="gray"
        )
        plt.axis("off")
        plt.title(title)
        return count + 1

    def _save_or_show(self, filename):
        # If save_dir is set -> save to file, otherwise -> show on screen
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"Saqlandi -> {path}")
        else:
            plt.show()

    def visualize_dataset(self, dataset, num_images, save_name="dataset_samples.png"):
        plt.figure(figsize=(25, 20))
        rows = num_images // 4
        cols = num_images // rows
        count = 1
        indices = [random.randint(0, len(dataset) - 1) for _ in range(num_images)]

        for index in indices:
            if count > num_images:
                break
            image, mask = dataset[index]
            # Plot the original image
            count = self._plot(rows, cols, count, image)
            # Plot the ground truth mask
            count = self._plot(rows, cols, count, mask, is_gt=True, title="GT Mask")

        self._save_or_show(save_name)


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from pathlib import Path
    from custom_dataset import get_dls

    PROJECT_ROOT = Path(__file__).resolve().parent
    root = PROJECT_ROOT / "datasetlar" / "cracks"

    mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
    trans = A.Compose([
        A.Resize(im_h, im_w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)

    tr_dl, val_dl, test_dl, n_cls = get_dls(root=root, transformations=trans, bs=8)

    visualizer = DataVisualizer(mean=mean, std=std, save_dir=PROJECT_ROOT / "results")
    visualizer.visualize_dataset(val_dl.dataset, num_images=20)
