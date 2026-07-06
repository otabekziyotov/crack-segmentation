import numpy as np
import albumentations as A
from glob import glob
from PIL import Image, ImageFile
from pathlib import Path
from torch.utils.data import random_split, Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

# transforms: PIL class
# albumentations: array (cv2.imread, plt.imread) -> np.array
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project root (folder of this file) — works on any machine
PROJECT_ROOT = Path(__file__).resolve().parent


class CrackDataset(Dataset):
    def __init__(self, im_dir, gt_dir, transformations=None):
        self.im_paths = sorted(glob(f"{im_dir}/*.*"))
        self.gt_paths = sorted(glob(f"{gt_dir}/*.*"))

        print(f"Found {len(self.im_paths)} images and {len(self.gt_paths)} masks in {im_dir}")
        self.transformations = transformations
        self.n_cls = 2  # binary segmentation: 0 (black) - background, 1 (white) - crack
        assert len(self.im_paths) == len(self.gt_paths), "Mismatch between images and masks!"

    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        im, gt = self.get_im_gt(self.im_paths[idx], self.gt_paths[idx])

        if self.transformations:
            im, gt = self.apply_transformations(im, gt)  # gt mask (0~255)

        return im, (gt / 255).int()

    def get_im_gt(self, im_path, gt_path): return self.read_im(im_path, gt_path)

    def read_im(self, im_path, gt_path):
        # L -> grayscale
        return np.array(Image.open(im_path).convert("RGB")), np.array(Image.open(gt_path).convert("L"))

    def apply_transformations(self, im, gt):
        transformed = self.transformations(image=im, mask=gt)
        return transformed["image"], transformed["mask"]


def get_dls(root, transformations, bs, val_split=0.5, ns=2):

    # 1. Load the FULL training dataset (100% used for training)
    print("Loading Training Data:")
    tr_ds = CrackDataset(
        im_dir=f"{root}/train_img",
        gt_dir=f"{root}/train_lab",
        transformations=transformations
    )
    n_cls = tr_ds.n_cls

    # 2. Load the FULL test dataset
    print("\nLoading Test Data:")
    full_test_ds = CrackDataset(
        im_dir=f"{root}/test_img",
        gt_dir=f"{root}/test_lab",
        transformations=transformations
    )

    # 3. Split the test dataset into Validation and Test
    val_len = int(len(full_test_ds) * val_split)
    test_len = len(full_test_ds) - val_len

    val_ds, test_ds = random_split(full_test_ds, [val_len, test_len])

    print(f"\nThere are {len(tr_ds)} images in the train set (100% of train folder)")
    print(f"There are {len(val_ds)} images in the validation set ({int(val_split * 100)}% of test folder)")
    print(f"There are {len(test_ds)} images in the test set ({int((1 - val_split) * 100)}% of test folder)\n")

    # 4. Create DataLoaders for all three
    tr_dl  = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True,  num_workers=ns)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=ns)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False, num_workers=ns)

    return tr_dl, val_dl, test_dl, n_cls


if __name__ == "__main__":
    root = PROJECT_ROOT / "datasetlar" / "cracks"

    mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256

    trans = A.Compose([
        A.Resize(im_h, im_w),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)

    tr_dl, val_dl, test_dl, n_cls = get_dls(root=root, transformations=trans, bs=8)
