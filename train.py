# !pip install segmentation_models_pytorch
import os, time
import numpy as np
import torch
import albumentations as A
from tqdm import tqdm
from pathlib import Path
from albumentations.pytorch import ToTensorV2

from downloader import DatasetDownloader
from custom_dataset import get_dls
from model import UNet

# Project root (folder of this file) — works on any machine
PROJECT_ROOT = Path(__file__).resolve().parent


class Metrics():
    def __init__(self, pred, gt, loss_fn, eps=1e-10, n_cls=2):
        self.pred, self.gt = torch.argmax(pred, dim=1), gt.squeeze(1)  # (batch, width, height)
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred

    def to_contiguous(self, inp): return inp.contiguous().view(-1)  # memory

    def PA(self):
        with torch.no_grad():
            match = torch.eq(self.pred, self.gt).int()  # acc += (self.pred == self.gt)

        return float(match.sum()) / float(match.numel())  # number of elements

    def mIoU(self):  # mean intersection over union
        with torch.no_grad():

            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            iou_per_class = []

            for c in range(self.n_cls):  # 0, 1

                match_pred = pred == c
                match_gt   = gt == c

                if match_gt.long().sum().item() == 0:
                    iou_per_class.append(np.nan)  # not a value

                else:
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()  # tensor -> float
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()

                    iou = intersect / (union + self.eps)  # numeric stability
                    iou_per_class.append(iou)

            return np.nanmean(iou_per_class)  # nanmean -> drops nan then averages

    def loss(self): return self.loss_fn(self.pred_, self.gt.long())  # int


def tic_toc(start_time=None): return time.time() if start_time is None else time.time() - start_time


class Trainer:
    def __init__(self, model, tr_dl, val_dl, loss_fn, optimizer, device, n_cls,
                 save_path="saved_models", early_stop_threshold=5, threshold=0.005):

        self.model = model
        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.n_cls = n_cls
        self.save_path = save_path
        self.early_stop_threshold = early_stop_threshold
        self.threshold = threshold

        os.makedirs(self.save_path, exist_ok=True)

    def run(self, epochs, save_prefix):

        self.model.to(self.device)

        tr_loss, tr_pa, tr_iou = [], [], []
        val_loss, val_pa, val_iou = [], [], []
        best_loss, not_improve = np.inf, 0

        print("Starting training process...")
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # Training Phase
            self.model.train()  # self.model.eval() -> validation & test (inference, deployment)
            train_metrics = self._process_epoch(self.tr_dl, is_training=True)

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                val_metrics = self._process_epoch(self.val_dl, is_training=False)

            # Log Metrics
            tr_loss.append(train_metrics["loss"])
            tr_iou.append(train_metrics["iou"])
            tr_pa.append(train_metrics["pa"])
            val_loss.append(val_metrics["loss"])
            val_iou.append(val_metrics["iou"])
            val_pa.append(val_metrics["pa"])

            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Train Loss: {train_metrics['loss']:.3f} | Train PA: {train_metrics['pa']:.3f} | Train IoU: {train_metrics['iou']:.3f}")
            print(f"Val Loss:   {val_metrics['loss']:.3f} | Val PA:   {val_metrics['pa']:.3f} | Val IoU:   {val_metrics['iou']:.3f}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            # Save the Best Model
            if best_loss > (val_metrics["loss"] + self.threshold):
                print(f"Validation loss decreased from {best_loss:.3f} to {val_metrics['loss']:.3f}. Saving model...")
                best_loss = val_metrics["loss"]
                not_improve = 0
                torch.save(self.model, f"{self.save_path}/{save_prefix}_best_model.pt")
            else:
                not_improve += 1
                print(f"No improvement for {not_improve} epoch(s).")
                if not_improve >= self.early_stop_threshold:
                    print(f"Early stopping: No improvement for {self.early_stop_threshold} epochs.")
                    break

        return {
            "tr_loss": tr_loss, "tr_iou": tr_iou, "tr_pa": tr_pa,
            "val_loss": val_loss, "val_iou": val_iou, "val_pa": val_pa,
        }

    def _process_epoch(self, dataloader, is_training):

        phase = "Train" if is_training else "Validation"
        print(f"{phase} phase started...")

        total_loss, total_iou, total_pa = 0, 0, 0
        for ims, gts in tqdm(dataloader, desc=f"{phase} Progress"):
            ims, gts = ims.to(self.device), gts.to(self.device)

            if is_training:
                preds = self.model(ims)  # segmentation: predicted segmentation mask (logits)
                metrics = Metrics(preds, gts, self.loss_fn, n_cls=self.n_cls)
                loss = metrics.loss()  # computes loss

                self.optimizer.zero_grad()  # zero grad
                loss.backward()  # backprop
                self.optimizer.step()  # optimization
            else:  # validation
                with torch.no_grad():
                    preds = self.model(ims)
                    metrics = Metrics(preds, gts, self.loss_fn, n_cls=self.n_cls)
                    loss = metrics.loss()

            # Accumulate Metrics
            total_loss += loss.item()
            total_iou += metrics.mIoU()
            total_pa += metrics.PA()

        num_batches = len(dataloader)  # to compute mean
        return {
            "loss": total_loss / num_batches,
            "iou": total_iou / num_batches,
            "pa": total_pa / num_batches,
        }


if __name__ == "__main__":
    import segmentation_models_pytorch as smp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Download the dataset (idempotent)
    root = DatasetDownloader().download(ds_nomi="cracks")

    # 2. Build dataloaders
    mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
    trans = A.Compose([
        A.Resize(im_h, im_w),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)
    tr_dl, val_dl, test_dl, n_cls = get_dls(root=root, transformations=trans, bs=8)

    loss_fn = torch.nn.CrossEntropyLoss()

    # 3. Train UNet (from scratch)
    m1 = UNet(in_chs=3, out_chs=64, n_cls=n_cls, up_method="tr_conv")
    optimizer_unet = torch.optim.Adam(params=m1.parameters(), lr=3e-4)
    trainer_unet = Trainer(
        model=m1, tr_dl=tr_dl, val_dl=val_dl, loss_fn=loss_fn,
        optimizer=optimizer_unet, device=device, n_cls=n_cls,
        save_path=str(PROJECT_ROOT / "saved_models_unet")
    )
    unet_history = trainer_unet.run(epochs=30, save_prefix="crack_unet_comparison")

    # 4. Train DeepLabV3Plus (pretrained library model)
    model_deeplab = smp.DeepLabV3Plus(classes=n_cls)
    optimizer_deeplab = torch.optim.Adam(params=model_deeplab.parameters(), lr=3e-4)
    trainer_deeplab = Trainer(
        model=model_deeplab, tr_dl=tr_dl, val_dl=val_dl, loss_fn=loss_fn,
        optimizer=optimizer_deeplab, device=device, n_cls=n_cls,
        save_path=str(PROJECT_ROOT / "saved_models_deeplab")
    )
    deeplab_history = trainer_deeplab.run(epochs=30, save_prefix="crack_deeplab")
