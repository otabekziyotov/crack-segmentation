# !pip install kagglehub
import os, shutil, kagglehub
from pathlib import Path


# Project root (folder of this file) — works on any machine
PROJECT_ROOT = Path(__file__).resolve().parent


class DatasetDownloader:
    def __init__(self, save_dir=PROJECT_ROOT / "datasetlar"):
        self.save_dir = Path(save_dir)
        self.available_datasets = {
            "cracks": "rukiyeaydn/deepcrack-dataset",
        }

    def download(self, ds_nomi="cracks"):
        assert ds_nomi in self.available_datasets, \
            f"Mavjud bo'lgan datasetlardan birini tanlang: {list(self.available_datasets.keys())}"

        dataset_path = self.save_dir / ds_nomi

        # Check whether the data is already downloaded
        if dataset_path.is_dir() and any(dataset_path.iterdir()):
            print(f"{ds_nomi} dataset allaqachon yuklab olingan: {dataset_path}")
            return dataset_path

        os.makedirs(self.save_dir, exist_ok=True)

        # Download the latest version into kagglehub cache
        print(f"{ds_nomi} dataset yuklanmoqda...")
        cache_path = kagglehub.dataset_download(self.available_datasets[ds_nomi])

        # Copy from cache into the project folder
        shutil.copytree(cache_path, dataset_path, dirs_exist_ok=True)
        print(f"{ds_nomi} dataset '{dataset_path}' ga muvaffaqiyatli yuklandi!")
        print(f"Papkalar: {os.listdir(dataset_path)}")

        return dataset_path


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download(ds_nomi="cracks")
