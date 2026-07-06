"""Copy a few test image/mask pairs into demo_samples/ so the Streamlit demo has
committed samples on Streamlit Cloud (where the full datasetlar/ is not deployed)."""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMG = PROJECT_ROOT / "datasetlar" / "cracks" / "test_img"
TEST_LAB = PROJECT_ROOT / "datasetlar" / "cracks" / "test_lab"

DEMO_DIR  = PROJECT_ROOT / "demo_samples"
DEMO_IMG  = DEMO_DIR / "images"
DEMO_LAB  = DEMO_DIR / "masks"

N_SAMPLES = 6


def main():
    if not TEST_IMG.exists():
        raise FileNotFoundError(
            f"{TEST_IMG} not found. Run `python downloader.py` first to fetch the dataset."
        )

    DEMO_IMG.mkdir(parents=True, exist_ok=True)
    DEMO_LAB.mkdir(parents=True, exist_ok=True)

    images = sorted(TEST_IMG.glob("*.*"))[:N_SAMPLES]
    copied = 0
    for im_path in images:
        gt_path = TEST_LAB / f"{im_path.stem}.png"   # mask shares the image's stem
        if not gt_path.exists():
            continue
        shutil.copy(im_path, DEMO_IMG / im_path.name)
        shutil.copy(gt_path, DEMO_LAB / gt_path.name)
        copied += 1

    print(f"Copied {copied} image/mask pairs -> {DEMO_DIR}")


if __name__ == "__main__":
    main()
