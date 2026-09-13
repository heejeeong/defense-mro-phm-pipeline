"""Prepare a small YOLO sample from the AI Hub validation ZIP files."""

import logging
import os
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "51.군 경계 작전 환경 내 인식 데이터"
    / "3.개방데이터"
    / "1.데이터"
)
IMAGE_ZIP = DATASET_ROOT / "Validation" / "01.원천데이터" / "VS_I1_S0_C5.zip"
LABEL_ZIP = DATASET_ROOT / "Validation" / "02.라벨링데이터" / "VL_I1_S0_C5.zip"

TEMP_IMAGE_DIR = PROJECT_ROOT / "data" / "temp_val_img"
TEMP_LABEL_DIR = PROJECT_ROOT / "data" / "temp_val_lbl"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "data" / "sample" / "images"
SAMPLE_LABEL_DIR = PROJECT_ROOT / "data" / "sample" / "labels"

MAX_PAIRS = 500
IMAGE_EXTENSIONS = {".jpg", ".png"}
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable terminal log messages."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract an archive without allowing entries outside destination."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = member.filename.replace("\\", "/").lstrip("/")
            target = (destination / member_name).resolve()
            if os.path.commonpath((str(destination_root), str(target))) != str(destination_root):
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as target_file:
                shutil.copyfileobj(source, target_file)


def find_files(directory: Path, suffixes: set[str]) -> list[Path]:
    """Find files recursively with one of the requested suffixes."""
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )


def build_file_index(files: list[Path]) -> dict[str, Path]:
    """Index files by lowercase stem, keeping the first deterministic match."""
    index: dict[str, Path] = {}
    for path in files:
        index.setdefault(path.stem.lower(), path)
    return index


def select_matching_pairs() -> list[tuple[Path, Path]]:
    """Return up to MAX_PAIRS image/JSON pairs with matching filenames."""
    image_index = build_file_index(find_files(TEMP_IMAGE_DIR, IMAGE_EXTENSIONS))
    label_index = build_file_index(find_files(TEMP_LABEL_DIR, {".json"}))
    matching_stems = sorted(image_index.keys() & label_index.keys())
    pairs = [(image_index[stem], label_index[stem]) for stem in matching_stems[:MAX_PAIRS]]
    logger.info("Found %d matching image/label pairs; selecting %d.", len(matching_stems), len(pairs))
    return pairs


def copy_pairs(pairs: list[tuple[Path, Path]]) -> None:
    """Copy selected images and labels into the sample directories."""
    SAMPLE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        shutil.copy2(image_path, SAMPLE_IMAGE_DIR / image_path.name)
        shutil.copy2(label_path, SAMPLE_LABEL_DIR / label_path.name)
    logger.info("Copied %d images to %s", len(pairs), SAMPLE_IMAGE_DIR)
    logger.info("Copied %d labels to %s", len(pairs), SAMPLE_LABEL_DIR)


def print_training_cleanup_instructions() -> None:
    """Print manual cleanup instructions without deleting training data."""
    training_zips = sorted(DATASET_ROOT.rglob("TS_*.zip"))
    logger.info("Temporary validation folders were removed.")
    print("\nValidation sample preparation is complete.")
    if training_zips:
        print("Large Training ZIP files remain. Delete them manually after verification:")
        for archive in training_zips:
            print(f'  rm "{archive}"')
    else:
        print("No TS_*.zip Training files were found under the dataset directory.")


def prepare_dataset() -> None:
    """Extract validation data, copy matching pairs, and clean temporary files."""
    for archive_path in (IMAGE_ZIP, LABEL_ZIP):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Required ZIP file not found: {archive_path}")

    for temporary_dir in (TEMP_IMAGE_DIR, TEMP_LABEL_DIR):
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    try:
        logger.info("Extracting validation images: %s", IMAGE_ZIP.name)
        safe_extract(IMAGE_ZIP, TEMP_IMAGE_DIR)
        logger.info("Extracting validation labels: %s", LABEL_ZIP.name)
        safe_extract(LABEL_ZIP, TEMP_LABEL_DIR)

        pairs = select_matching_pairs()
        if not pairs:
            raise RuntimeError("No matching image/JSON pairs were found in the validation ZIP files.")
        copy_pairs(pairs)
    finally:
        for temporary_dir in (TEMP_IMAGE_DIR, TEMP_LABEL_DIR):
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

    print_training_cleanup_instructions()


if __name__ == "__main__":
    configure_logging()
    prepare_dataset()