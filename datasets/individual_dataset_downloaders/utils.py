import logging
import urllib.request
from pathlib import Path


logger = logging.getLogger("dataset_downloaders")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, out_path: Path, progress_label: str) -> None:
    ensure_dir(out_path.parent)
    logger.info("Downloading %s from %s", progress_label, url)
    urllib.request.urlretrieve(url, out_path)
    logger.info("Saved %s to %s", progress_label, out_path)


def download_gdrive(file_id: str, out_path: Path, progress_label: str) -> None:
    import gdown

    ensure_dir(out_path.parent)
    logger.info("Downloading %s from Google Drive", progress_label)
    result = gdown.download(id=file_id, output=str(out_path), quiet=False)
    if result is None:
        raise RuntimeError(f"Google Drive download failed for {progress_label}")
    logger.info("Saved %s to %s", progress_label, out_path)
