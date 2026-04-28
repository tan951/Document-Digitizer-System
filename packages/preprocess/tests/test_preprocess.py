# tests/test_preprocess.py
import cv2
import numpy as np
from pathlib import Path
from src.preprocess import preprocess_image, load_config

ASSETS = Path("tests/assets")

def _ensure_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)

    blank_path = ASSETS / "blank.png"
    text_path = ASSETS / "text_line.png"
    rot_path = ASSETS / "rotated_90.png"

    if not blank_path.exists():
        blank = np.full((800, 600, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(blank_path), blank)

    if not text_path.exists():
        text = np.full((800, 600, 3), 255, dtype=np.uint8)
        cv2.putText(text, "Hello OCR", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imwrite(str(text_path), text)

    if not rot_path.exists():
        text = np.full((800, 600, 3), 255, dtype=np.uint8)
        cv2.putText(text, "Hello OCR", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        rot = cv2.rotate(text, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(rot_path), rot)

def test_blank_image():
    _ensure_assets()
    cfg = load_config("config.yaml")
    img = cv2.imread(str(ASSETS / "blank.png"))
    clean, bin_img, meta = preprocess_image(img, cfg)
    assert meta["is_blank"] is True

def test_text_line_image():
    _ensure_assets()
    cfg = load_config("config.yaml")
    img = cv2.imread(str(ASSETS / "text_line.png"))
    clean, bin_img, meta = preprocess_image(img, cfg)
    assert meta["is_blank"] is False

def test_rotated_image():
    _ensure_assets()
    cfg = load_config("config.yaml")
    img = cv2.imread(str(ASSETS / "rotated_90.png"))
    clean, bin_img, meta = preprocess_image(img, cfg)
    assert clean is not None and bin_img is not None