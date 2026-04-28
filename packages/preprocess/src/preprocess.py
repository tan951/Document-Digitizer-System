SCHEMA_VERSION = "1.0.0"

from pathlib import Path
import json, uuid
from datetime import datetime
import cv2
import numpy as np
import pytesseract
import yaml
from jsonschema import validate, ValidationError
import logging
import time

from .utils import resize_to_width, four_point_transform, count_black_ratio

from importlib.metadata import version as pkg_version

REQUIRED_KEYS = [
    "io", "resize", "denoise", "crop", "deskew",
    "normalize", "binarize", "morphology", "blank_detect", "osd"
]

DEFAULT_CONFIG = {
    "io": {"input_dir": "./data/input", "output_dir": "./data/output", "json_dir": "./data/json"},
    "resize": {"target_width": 2000},
    "denoise": {"method": "fastNlMeans", "h": 10},
    "crop": {"enable": True, "min_area_ratio": 0.20, "min_w_ratio": 0.40, "min_h_ratio": 0.40, "padding": 20},
    "deskew": {"enable": True},
    "normalize": {"method": "clahe", "clip_limit": 2.0, "tile_grid": [8, 8]},
    "binarize": {"method": "adaptive", "block_size": 31, "C": 15},
    "morphology": {"kernel": [3, 3]},
    "blank_detect": {"threshold": 0.005},
    "osd": {"enable": True, "min_width": 600, "min_height": 200, "min_black_ratio": 0.01}
}

def deep_merge(defaults, user_cfg):
    for k, v in user_cfg.items():
        if isinstance(v, dict) and k in defaults:
            defaults[k] = deep_merge(defaults[k], v)
        else:
            defaults[k] = v
    return defaults

def get_version():
    try:
        return pkg_version("module1-preprocess")
    except Exception:
        return "1.0.0"

def validate_config(cfg):
    for k in REQUIRED_KEYS:
        if k not in cfg:
            raise ValueError(f"Missing config section: '{k}'")

    if "target_width" not in cfg["resize"]:
        raise ValueError("Missing resize.target_width in config.yaml")
    if "threshold" not in cfg["blank_detect"]:
        raise ValueError("Missing blank_detect.threshold in config.yaml")
    if "min_width" not in cfg["osd"] or "min_height" not in cfg["osd"]:
        raise ValueError("Missing osd.min_width/min_height in config.yaml")

def load_config(config_path="config.yaml", schema_path="schemas/config.schema.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        validate(instance=cfg, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Config schema invalid: {e.message}")

    return cfg

def validate_output(result, schema_path="schemas/preprocess.schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        validate(instance=result, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Output JSON invalid: {e.message}")

def setup_logger(log_path=None):
    logger = logging.getLogger("module1")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file (optional)
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def detect_rotation_osd(gray):
    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0)), True, ""
    except Exception as e:
        return 0, False, str(e)

def preprocess_image(img, cfg):
    # --- Resize ---
    target_width = cfg["resize"]["target_width"]
    img = resize_to_width(img, target_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Denoise ---
    if cfg["denoise"]["method"] == "fastNlMeans":
        h = cfg["denoise"]["h"]
        denoise = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
    else:
        denoise = gray

    # --- Auto-crop ---
    crop_applied = False
    if cfg["crop"]["enable"]:
        edges = cv2.Canny(denoise, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w = img.shape[:2]
        img_area = w * h
        min_area = cfg["crop"]["min_area_ratio"] * img_area
        min_w = cfg["crop"]["min_w_ratio"] * w
        min_h = cfg["crop"]["min_h_ratio"] * h

        if contours:
            c = contours[0]
            area = cv2.contourArea(c)
            x, y, cw, ch = cv2.boundingRect(c)

            if area >= min_area and cw >= min_w and ch >= min_h:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = box.astype("int32")
                img = four_point_transform(img, box)

                pad = cfg["crop"]["padding"]
                img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
                crop_applied = True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Deskew ---
    deskew_angle = 0.0
    if cfg["deskew"]["enable"]:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh < 255))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            deskew_angle = float(angle)
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Normalize ---
    if cfg["normalize"]["method"] == "clahe":
        clahe = cv2.createCLAHE(clipLimit=cfg["normalize"]["clip_limit"],
                                tileGridSize=tuple(cfg["normalize"]["tile_grid"]))
        gray = clahe.apply(gray)

    # --- Binarize ---
    if cfg["binarize"]["method"] == "adaptive":
        block = cfg["binarize"]["block_size"]
        C = cfg["binarize"]["C"]
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block, C)
    else:
        bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # --- Morphology ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(cfg["morphology"]["kernel"]))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # --- Blank detect ---
    black_ratio = count_black_ratio(bin_img)
    is_blank = black_ratio < cfg["blank_detect"]["threshold"]

    # --- OSD rotation ---
    osd_cfg = cfg["osd"]
    osd_attempted = False
    osd_success = False
    osd_message = ""
    rotation = 0

    if osd_cfg["enable"]:
        h, w = gray.shape[:2]
        if h >= osd_cfg["min_height"] and w >= osd_cfg["min_width"] and black_ratio > osd_cfg["min_black_ratio"]:
            osd_attempted = True
            rotation, osd_success, osd_message = detect_rotation_osd(gray)

    if rotation in (90, 180, 270):
        img = cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE,
                               180: cv2.ROTATE_180,
                               270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])

    return img, bin_img, {
        "is_blank": is_blank,
        "black_ratio": float(black_ratio),
        "rotation": rotation,
        "crop_applied": crop_applied,
        "deskew_angle": deskew_angle,
        "osd": {
            "attempted": osd_attempted,
            "success": osd_success,
            "message": osd_message
        }
    }

def process_folder(input_dir: Path, output_dir: Path, json_dir: Path, config_path="config.yaml", log_path=None):
    cfg = load_config(config_path)
    logger = setup_logger(log_path)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for idx, p in enumerate(sorted(input_dir.iterdir()), start=1):
        if p.suffix.lower() not in exts:
            continue

        start = time.time()

        try:
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError("cv2.imread returned None (file may be corrupt or unreadable)")

            clean_img, bin_img, meta = preprocess_image(img, cfg)

            out_clean = output_dir / f"{p.stem}_clean.png"
            out_bin = output_dir / f"{p.stem}_bin.png"
            cv2.imwrite(str(out_clean), clean_img)
            cv2.imwrite(str(out_bin), bin_img)

            result = {
                "request_id": str(uuid.uuid4()),
                "document_id": f"local_doc_{idx}",
                "module": "preprocess",
                "version": get_version(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "success",
                "error": None,
                "payload": {
                    "page": idx,
                    "input_image": str(p),
                    "output_image": str(out_clean),
                    "width": int(clean_img.shape[1]),
                    "height": int(clean_img.shape[0]),
                    "rotation": meta["rotation"],
                    "is_blank": bool(meta["is_blank"]),
                    "blank_ratio": meta["black_ratio"],
                    "crop_applied": meta["crop_applied"],
                    "deskew_angle": meta["deskew_angle"],
                    "osd": meta["osd"]
                }
            }

            validate_output(result)
            status = "SUCCESS"

        except Exception as e:
            result = {
                "request_id": str(uuid.uuid4()),
                "document_id": f"local_doc_{idx}",
                "module": "preprocess",
                "version": get_version(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "error",
                "error": {
                    "code": "PREPROCESS_ERROR",
                    "message": str(e)
                },
                "payload": {
                    "page": idx,
                    "input_image": str(p),
                    "output_image": "",
                    "width": 0,
                    "height": 0,
                    "rotation": 0,
                    "is_blank": False
                }
            }
            status = "ERROR"

        json_path = json_dir / f"{p.stem}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        elapsed = time.time() - start
        if status == "SUCCESS":
            logger.info(f"{p.name} | time={elapsed:.3f}s | blank={result['payload']['is_blank']} | rot={result['payload']['rotation']}")
        else:
            logger.error(f"{p.name} | time={elapsed:.3f}s | error={result['error']['message']}")