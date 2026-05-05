import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dateutil import parser as date_parser

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def now_iso_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"

def ocr_confusions_replace(text: str, confusions: Dict[str, str]) -> str:
    out = text
    for k, v in confusions.items():
        out = out.replace(k, v)
    return out

def normalize_date_dmy_to_iso(text: str) -> Optional[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"[\.\/]", "-", cleaned)
    try:
        dt = date_parser.parse(cleaned, dayfirst=True, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return None

def normalize_number_generic(text: str) -> str:
    return re.sub(r"[^0-9]", "", text.strip())

def normalize_bhxh(text: str, confusions: Dict[str, str]) -> Optional[str]:
    fixed = ocr_confusions_replace(text, confusions)
    digits = re.findall(r"[0-9]", fixed)
    return "".join(digits) if digits else None

def normalize_name_strip(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    return t.strip(":-").strip()

def find_label_match(line_text: str, label_regexes: List[str]) -> Optional[re.Match]:
    for rx in label_regexes:
        m = re.search(rx, line_text, flags=re.IGNORECASE)
        if m: return m
    return None

def extract_value_from_same_line(line_text: str, label_regexes: List[str], value_regexes: Optional[List[str]] = None) -> Optional[str]:
    if value_regexes:
        for vr in value_regexes:
            mv = re.search(vr, line_text, flags=re.IGNORECASE)
            if mv: return mv.group(0).strip()

    m = find_label_match(line_text, label_regexes)
    if not m: return None
    
    remainder = line_text[m.end():]
    remainder = re.sub(r"^\s*[:\-–—]\s*", "", remainder).strip()
    return remainder if remainder else None