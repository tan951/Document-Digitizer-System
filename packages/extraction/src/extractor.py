import json
import logging
import time
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
from gliner import GLiNER
from jsonschema import validate
from symspellpy.symspellpy import SymSpell, Verbosity

from src.utils import (
    load_json,
    load_yaml,
    now_iso_utc,
    normalize_bhxh,
    normalize_date_dmy_to_iso,
    normalize_name_strip,
    normalize_number_generic,
    extract_value_from_same_line,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# ================= INITIALIZE MODELS (OFFLINE MODE) =================
print("🚀 Đang khởi tạo AI models...")

# Đường dẫn trỏ tới thư mục chứa model đã tải offline
MODEL_DIR = BASE_DIR / "local_models" / "gliner_multi-v2.1"

try:
    gliner_model = GLiNER.from_pretrained(
        str(MODEL_DIR), 
        local_files_only=True
    )
except Exception as e:
    print(f"⚠️ Cảnh báo: Không thể load model offline từ {MODEL_DIR}. Lỗi: {e}")
    print("Đang thử fallback tải từ HuggingFace (Cần có mạng)...")
    gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# nlp spacy.blank("vi") chạy 100% offline không cần tải weight ngoài
nlp = spacy.blank("vi") 

# ================= SYMSPELL =================
print("📚 Đang khởi tạo SymSpell...")

sym_spell = SymSpell(
    max_dictionary_edit_distance=2,
    prefix_length=7
)

DICT_PATH = BASE_DIR / "dictionary" / "Viet74K.txt"

if not DICT_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy dictionary: {DICT_PATH}")

with open(DICT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip().lower()
        if word:
            sym_spell.create_dictionary_entry(word, 1)

# Boost domain/legal vocabulary
DOMAIN_WORDS = {
    "quyết": 5000, "định": 5000, "bản": 4000, "án": 4000,
    "tòa": 4000, "bị": 4000, "cáo": 4000,
    "bhxh": 6000, "bảo": 3000, "hiểm": 3000, "xã": 3000, "hội": 3000,
    "ủy": 3000, "ban": 3000, "nhân": 3000, "dân": 3000,
    "nguyễn": 8000, "trần": 8000, "lê": 8000, "phạm": 8000,
    "hoàng": 8000, "thị": 7000, "văn": 7000, "tài": 5000, "sản": 5000,
}

for word, freq in DOMAIN_WORDS.items():
    sym_spell.create_dictionary_entry(word, freq)

print("✅ SymSpell khởi tạo thành công!")

# ================= LOGGER =================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("module3")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


# ================= HARD OCR CORRECTIONS =================
_HARD_CORRECTIONS: Dict[str, str] = {
    "TROM CAP": "TRỘM CẮP",
    "TR0M CAP": "TRỘM CẮP",
    "TR0M CÁP": "TRỘM CẮP",
    "TROMCAP": "TRỘM CẮP",
    "TRÇN THÞ": "TRẦN THỊ",
    "Mò Cây Nam": "Mỏ Cày Nam",
    "THÞ": "THỊ",
    "ÇN": "ẦN",
}


# ================= OCR CORRECTION =================
def apply_ocr_correction(text: str, confusions: Dict[str, str]) -> str:
    for wrong, correct in _HARD_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    for wrong, correct in confusions.items():
        text = text.replace(wrong, correct)
    return text


def apply_ocr_correction_to_value(
    value: str,
    confusions: Dict[str, str]
) -> str:
    for wrong, correct in _HARD_CORRECTIONS.items():
        value = value.replace(wrong, correct)
    for wrong, correct in confusions.items():
        value = value.replace(wrong, correct)
    return value


# ================= SYMSPELL CORRECTION =================
def symspell_word_correction(word: str) -> str:
    if not word.strip():
        return word

    # Giữ nguyên nếu hoàn toàn là số (để bảo vệ số quyết định, ngày tháng, v.v.)
    if re.fullmatch(r"\d+", word):
        return word

    # ĐÃ XÓA BLOCK "GIỮ NGUYÊN MÃ" Ở ĐÂY ĐỂ TRÁNH BỎ LỌT OCR HALLUCINATION (VD: "TÀ1")

    suggestions = sym_spell.lookup(
        word.lower(),
        Verbosity.CLOSEST,
        max_edit_distance=2
    )

    if not suggestions:
        return word

    best = suggestions[0].term

    # Preserve case formatting
    if word.isupper():
        return best.upper()
    if word.istitle():
        return best.title()

    return best


def apply_symspell_correction(text: str) -> str:
    if not text:
        return text

    tokens = re.findall(r"\w+|\S", text, re.UNICODE)
    corrected_tokens = []

    for token in tokens:
        if re.fullmatch(r"\w+", token, re.UNICODE):
            corrected_tokens.append(
                symspell_word_correction(token)
            )
        else:
            corrected_tokens.append(token)

    text = " ".join(corrected_tokens)

    # Fix punctuation spacing
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


# ================= CONFIDENCE =================
def compute_field_confidence(
    line_conf: float,
    raw_value: str,
    corrected_value: str,
    rule: Dict[str, Any],
    full_text_line: str,
) -> float:
    score = line_conf
    value_regexes = rule.get("value_regexes", [])

    strong_match = any(
        re.fullmatch(rx, raw_value.strip())
        for rx in value_regexes
    ) if value_regexes else True

    weak_match = any(
        re.search(rx, raw_value)
        for rx in value_regexes
    ) if value_regexes else False

    if strong_match:
        score += 0.12
    elif weak_match:
        score += 0.03
    else:
        score -= 0.08

    if raw_value != corrected_value:
        score -= 0.05

    label_regexes = rule.get("label_regexes", [])
    if any(
        re.search(rx, full_text_line, re.IGNORECASE)
        for rx in label_regexes
    ):
        score += 0.08

    return max(0.0, min(1.0, round(score, 3)))


# ================= SAFE VALUE EXTRACTION =================
def safe_extract_value(
    text: str,
    label_regexes: List[str],
    value_regexes: List[str] = None,
) -> Optional[str]:
    if not value_regexes:
        return extract_value_from_same_line(
            text,
            label_regexes,
            value_regexes
        )

    text_clean = text.strip()

    for v_regex in value_regexes:
        try:
            match = re.search(
                v_regex,
                text_clean,
                re.IGNORECASE
            )
            if match and len(match.groups()) > 0:
                return match.group(1).strip()
            elif match:
                return match.group(0).strip()
        except re.error:
            continue

    return extract_value_from_same_line(
        text,
        label_regexes,
        value_regexes
    )


# ================= ROUTER =================
def route_template(
    text_lines: List[Dict[str, Any]],
    templates: List[Dict[str, Any]]
) -> str:
    line_texts = [str(x.get("text", "")).upper() for x in text_lines]
    full_text = " ".join(line_texts)

    for tpl in templates:
        tpl_id = tpl.get("template_id")
        detect = tpl.get("document_detection", {})
        keywords = detect.get("keywords", [])
        regexes = detect.get("regexes", [])

        kw_score = sum(3 for t in line_texts for kw in keywords if kw.upper() in t)
        regex_score = sum(4 for t in line_texts for rx in regexes if re.search(rx, t, re.IGNORECASE))

        if tpl_id == "toa_an" and any(x in full_text for x in ["HS-ST", "BỊ CÁO", "XÉT XỬ"]):
            return "toa_an"

        if tpl_id == "bao_hiem" and any(x in full_text for x in ["BHXH", "QĐ-BHXH"]):
            return "bao_hiem"

        if kw_score + regex_score >= 3:
            return tpl_id

    return templates[0].get("template_id", "unknown")


# ================= NORMALIZER =================
def apply_normalizer(
    field_rule: Dict[str, Any],
    value: str,
    cfg: Dict[str, Any]
) -> str:
    normalizer = field_rule.get("normalizer")
    auto_corr = cfg.get("auto_correction", {})

    is_numeric_field = normalizer in [
        "date_dmy_to_iso",
        "number_slash",
        "bhxh_fix_confusions",
    ]

    confusions = (
        auto_corr.get("ocr_confusions", {})
        if (auto_corr.get("enabled") and is_numeric_field)
        else {}
    )

    value = apply_ocr_correction_to_value(value, confusions)

    if normalizer == "date_dmy_to_iso":
        return normalize_date_dmy_to_iso(value) or value.strip()
    if normalizer == "number_slash":
        return normalize_number_generic(value) or value.strip()
    if normalizer == "name_strip":
        return normalize_name_strip(value)
    if normalizer == "bhxh_fix_confusions":
        return normalize_bhxh(value, confusions) or value.strip()

    return value.strip()


# ================= DEDUP =================
def dedup_fixed_fields(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for f in fields:
        name = f["field_name"]
        if name not in best or f["confidence"] > best[name]["confidence"]:
            best[name] = f
    return list(best.values())


# ================= FIXED FIELD EXTRACTION =================
def extract_fixed_fields(
    text_lines: List[Dict[str, Any]],
    confusions: Dict[str, str] = None,
) -> List[Dict[str, Any]]:
    if confusions is None:
        confusions = {}

    results = []

    for line in text_lines:
        raw_text = line.get("text", "").strip()
        text_string = apply_ocr_correction(raw_text, confusions)
        bbox = line.get("bbox", [0, 0, 0, 0])
        line_conf = float(line.get("confidence", 0.0))

        # Số quyết định / Ký hiệu
        match = re.search(
            r"Số:\s*([\d]+(?:/[\d]+)?/[\w\-Đ]+)",
            text_string,
            re.IGNORECASE,
        )

        if match:
            raw_full = match.group(1)
            fixed_full = apply_ocr_correction_to_value(raw_full, confusions)
            parts = fixed_full.split("/")

            if len(parts) >= 2:
                if parts[1].isdigit():
                    so_qd = f"{parts[0]}/{parts[1]}"
                    ky_hieu_idx = 2
                else:
                    so_qd = parts[0]
                    ky_hieu_idx = 1
            else:
                so_qd = parts[0]
                ky_hieu_idx = -1

            results.append({
                "field_name": "so_quyet_dinh",
                "value": so_qd,
                "confidence": line_conf,
                "bbox": bbox,
                "method": "regex",
            })

            if ky_hieu_idx != -1 and len(parts) > ky_hieu_idx:
                results.append({
                    "field_name": "ky_hieu",
                    "value": "/".join(parts[ky_hieu_idx:]),
                    "confidence": line_conf,
                    "bbox": bbox,
                    "method": "regex",
                })

        # Ngày tháng năm
        date_match = re.search(
            r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
            text_string,
            re.IGNORECASE,
        )

        if date_match:
            d, m, y = date_match.groups()
            results.append({
                "field_name": "ngay_thang_nam",
                "value": f"{y}-{int(m):02d}-{int(d):02d}",
                "confidence": line_conf,
                "bbox": bbox,
                "method": "regex",
            })

    return dedup_fixed_fields(results)


# ================= GLINER =================
def map_gliner_to_config_fields(gliner_label: str) -> str:
    mapping = {
        "tên bị cáo": "ten_bi_cao",
        "tội danh": "toi_danh",
        "cơ quan ban hành": "ten_co_quan_to_chuc",
    }
    return mapping.get(gliner_label, gliner_label)


def extract_dynamic_fields(
    full_text: str,
    required_fields: List[str],
    existing_fields: List[str],
    text_lines: List[Dict],
    confusions: Dict[str, str] = None,
) -> List[Dict[str, Any]]:
    if confusions is None:
        confusions = {}

    results = []
    REGEX_HANDLED_FIELDS = {"so_quyet_dinh", "so_bhxh"}

    missing_fields = [
        f for f in required_fields
        if f not in existing_fields and f not in REGEX_HANDLED_FIELDS
    ]

    if not missing_fields:
        return results

    reverse_mapping = {
        "ten_bi_cao": "tên bị cáo",
        "toi_danh": "tội danh",
        "ten_co_quan_to_chuc": "cơ quan ban hành",
    }

    labels_to_predict = [reverse_mapping.get(f, f) for f in missing_fields]
    corrected_text = apply_ocr_correction(full_text, confusions)

    entities = gliner_model.predict_entities(
        corrected_text,
        labels_to_predict,
        threshold=0.35 # Giảm threshold để tăng độ nhạy, hạn chế bị miss field (đặc biệt cho khối BHXH)
    )

    grouped_entities = {}

    for ent in entities:
        field_name = map_gliner_to_config_fields(ent["label"])
        raw_value = ent["text"]
        
        # Sửa lỗi chính tả trước khi lưu trữ
        clean_value = apply_symspell_correction(raw_value)
        matched_bbox = [0, 0, 0, 0]

        for line in text_lines:
            if raw_value in line.get("text", ""):
                matched_bbox = line.get("bbox", [0, 0, 0, 0])
                break

        grouped_entities.setdefault(field_name, []).append({
            "field_name": field_name,
            "value": clean_value,
            "confidence": round(ent["score"], 3),
            "bbox": matched_bbox,
            "method": "gliner_ner",
        })

    for field_name, items in grouped_entities.items():
        best_match = max(items, key=lambda x: x["confidence"])
        results.append(best_match)

    return results


# ================= MAIN =================
def process_folder(
    input_dir: Path,
    output_dir: Path,
    config_path: str,
    schema_path: str,
    config_schema_path: str,
) -> None:
    logger = setup_logger()
    cfg = load_yaml(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    total_start_time = time.time()
    processed_count = 0

    TEXT_FIELDS_FOR_CORRECTION = [
        "ten_bi_cao",
        "toi_danh",
        "ten_co_quan_to_chuc",
        "ten_loai_van_ban",
    ]

    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() != ".json":
            continue

        file_start_time = time.time()

        try:
            in_json = load_json(str(p))
            doc_id = in_json.get("document_id", "unknown")
            req_id = str(uuid.uuid4())
            full_text = in_json.get("full_text", "")

            text_lines = []
            for page in in_json.get("pages", []):
                for box in page.get("bboxes", []):
                    text_lines.append({
                        "text": box.get("text", ""),
                        "bbox": box.get("bbox", [0, 0, 0, 0]),
                        "confidence": float(box.get("conf", 0.0)),
                    })

            templates = cfg.get("router", {}).get("templates", [])
            doc_type = route_template(text_lines, templates)
            matched_tpl = next((t for t in templates if t["template_id"] == doc_type), {})
            required_fields = matched_tpl.get("required_fields", [])

            auto_enabled = cfg.get("auto_correction", {}).get("enabled", False)
            confusions = cfg.get("auto_correction", {}).get("ocr_confusions", {}) if auto_enabled else {}
            changes_log = []

            # Lấy các trường tĩnh (Fixed) bằng Regex
            fixed_fields = extract_fixed_fields(text_lines, confusions)

            for field in fixed_fields:
                if auto_enabled and field["field_name"] in TEXT_FIELDS_FOR_CORRECTION:
                    before = field["value"]
                    after = apply_symspell_correction(before)
                    if before != after:
                        changes_log.append({
                            "original": before,
                            "corrected": after,
                            "field": f"{field['field_name']} (SymSpell)",
                        })
                    field["value"] = after

            existing_field_names = [f["field_name"] for f in fixed_fields]

            # Lấy các trường động (Dynamic) bằng GLiNER
            dynamic_fields = extract_dynamic_fields(
                full_text[:2000],
                required_fields,
                existing_field_names,
                text_lines,
                confusions,
            )

            dynamic_fields_final = []
            for field in dynamic_fields:
                rule = cfg.get("fields", {}).get(field["field_name"], {})
                raw_val = field["value"]
                corrected_val = apply_normalizer(rule, raw_val, cfg)
                field["value"] = corrected_val
                dynamic_fields_final.append(field)

            out_json = {
                "request_id": req_id,
                "document_id": doc_id,
                "timestamp": now_iso_utc(),
                "status": "success",
                "error": None,
                "payload": {
                    "document_type": doc_type,
                    "extracted_fields": {
                        "fixed": fixed_fields,
                        "dynamic": dynamic_fields_final,
                    },
                    "correction_log": {
                        "enabled": auto_enabled,
                        "changes": changes_log,
                    },
                },
            }

            validate(instance=out_json, schema=out_schema)

            out_path = output_dir / f"{p.stem}_extracted.json"
            out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

            file_elapsed_time = time.time() - file_start_time
            processed_count += 1
            logger.info(f"✅ {p.name} -> {doc_type} | Fields: {len(fixed_fields) + len(dynamic_fields_final)} | Time: {file_elapsed_time:.2f}s")

        except Exception as e:
            file_elapsed_time = time.time() - file_start_time
            logger.error(f"❌ Lỗi {p.name}: {e} | Time: {file_elapsed_time:.2f}s")

    total_elapsed_time = time.time() - total_start_time
    logger.info(f"🎉 Hoàn thành xử lý {processed_count} files. Tổng thời gian: {total_elapsed_time:.2f}s")