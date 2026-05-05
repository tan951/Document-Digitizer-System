import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import validate, ValidationError

from src.utils import (
    load_json, load_yaml, now_iso_utc,
    normalize_bhxh, normalize_date_dmy_to_iso,
    normalize_name_strip, normalize_number_generic,
    find_label_match, extract_value_from_same_line
)

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("module3")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger

def route_template(text_lines: List[Dict[str, Any]], templates: List[Dict[str, Any]]) -> str:
    best_template = templates[0].get("template_id", "unknown") if templates else "unknown"
    best_score = -1
    line_texts = [str(x.get("text", "")) for x in text_lines]

    for tpl in templates:
        tpl_id = tpl.get("template_id", "unknown")
        detect = tpl.get("document_detection", {})
        keywords, regexes = detect.get("keywords", []), detect.get("regexes", [])
        
        score = sum(2 for t in line_texts for kw in keywords if str(kw).lower() in t.lower())
        score += sum(3 for t in line_texts for rx in regexes if re.search(rx, t, flags=re.IGNORECASE))

        if score > best_score:
            best_score, best_template = score, tpl_id

    return best_template

def extract_basic_metadata(text_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bóc tách các trường biên mục cố định (Fixed Fields)"""
    results = []
    doc_type_mapping = {"QĐ": "Quyết định", "NQ": "Nghị quyết", "CT": "Chỉ thị"}

    for line in text_lines:
        text = line.get("text", "")
        bbox = line.get("bbox", [0, 0, 0, 0])
        conf = float(line.get("confidence", 0.0))

        # 1. Bóc tách Số văn bản & Ký hiệu
        if "Số:" in text:
            match = re.search(r"Số:\s*([\d]+(?:/\d{4})?)/([A-ZĐa-zđ-]+)", text)
            if match:
                results.append({"field_name": "so_van_ban", "value": match.group(1), "confidence": conf, "bbox": bbox, "method": "regex"})
                results.append({"field_name": "ky_hieu", "value": match.group(2), "confidence": conf, "bbox": bbox, "method": "regex"})
                
                prefix = match.group(2).split('-')[0].upper()
                results.append({"field_name": "ten_loai_van_ban", "value": doc_type_mapping.get(prefix, "Không xác định"), "confidence": conf, "bbox": bbox, "method": "rule_mapping"})

        # 2. Bóc tách Tên cơ quan & Ngày tháng năm
        if "ngày" in text.lower() and "," in text:
            parts = text.split(",", 1)
            date_match = re.search(r"ngày\s+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", parts[1].lower())
            
            results.append({"field_name": "ten_co_quan_to_chuc", "value": parts[0].strip(), "confidence": conf, "bbox": bbox, "method": "string_split"})
            if date_match:
                results.append({"field_name": "ngay_thang_nam", "value": date_match.group(1), "confidence": conf, "bbox": bbox, "method": "regex"})

    return results

def apply_normalizer(field_rule: Dict[str, Any], value: str, cfg: Dict[str, Any]) -> str:
    normalizer = field_rule.get("normalizer")
    if not normalizer: return value.strip()

    auto_corr = cfg.get("auto_correction", {})
    confusions = auto_corr.get("ocr_confusions", {}) if auto_corr.get("enabled") else {}

    if normalizer == "date_dmy_to_iso": return normalize_date_dmy_to_iso(value) or value.strip()
    if normalizer == "number_slash": return normalize_number_generic(value) or value.strip()
    if normalizer == "name_strip": return normalize_name_strip(value)
    if normalizer == "bhxh_fix_confusions": return normalize_bhxh(value, confusions) or value.strip()
    return value.strip()

def process_folder(input_dir: Path, output_dir: Path, config_path: str, schema_path: str, config_schema_path: str):
    logger = setup_logger()
    cfg = load_yaml(config_path)

    # Validate Config Schema
    try:
        validate(instance=cfg, schema=json.loads(Path(config_schema_path).read_text(encoding="utf-8")))
    except ValidationError as e:
        logger.error(f"Config lỗi cấu trúc: {e.message}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() != ".json": continue
        try:
            in_json = load_json(str(p))
            
            # --- CẬP NHẬT: Xử lý cấu trúc input mới ---
            doc_id = in_json.get("document_id", "unknown")
            req_id = str(uuid.uuid4()) # Tự sinh request_id vì input mới không cung cấp[cite: 1]
            
            # Phẳng hóa các bboxes từ các pages
            text_lines = []
            for page in in_json.get("pages", []):
                for box in page.get("bboxes", []):
                    text_lines.append({
                        "text": box.get("text", ""),
                        "bbox": box.get("bbox", [0, 0, 0, 0]),
                        "confidence": float(box.get("conf", 0.0)) # Ánh xạ 'conf' thành 'confidence'[cite: 1]
                    })
            # ------------------------------------------

            doc_type = route_template(text_lines, cfg.get("router", {}).get("templates", []))
            
            # --- 1. Fixed Fields ---
            fixed_fields = extract_basic_metadata(text_lines)

            # --- 2. Dynamic Fields & Correction Log ---
            dynamic_fields = []
            changes_log = []
            auto_enabled = cfg.get("auto_correction", {}).get("enabled", False)

            matched_tpl = next((t for t in cfg.get("router", {}).get("templates", []) if t["template_id"] == doc_type), {})
            
            for field_name in matched_tpl.get("required_fields", []):
                # Bỏ qua nếu đã được bóc tách ở fixed_fields
                if any(f["field_name"] == field_name for f in fixed_fields): continue
                
                rule = cfg.get("fields", {}).get(field_name, {})
                if not rule: continue

                for line in text_lines:
                    t = str(line.get("text", ""))
                    if not find_label_match(t, rule.get("label_regexes", [])): continue

                    raw_val = extract_value_from_same_line(t, rule.get("label_regexes", []), rule.get("value_regexes"))
                    if raw_val:
                        corrected_val = apply_normalizer(rule, raw_val, cfg)
                        
                        # Theo dõi lịch sử sửa lỗi
                        if auto_enabled and raw_val != corrected_val:
                            changes_log.append({"original": raw_val, "corrected": corrected_val, "field": field_name})

                        dynamic_fields.append({
                            "field_name": field_name,
                            "value": corrected_val,
                            "confidence": float(line.get("confidence", 0.0)),
                            "bbox": line.get("bbox", [0, 0, 0, 0]),
                            "method": "config_regex"
                        })
                        break # Tìm thấy thì dừng cho trường này

            # --- 3. Đóng gói Output ---
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
                        "dynamic": dynamic_fields
                    },
                    "correction_log": {
                        "enabled": auto_enabled,
                        "changes": changes_log
                    }
                }
            }

            # Validate với Output Schema
            validate(instance=out_json, schema=out_schema)
            
            out_path = output_dir / f"{p.stem}_extracted.json"
            out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"✅ Thành công: {p.name} -> {doc_type}")

        except Exception as e:
            logger.error(f"❌ Lỗi xử lý {p.name}: {e}")