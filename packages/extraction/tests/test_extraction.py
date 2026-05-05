import pytest
import uuid
from jsonschema import validate
from src.extractor import route_template, extract_basic_metadata, apply_normalizer

def get_flattened_text_lines(mock_input_data):
    """Hàm hỗ trợ trích xuất text_lines từ cấu trúc pages mới cho các bài test[cite: 7]"""
    lines = []
    for page in mock_input_data.get("pages", []):
        for box in page.get("bboxes", []):
            lines.append({
                "text": box.get("text", ""),
                "bbox": box.get("bbox", [0, 0, 0, 0]),
                "confidence": float(box.get("conf", 0.0))
            })
    return lines

def test_route_template(mock_input, config):
    """Kiểm tra khả năng định dạng loại văn bản (Router)"""
    text_lines = get_flattened_text_lines(mock_input) # Sử dụng cấu trúc đã convert[cite: 7]
    templates = config.get("router", {}).get("templates", [])
    
    doc_type = route_template(text_lines, templates)
    assert doc_type == "toa_an"  

def test_extract_basic_metadata(mock_input):
    """Kiểm tra bóc tách các trường biên mục cố định (Fixed Fields)"""
    text_lines = get_flattened_text_lines(mock_input) # Sử dụng cấu trúc đã convert[cite: 7]
    fixed_fields = extract_basic_metadata(text_lines)
    
    field_map = {f["field_name"]: f["value"] for f in fixed_fields}
    
    assert "so_van_ban" in field_map
    assert field_map["so_van_ban"] == "102/2026"
    assert "ten_loai_van_ban" in field_map
    assert field_map["ten_loai_van_ban"] == "Quyết định"
    assert field_map["ngay_thang_nam"] == "04/05/2026"

def test_apply_normalizer_date(config):
    """Kiểm tra chức năng chuẩn hóa ngày tháng"""
    field_rule = {"normalizer": "date_dmy_to_iso"}
    
    result = apply_normalizer(field_rule, "04/05/2026", config)
    assert result == "2026-05-04"
    
    result = apply_normalizer(field_rule, "04.05.2026", config)
    assert result == "2026-05-04"

def test_output_structure_validation(mock_input, config, extraction_schema):
    """Kiểm tra cấu trúc đóng gói dữ liệu cuối cùng có khớp Schema không"""
    from src.extractor import now_iso_utc
    
    text_lines = get_flattened_text_lines(mock_input)
    fixed = extract_basic_metadata(text_lines)
    
    output = {
        "request_id": str(uuid.uuid4()), # Cập nhật do mock_input mới không có[cite: 7]
        "document_id": mock_input.get("document_id", "unknown"),
        "timestamp": now_iso_utc(),
        "status": "success",
        "error": None,
        "payload": {
            "document_type": "toa_an",
            "extracted_fields": {
                "fixed": fixed,
                "dynamic": [] 
            },
            "correction_log": {
                "enabled": True,
                "changes": []
            }
        }
    }
    
    # Validation không báo lỗi là đạt
    validate(instance=output, schema=extraction_schema)

def test_normalization_confusions(config):
    """Kiểm tra sửa lỗi OCR (quẩn -> quận)"""
    field_rule = {"normalizer": "name_strip"}
    raw_text = "BHXH quẩn Ba"
    
    result = apply_normalizer(field_rule, raw_text, config)
    
    if "quẩn" in config.get("auto_correction", {}).get("ocr_confusions", {}):
        assert "quận" in result