import sys
import pytest
from pathlib import Path
from src.utils import load_yaml, load_json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def config():
    """Load cấu hình từ file config.yaml hiện tại"""
    return load_yaml(str(ROOT / "config.yaml"))

@pytest.fixture
def extraction_schema():
    """Load schema đầu ra để validate kết quả"""
    return load_json(str(ROOT / "schemas" / "extraction.schema.json"))

@pytest.fixture
def mock_input():
    """Cung cấp dữ liệu mẫu giả lập theo cấu trúc JSON mới[cite: 6]"""
    return {
        "document_id": "DOC-ABC",
        "full_text": "Số: 102/2026/QĐ-HS\nTòa án nhân dân quận Ba, ngày 04/05/2026\nBị cáo Nguyễn Văn A bị truy tố về tội trộm cắp tài sản.",
        "pages": [
            {
                "page_num": 1,
                "text": "Số: 102/2026/QĐ-HS\nTòa án nhân dân quận Ba, ngày 04/05/2026\nBị cáo Nguyễn Văn A bị truy tố về tội trộm cắp tài sản.",
                "bboxes": [
                    {"text": "Số: 102/2026/QĐ-HS", "bbox": [10, 10, 50, 20], "conf": 0.99},
                    {"text": "Tòa án nhân dân quận Ba, ngày 04/05/2026", "bbox": [10, 40, 100, 20], "conf": 0.98},
                    {"text": "Bị cáo Nguyễn Văn A bị truy tố về tội trộm cắp tài sản.", "bbox": [10, 80, 200, 20], "conf": 0.95}
                ],
                "avg_conf": 0.97
            }
        ]
    }