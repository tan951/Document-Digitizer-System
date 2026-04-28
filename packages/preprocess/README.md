# Module 1 - Image Pre-processing (Local)

## Cấu trúc
```
module1/
├── data/
│   ├── input/          # ảnh gốc
│   └── output/         # ảnh đã xử lý
├── src/
│   ├── preprocess.py   # pipeline chính
│   ├── utils.py        # helper
│   └── config.py       # tham số
├── tests/
│   └── test_basic.py
├── run.py
├── requirements.txt
└── README.md
```

## 1. Run
```bash
pip install -r requirements.txt
python run.py --config ./config.yaml
```

Optional:
```bash
python run.py --config ./config.yaml --log ./logs/run.log
```

## 2. Config
All params are defined in `config.yaml`.  
Config is validated against `schemas/config.schema.json`.

## 3. Output JSON
Each input image produces a JSON file in `json_dir`.

**Output schema:** `schemas/preprocess.schema.json`

**Important fields:**
- `request_id`: UUIDD
- `document_id`: per batch index
- `status`: success | error
- `payload.page`: page index
- `payload.rotation`: OSD rotation
- `payload.is_blank`: blank detection
- `payload.blank_ratio`: optional debug
- `payload.crop_applied`, `payload.deskew_angle`, `payload.osd`

## 4. Testing
```bash
pytest -q
```
Tests auto-generate sample assets into `tests/assets`.

## 5. Config Reference

### io
- `input_dir` (string): folder chứa ảnh input  
- `output_dir` (string): folder ảnh đã xử lý  
- `json_dir` (string): folder JSON output  

### resize
- `target_width` (int): chiều rộng chuẩn sau resize  

### denoise
- `method` (string): `fastNlMeans` / `median`  
- `h` (int): strength cho fastNlMeans  

### crop
- `enable` (bool): bật/tắt crop  
- `min_area_ratio` (float): tỷ lệ tối thiểu vùng crop  
- `min_w_ratio`, `min_h_ratio` (float): tỷ lệ tối thiểu theo w/h  
- `padding` (int): padding thêm  

### deskew
- `enable` (bool): bật/tắt deskew  

### normalize
- `method` (string): `clahe` / `hist`  
- `clip_limit` (float): CLAHE  
- `tile_grid` (list[int,int]): CLAHE grid size  

### binarize
- `method` (string): `adaptive` / `otsu`  
- `block_size` (int): block size cho adaptive  
- `C` (int): hằng số offset  

### morphology
- `kernel` (list[int,int]): kernel size  

### blank_detect
- `threshold` (float): tỉ lệ pixel đen để coi là blank  

### osd
- `enable` (bool): bật/tắt rotation detect  
- `min_width`, `min_height` (int): tối thiểu để OSD chạy  
- `min_black_ratio` (float): threshold để skip OSD  