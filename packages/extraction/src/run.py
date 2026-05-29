import argparse
from pathlib import Path

# Import process_folder từ thư mục src (sẽ đi qua __init__.py)
from src import process_folder

def main():
    parser = argparse.ArgumentParser(description="Module 3: Dynamic Extraction")
    parser.add_argument("--input", default="./data/input", help="Thư mục chứa file input.json")
    parser.add_argument("--output", default="./data/output", help="Thư mục lưu kết quả đầu ra")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn tới file config.yaml")
    parser.add_argument("--schema", default="./schemas/extraction.schema.json", help="Schema đầu ra")
    parser.add_argument("--config-schema", default="./schemas/config.schema.json", help="Schema cấu hình")
    args = parser.parse_args()

    process_folder(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        config_path=args.config,
        schema_path=args.schema,
        config_schema_path=args.config_schema,
        max_workers=1
    )

if __name__ == "__main__":
    main()