import argparse
from pathlib import Path
import yaml
from src.preprocess import process_folder, load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input folder (optional)")
    parser.add_argument("--output", help="Output folder (optional)")
    parser.add_argument("--json", help="JSON output folder (optional)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=None, help="Path to log file (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    in_dir = Path(args.input or cfg["io"]["input_dir"])
    out_dir = Path(args.output or cfg["io"]["output_dir"])
    json_dir = Path(args.json or cfg["io"]["json_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    process_folder(in_dir, out_dir, json_dir, config_path=args.config, log_path=args.log)

if __name__ == "__main__":
    main()