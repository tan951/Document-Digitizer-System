import os
import json
import logging
import time
import re
import uuid
import hashlib
import threading
import concurrent.futures
import warnings
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from collections import OrderedDict
from difflib import SequenceMatcher

import spacy
import torch
from gliner import GLiNER
from jsonschema import validate
from symspellpy.symspellpy import SymSpell, Verbosity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as hf_logging

torch.set_num_threads(1)
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message="Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length.*",
)

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

# ==========================================
# [Tối ưu 4]: Cache LRU an toàn (Thread-safe)
# ==========================================
class SafeLRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key, default=None):
        with self.lock:
            if key not in self.cache:
                return default
            self.cache.move_to_end(key)
            return self.cache[key]

    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def __contains__(self, key):
        with self.lock:
            return key in self.cache

_OCR_CACHE = SafeLRUCache(10000)
_ROUTE_CACHE = SafeLRUCache(2000)
_CORRECTION_CACHE = SafeLRUCache(10000)

print("🚀 Đang khởi tạo AI models...")

MODEL_DIR = BASE_DIR / "local_models" / "gliner_multi-v2.1"
try:
    gliner_model = GLiNER.from_pretrained(str(MODEL_DIR), local_files_only=True)
except Exception as e:
    print(f"⚠️ Cảnh báo: Không thể load GLiNER local: {e}")
    gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# ==========================================
# [Tối ưu 4]: Giảm bớt các component SpaCy
# ==========================================
def _load_spacy_model() -> spacy.Language:
    disabled_pipes = ["tok2vec", "parser", "tagger", "attribute_ruler", "lemmatizer", "senter", "sentencizer"]
    for model_name in ["vi_core_news_lg", "vi_core_news_sm", "xx_ent_wiki_sm"]:
        try:
            nlp = spacy.load(model_name, disable=disabled_pipes)
            print(f"✅ Loaded spaCy model: {model_name} (Optimized)")
            return nlp
        except OSError:
            continue
    print("⚠️ Dùng blank pipeline")
    return spacy.blank("vi")

nlp = _load_spacy_model()

# ==========================================
# [Tối ưu 4]: Serialize SymSpell bằng Pickle
# ==========================================
print("📚 Đang khởi tạo SymSpell...")
DICT_PATH = BASE_DIR / "dictionary" / "Viet74K.txt"
PICKLE_PATH = BASE_DIR / "dictionary" / "symspell_cache.pkl"

if PICKLE_PATH.exists():
    with open(PICKLE_PATH, "rb") as f:
        sym_spell = pickle.load(f)
    print("✅ Loaded SymSpell từ pickle cache!")
else:
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dictionary: {DICT_PATH}")

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                sym_spell.create_dictionary_entry(word, 1)

    DOMAIN_WORDS = {
        "quyết": 5000, "định": 5000, "bản": 4000, "án": 4000, "tòa": 4000, "bị": 4000, "cáo": 4000,
        "blhs": 9000, "bltths": 9000, "điều": 6000, "khoản": 6000, "điểm": 5000, "tội": 5000, 
        "danh": 4000, "phạt": 4000, "tù": 5000, "bhxh": 6000, "bảo": 5000, "hiểm": 3000, 
        "xã": 3000, "hội": 3000, "ủy": 3000, "ban": 3000, "nhân": 3000, "dân": 3000,
        "nguyễn": 8000, "trần": 8000, "lê": 8000, "phạm": 8000, "hoàng": 8000, "thị": 7000, 
        "văn": 7000, "tài": 5000, "sản": 5000, "minh": 9000, "anh": 9000, "bình": 8000, 
        "hồ": 7000, "chí": 7000, "thành": 7000, "phố": 6000, "quận": 6000, "huyện": 6000,
        "tỉnh": 6000, "phường": 6000, "xã": 5000, "nẵng": 8000, "huế": 8000, "đồng": 8000, 
        "nai": 8000, "dương": 8000, "hà": 9000, "nội": 9000, "hải": 8000, "phòng": 8000, 
        "cần": 8000, "thơ": 8000, "bắc": 7000, "ninh": 7000, "giang": 7000, "cp": 8000, 
        "tnhh": 8000, "mtv": 7000, "tổng": 7000, "cục": 7000, "chi": 7000, "nhánh": 7000, 
        "vụ": 6000, "viện": 6000
    }
    for word, freq in DOMAIN_WORDS.items():
        sym_spell.create_dictionary_entry(word, freq)
        
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(sym_spell, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ SymSpell khởi tạo và lưu pickle thành công!")

print("🤖 Đang khởi tạo Vietnamese Correction Model...")

_correction_tokenizer = None
_correction_model = None
CORRECTION_MODEL_DIR = BASE_DIR / "local_models" / "vietnamese-correction-v2"
_CORRECTION_HF_ID = "bmd1905/vietnamese-correction-v2"

def _load_correction_model(model_path: str, local_only: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
    try:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only, use_fast=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=local_only,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        mdl.eval()
        return tok, mdl
    except Exception as e:
        print(f" ⚠️ Lỗi load model từ '{model_path}': {e}")
        return None, None

if (CORRECTION_MODEL_DIR / "config.json").exists():
    _correction_tokenizer, _correction_model = _load_correction_model(str(CORRECTION_MODEL_DIR), local_only=True)
    if _correction_model is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _correction_model.to(device)
        print(f"✅ Loaded Vietnamese Correction Model từ LOCAL trên {device}")

if _correction_tokenizer is None:
    print("⚠️ Thử tải từ HuggingFace...")
    _correction_tokenizer, _correction_model = _load_correction_model(_CORRECTION_HF_ID, local_only=False)
    if _correction_model is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _correction_model.to(device)
        print(f"✅ Loaded Vietnamese Correction Model từ HuggingFace trên {device}")

CORRECTION_MODEL_AVAILABLE = _correction_tokenizer is not None and _correction_model is not None
print(f"📊 Correction Model: {'✅ Sẵn sàng' if CORRECTION_MODEL_AVAILABLE else '❌ Không khả dụng (chỉ dùng SymSpell)'}")

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("module3")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger

_LEGAL_ABBREVIATION_PROTECTED = frozenset({
    "TNHH", "CP", "MTV", "BHXH", "UBND", "HĐND", "DNTN", "CTCP",
})

_ORG_UNIT_FIELD_NAMES = frozenset({
    "ten_co_quan_to_chuc", "ten_don_vi", "kinh_gui", "ten_loai_van_ban",
    "ma_don_vi", "nguoi_ky",
})

_HARD_CORRECTIONS_RAW: Dict[str, str] = {
    "TROM CAP": "TRỘM CẮP", "TR0M CAP": "TRỘM CẮP", "TR0M CÁP": "TRỘM CẮP", "TROMCAP": "TRỘM CẮP",
    "TRÇN THÞ": "TRẦN THỊ", "Mò Cây Nam": "Mỏ Cày Nam", "THÞ": "THỊ", "ÇN": "ẦN",
    "NGÂN HÀN": "NGÂN HÀNG", "NHÁ NƯỚC": "NHÀ NƯỚC", "THÀNH PHÓ": "THÀNH PHỐ",
    "HỒ CHÍ M1NH": "HỒ CHÍ MINH", "VIỆT N4M": "VIỆT NAM", "BẢO HIỂM XÃ HỘl": "BẢO HIỂM XÃ HỘI",
    "C0NG TY": "CÔNG TY", "CỔ PHÀN": "CỔ PHẦN",
    "QUYẾT ĐINH": "QUYẾT ĐỊNH",
    "Quyêt định": "Quyết định", "CÔNG HÒA": "CỘNG HÒA", "XÃ HÔI": "XÃ HỘI",
    "ĐỌC LẬP": "ĐỘC LẬP", "TƯ DO": "TỰ DO", "HANH PHÚC": "HẠNH PHÚC",
    "NĂM SINH": "NĂM SINH",
}

_ABBREV_EXPANSION_CORRECTIONS = {
    "UBND": "Ủy ban nhân dân",
    "HĐND": "Hội đồng nhân dân",
    "TP": "Thành phố",
    "QĐ": "Quyết định",
    "TT": "Thông tư",
    "NĐ": "Nghị định"
}

_HARD_CORRECTIONS_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(re.escape(wrong)), correct) for wrong, correct in _HARD_CORRECTIONS_RAW.items()
]

_ABBREV_EXPANSION_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(wrong) + r"\b"), correct) for wrong, correct in _ABBREV_EXPANSION_CORRECTIONS.items()
]

LEGAL_DOCUMENT_TITLES_WHITELIST = {
    "QUYẾT ĐỊNH", "THÔNG BÁO", "QUYẾT NGHỊ", "NGHỊ QUYẾT", "THÔNG TƯ", 
    "CÔNG VĂN", "BIÊN BẢN", "BẢN ÁN", "LỆNH", "THÔNG CÁO", "CHỈ THỊ", 
    "NGHỊ ĐỊNH", "THÔNG TƯ LIÊN TỊCH", "HỢP ĐỒNG", "GIẤY CHỨNG NHẬN", 
    "GIẤY PHÉP", "GIẤY ỦY QUYỀN", "THỎA THUẬN", "BIÊN BẢN GHI NHỚ"
}

_LEADING_JUNK_RE = re.compile(r'^[\s:;\-!,\.]+')
_TRAILING_JUNK_RE = re.compile(r'[\s:;\-!,\.]+$')
_INLINE_EXCLAMATION_RE = re.compile(r'(?<!\w)!(?!\w)')
_ORPHAN_COLON_RE = re.compile(r'^:\s*')
_COLON_ALLOWED_FIELDS: set = set()

def post_process_value(value: str, field_name: str = "") -> str:
    if not value: return value
    if field_name in ["ho_ten_nguoi_tham_gia", "ten_bi_cao"]:
        value = re.split(r'(?i)\s*(?:,|\bsinh\b|\bCMND\b|\bCCCD\b)', value)[0]
    if field_name not in _COLON_ALLOWED_FIELDS:
        value = _ORPHAN_COLON_RE.sub("", value)
    value = _LEADING_JUNK_RE.sub("", value)
    value = _TRAILING_JUNK_RE.sub("", value)
    value = _INLINE_EXCLAMATION_RE.sub("", value)
    if field_name in ["ten_loai_van_ban", "ten_co_quan_to_chuc", "toi_danh"]:
        value = re.sub(r'([^\W\d_])\s*,\s*([^\W\d_])', r'\1 \2', value, flags=re.UNICODE)
    return re.sub(r'\s{2,}', ' ', value).strip()

_WHITELIST_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("date_slash",      re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b')),
    ("date_text",       re.compile(r'ngày\s*\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{4}', re.IGNORECASE)),
    ("so_bhxh",         re.compile(r'\b\d{10}\b')),
    ("so_van_ban",      re.compile(r'\b\d+\/(?:\d+\/)?[A-ZĐ][A-ZĐ0-9\-]+\b')),
    ("ma_don_vi",       re.compile(r'\b(?=.*\d)[A-Z0-9]{6,10}\b')),
    ("luat_reference",  re.compile(r'\b(?:Điều|Khoản|Điểm)\s+\d+[a-z]?\b', re.IGNORECASE)),
]

_NUMERIC_FIELD_NAMES = {
    "ngay_thang_nam", "so_quyet_dinh", "ky_hieu",
    "so_bhxh", "ma_don_vi", "so_to_khai",
    "so_tien", "gia_tri_tranh_chap", "gia_tri_tai_san_chiem_doat", "so_dien_thoai",
}

_FIXED_HEADER_FIELDS = frozenset({
    "so_quyet_dinh", "ky_hieu", "ngay_thang_nam",
    "ten_co_quan_to_chuc", "ten_loai_van_ban",
})

_DYNAMIC_SEMANTIC_FIELDS = frozenset({
    "ten_bi_cao", "toi_danh", "nguyen_don", "bi_don", "dia_chi",
    "gia_tri_tranh_chap", "gia_tri_tai_san_chiem_doat",
    "so_bhxh", "ma_don_vi", "so_tien", "so_dien_thoai", "kinh_gui",
    "trich_yeu", "nguoi_ky", "ten_don_vi", "thoi_han_dong",
    "ho_ten_nguoi_tham_gia", "noi_dung_tranh_chap",
})

def mask_whitelisted_spans(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    masked = text
    replacements: List[Tuple[str, str]] = []
    idx = 0
    for _, pat in _WHITELIST_PATTERNS:
        for m in reversed(list(pat.finditer(masked))):
            token = f"__WLMASK{idx}__"
            replacements.append((token, m.group(0)))
            masked = masked[:m.start()] + token + masked[m.end():]
            idx += 1
    return masked, replacements

def restore_masked_spans(text: str, replacements: List[Tuple[str, str]]) -> str:
    for token, original in replacements:
        text = text.replace(token, original)
        text = text.replace(token.lower(), original)
    return text

def apply_ocr_correction(text: str, confusions: Dict[str, str] = None, field_name: Optional[str] = None) -> str:
    if not text: return text
    confusions = confusions or {}
    cache_key = (text, tuple(sorted(confusions.items())), field_name or "")
    
    cached = _OCR_CACHE.get(cache_key)
    if cached is not None: return cached
    
    out = text
    for pattern, correct in _HARD_CORRECTIONS_COMPILED:
        out = pattern.sub(correct, out)
    if not (field_name and field_name in _ORG_UNIT_FIELD_NAMES):
        for pattern, correct in _ABBREV_EXPANSION_COMPILED:
            def _abbrev_repl(m: re.Match, _correct: str = correct) -> str:
                token = m.group(0)
                if token.upper() in _LEGAL_ABBREVIATION_PROTECTED: return token
                return _correct
            out = pattern.sub(_abbrev_repl, out)
    for wrong, correct in confusions.items():
        out = out.replace(wrong, correct)
    
    _OCR_CACHE.set(cache_key, out)
    return out

_PROPER_NOUN_ALLOWLIST = {w.lower() for w in [
    "MINH", "ANH", "BÌNH", "DŨNG", "HÒA", "HÀ", "HÙNG", "KHOA", "LONG",
    "MAI", "NAM", "NGA", "NHUNG", "PHONG", "PHÚC", "QUÂN", "THẮNG", "THU",
    "THỦY", "TIẾN", "TOÀN", "TRUNG", "TÙNG", "TUYẾN", "ÚT", "XUÂN",
    "BLHS", "BLTTHS", "BHXH", "BHYT", "TP", "HCM", "TPHCM",
]}

@lru_cache(maxsize=50000)
def _symspell_word_correction_cached(word: str) -> str:
    if not word.strip() or re.fullmatch(r"\d+", word):
        return word
    if word.lower() in _PROPER_NOUN_ALLOWLIST:
        return word
        
    # [Tối ưu 3]: Giải pháp bảo vệ danh từ riêng (Title Case an toàn bỏ qua SymSpell)
    if word.istitle() and len(word) > 3:
        return word
        
    suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
    if not suggestions: return word
    best = suggestions[0].term
    if word.isupper(): return best.upper()
    if word.istitle(): return best.title()
    return best

@lru_cache(maxsize=50000)
def _apply_symspell_correction_cached(text: str) -> str:
    if not text: return text
    tokens = re.findall(r"\w+|\S", text, re.UNICODE)
    corrected_tokens = []
    for token in tokens:
        if re.fullmatch(r"\w+", token, re.UNICODE):
            corrected_tokens.append(_symspell_word_correction_cached(token))
        else:
            corrected_tokens.append(token)
    out = " ".join(corrected_tokens)
    return re.sub(r"\s+([,.;:])", r"\1", out).strip()

def apply_symspell_correction(text: str) -> str:
    return _apply_symspell_correction_cached(text)

def _restore_uppercase_tokens(original: str, corrected: str) -> str:
    orig_tokens = original.split()
    corr_tokens = corrected.split()
    if len(orig_tokens) != len(corr_tokens): return corrected
    result_tokens = []
    for orig, corr in zip(orig_tokens, corr_tokens):
        if orig.isupper() and len(orig) > 1:
            result_tokens.append(corr.upper())
        else:
            result_tokens.append(corr)
    return " ".join(result_tokens)

def apply_vietnamese_correction_batch(
    texts: List[str],
    field_names: Optional[List[str]] = None,
    use_model: bool = True,
    batch_size: int = 32, # [Tối ưu 1]: Tăng batch size Model Correction
    confusions: Optional[Dict[str, str]] = None,
    return_sub_methods: bool = False,
) -> Any:
    if not texts:
        return ([], []) if return_sub_methods else []

    masked_list: List[str] = []
    replacements_list: List[List[Tuple[str, str]]] = []
    hard_stage: List[str] = []
    
    for idx, text in enumerate(texts):
        if not text or not text.strip():
            masked_list.append(text or "")
            replacements_list.append([])
            hard_stage.append(text or "")
            continue

        fname = field_names[idx] if field_names else None
        
        # [Tối ưu 1]: Early return bỏ qua SymSpell/Pipeline cho Numeric fields
        if fname in _NUMERIC_FIELD_NAMES:
            masked_list.append(text)
            replacements_list.append([])
            hard_stage.append(text)
            continue

        text_hard = apply_ocr_correction(text, confusions, field_name=fname)
        hard_stage.append(text_hard)
        masked, reps = mask_whitelisted_spans(text_hard)
        sym_out = apply_symspell_correction(masked)
        masked_list.append(sym_out)
        replacements_list.append(reps)

    if not use_model or not CORRECTION_MODEL_AVAILABLE:
        final = [
            post_process_value(restore_masked_spans(sym_out, reps), field_names[idx] if field_names else "")
            for idx, (sym_out, reps) in enumerate(zip(masked_list, replacements_list))
        ]
        if return_sub_methods:
            sub_methods = []
            for idx, (orig, fin) in enumerate(zip(texts, final)):
                hard = hard_stage[idx] if idx < len(hard_stage) else orig
                sym = masked_list[idx] if idx < len(masked_list) else orig
                if (orig or "").strip() == (fin or "").strip(): sub_methods.append(None)
                elif hard != orig: sub_methods.append("hard_ocr")
                else: sub_methods.append("symspell")
            return final, sub_methods
        return final

    device = next(_correction_model.parameters()).device
    cache_keys: List[str] = [hashlib.md5(s.encode("utf-8")).hexdigest() for s in masked_list]
    miss_indices: List[int] = []
    
    for i, k in enumerate(cache_keys):
        if k not in _CORRECTION_CACHE:
            # [Tối ưu 3]: Bỏ qua Deep Learning Model cho text siêu ngắn (<3 tokens)
            if len(masked_list[i].split()) < 3:
                _CORRECTION_CACHE.set(k, masked_list[i])
            else:
                miss_indices.append(i)

    if miss_indices:
        miss_texts = [masked_list[i] for i in miss_indices]

        for batch_start in range(0, len(miss_texts), batch_size):
            batch_texts = miss_texts[batch_start: batch_start + batch_size]
            batch_idx   = miss_indices[batch_start: batch_start + batch_size]
            try:
                inputs = _correction_tokenizer(
                    batch_texts, return_tensors="pt", max_length=128, truncation=True, padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    output_ids = _correction_model.generate(
                        **inputs, max_length=128, num_beams=1, do_sample=False, pad_token_id=_correction_tokenizer.pad_token_id,
                    )

                for local_i, (orig_i, out_id) in enumerate(zip(batch_idx, output_ids)):
                    corrected = _correction_tokenizer.decode(out_id, skip_special_tokens=True).strip()
                    sym_out = masked_list[orig_i]
                    
                    similarity = SequenceMatcher(None, sym_out.lower(), corrected.lower()).ratio()
                    is_all_upper = sym_out.isupper()
                    is_title_case = sym_out.istitle()
                    
                    orig_word_count = len(sym_out.split())
                    corr_word_count = len(corrected.split())
                    
                    if corr_word_count < orig_word_count and orig_word_count >= 3: result = sym_out
                    elif is_all_upper and similarity < 0.85: result = sym_out
                    elif is_title_case and similarity < 0.90: result = sym_out
                    elif similarity < 0.75: result = sym_out
                    elif corrected and len(corrected) > len(sym_out) * 0.6:
                        result = _restore_uppercase_tokens(sym_out, corrected)
                    else:
                        result = sym_out
                    
                    _CORRECTION_CACHE.set(cache_keys[orig_i], result)

            except Exception as e:
                logging.warning(f"Batch correction error (batch {batch_start}): {e}")
                for orig_i in batch_idx:
                    if cache_keys[orig_i] not in _CORRECTION_CACHE:
                        _CORRECTION_CACHE.set(cache_keys[orig_i], masked_list[orig_i])

    results: List[str] = []
    model_stage: List[str] = []
    for i, (sym_out, reps, key) in enumerate(zip(masked_list, replacements_list, cache_keys)):
        cached = _CORRECTION_CACHE.get(key, sym_out)
        restored = restore_masked_spans(cached, reps)
        fname = field_names[i] if field_names else ""
        final_val = post_process_value(restored, fname)
        results.append(final_val)
        model_stage.append(restored)

    if not return_sub_methods: return results

    sub_methods: List[Optional[str]] = []
    for idx, (orig, fin) in enumerate(zip(texts, results)):
        hard = hard_stage[idx] if idx < len(hard_stage) else (orig or "")
        sym = masked_list[idx] if idx < len(masked_list) else hard
        mdl = model_stage[idx] if idx < len(model_stage) else sym
        
        if (orig or "").strip() == (fin or "").strip(): sub_methods.append(None)
        elif mdl.strip() != sym.strip(): sub_methods.append("vietnamese_model")
        elif sym.strip() != hard.strip(): sub_methods.append("symspell")
        elif hard.strip() != (orig or "").strip(): sub_methods.append("hard_ocr")
        else: sub_methods.append("pipeline_correction")
        
    return results, sub_methods


def compute_field_confidence(line_conf: float, raw_value: str, corrected_value: str, rule: Dict[str, Any], full_text_line: str) -> float:
    score = line_conf
    value_regexes = rule.get("value_regexes", [])
    strong_match = any(re.fullmatch(rx, raw_value.strip()) for rx in value_regexes) if value_regexes else True
    weak_match = any(re.search(rx, raw_value) for rx in value_regexes) if value_regexes else False
    if strong_match: score += 0.12
    elif weak_match: score += 0.03
    else: score -= 0.08
    if raw_value != corrected_value: score -= 0.05
    label_regexes = rule.get("label_regexes", [])
    if any(re.search(rx, full_text_line, re.IGNORECASE) for rx in label_regexes): score += 0.08
    return max(0.0, min(1.0, round(score, 3)))


def route_template(text_lines: List[Dict[str, Any]], templates: List[Dict[str, Any]]) -> Tuple[str, float]:
    full_text_upper = " ".join(x.get("upper_text", str(x.get("text", "")).upper()) for x in text_lines)
    cache_key = hashlib.md5(full_text_upper.encode("utf-8")).hexdigest()
    
    cached = _ROUTE_CACHE.get(cache_key)
    if cached is not None: return cached
        
    best_tpl_id = "unknown"
    best_conf = 0.0

    for tpl in templates:
        tpl_id = tpl.get("template_id")
        detect = tpl.get("document_detection", {})
        keywords = detect.get("keywords") or []
        regexes = detect.get("regexes") or []
        neg_keywords = detect.get("negative_keywords") or []
        
        if any(nk.upper() in full_text_upper for nk in neg_keywords): continue

        kw_score = sum(3 for kw in keywords if kw.upper() in full_text_upper)
        regex_score = sum(4 for rx in regexes if re.search(rx, full_text_upper, re.IGNORECASE))
        total_score = kw_score + regex_score
        
        target_score = 7.0 
        conf = total_score / target_score if total_score > 0 else 0.0
        
        if conf > best_conf:
            best_conf = conf
            best_tpl_id = tpl_id

    result = ("unknown", 0.0) if best_conf == 0.0 else (best_tpl_id, round(min(1.0, best_conf), 3))
    _ROUTE_CACHE.set(cache_key, result)
    return result


def apply_normalizer(field_rule: Dict[str, Any], value: str, cfg: Dict[str, Any]) -> str:
    normalizer = field_rule.get("normalizer")
    auto_corr = cfg.get("auto_correction", {})
    is_numeric_field = normalizer in ["date_dmy_to_iso", "number_slash", "bhxh_fix_confusions"]
    confusions = auto_corr.get("ocr_confusions", {}) if (auto_corr.get("enabled") and is_numeric_field) else {}
    value = apply_ocr_correction(value, confusions)
    if normalizer == "date_dmy_to_iso": return normalize_date_dmy_to_iso(value) or value.strip()
    if normalizer == "number_slash": return normalize_number_generic(value) or value.strip()
    if normalizer == "name_strip": return normalize_name_strip(value)
    if normalizer == "bhxh_fix_confusions": return normalize_bhxh(value, confusions) or value.strip()
    return post_process_value(value.strip(), field_rule.get("field_name", ""))

def dedup_fixed_fields(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict] = {}
    for f in fields:
        name = f["field_name"]
        if name not in best or f["confidence"] > best[name]["confidence"]:
            best[name] = f
    return list(best.values())

def validate_field_value(field_name: str, value: str, method: str, conf: float, full_text: str = "") -> bool:
    if not value or value.strip().upper() == "XX" or value.strip() == "": return False
    if field_name == "ngay_thang_nam":
        if conf < 0.5 and method == "gliner_ner": return False
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", value))
    if field_name == "so_quyet_dinh":
        return bool(re.search(r"\d", value))
    if field_name == "so_bhxh":
        if full_text:
            val_clean = value.replace(" ", "")
            escaped_val = re.escape(val_clean)
            if re.search(r'(?i)(mã số thuế|mst)[^\n]{0,30}' + escaped_val, full_text) or \
               re.search(escaped_val + r'[^\n]{0,30}(?i)(mã số thuế|mst)', full_text):
                return False
        return bool(re.fullmatch(r"\d{9,14}", value.replace(" ", "")))
    if field_name in ["so_tien", "gia_tri_tranh_chap", "gia_tri_tai_san_chiem_doat"]:
        return bool(re.search(r"\d", value)) 
    if field_name in ["ten_bi_cao", "nguyen_don", "bi_don", "ho_ten_nguoi_tham_gia"]:
        return len(value.split()) >= 2 
    return True

def _truncate_at_other_labels(value: str, current_field: str, field_configs: Dict[str, Any]) -> str:
    if not value: return value
    cut_points: List[int] = []
    for fname, rule in field_configs.items():
        if fname == current_field: continue
        for rx in (rule.get("label_regexes") or []):
            try:
                m = re.search(rx, value, re.IGNORECASE)
                if m and m.start() > 0: cut_points.append(m.start())
            except re.error: continue
    if cut_points: return value[: min(cut_points)].strip()
    return value

_SECTION_BREAK_RE = re.compile(
    r"^\s*(Điều\s+\d+|Nội dung|Người đại diện|Tài sản|Kính gửi|Căn cứ|Theo đề nghị)\b", re.IGNORECASE,
)

def _is_safe_text_correction(field_name: str, before: str, after: str) -> bool:
    if not after: return False
    before_n = (before or "").strip()
    after_n = (after or "").strip()
    if not before_n: return True

    sim = SequenceMatcher(None, before_n.lower(), after_n.lower()).ratio()
    
    # [Tối ưu 3]: Tăng similarity cứng cáp cho các trường tổ chức, tên riêng lên 0.85 
    if field_name in {"ten_co_quan_to_chuc", "nguyen_don", "bi_don"} and sim < 0.85:
        return False
    if field_name in {"ten_bi_cao", "ten_don_vi", "dia_chi", "toi_danh"} and sim < 0.72:
        return False

    phrase_pairs = [("thanh toán", "thanh toán"), ("chiếm đoạt", "chiếm đoạt"), ("việt nam", "việt nam")]
    before_l = before_n.lower()
    after_l = after_n.lower()
    for required_before, required_after in phrase_pairs:
        if required_before in before_l and required_after not in after_l: return False

    before_digits = re.findall(r"\d", before_n)
    after_digits = re.findall(r"\d", after_n)
    if field_name in {"dia_chi", "toi_danh", "so_tien", "gia_tri_tranh_chap", "gia_tri_tai_san_chiem_doat"}:
        if before_digits and len(after_digits) < max(1, len(before_digits) - 2): return False

    return True

def _sanitize_org_text(value: str) -> str:
    if not value: return value
    trans = str.maketrans({"0": "O", "1": "I", "4": "A", "5": "S"})
    fixed = value.translate(trans)
    fixed = re.sub(r"\b([A-Za-zÀ-ỹ]{2,}?)([A-Za-zÀ-ỹ])\2\b", r"\1\2", fixed, flags=re.UNICODE)
    return re.sub(r"\s{2,}", " ", fixed).strip()

_STRUCTURAL_FIELDS = {"so_quyet_dinh", "ky_hieu", "ngay_thang_nam"}
_SO_PREFIX_RE = r"(?:Số|S[Ôô6Oo]|SO)\s*[:\.]?\s*"
_SO_VAN_BAN_FULL_RE = re.compile(_SO_PREFIX_RE + r"([\d]+(?:/[\d]+)?/[A-ZĐa-zđ][A-ZĐa-zđ0-9\-]+)", re.IGNORECASE | re.UNICODE)
_SO_VAN_BAN_SIMPLE_RE = re.compile(_SO_PREFIX_RE + r"(\d+)\s*$", re.IGNORECASE)

_VAN_BAN_TITLE_RE = re.compile(
    r"^(QUYẾT ĐỊNH|THÔNG BÁO|QUYẾT NGHỊ|NGHỊ QUYẾT|THÔNG TƯ|CÔNG VĂN|BIÊN BẢN|"
    r"BẢN ÁN|LỆNH|THÔNG CÁO|CHỈ THỊ|NGHỊ ĐỊNH|THÔNG TƯ LIÊN TỊCH|"
    r"HỢP ĐỒNG|GIẤY CHỨNG NHẬN|GIẤY PHÉP|GIẤY ỦY QUYỀN|QUYẾT NGHỊ|THỎA THUẬN|BIÊN BẢN GHI NHỚ)"
    r"(?:[\s\w\/\-\(\),\.]*$|(?=\s*(?:Số|S[Ôô6Oo]|SO)\s*[:\.\d]))",
    re.IGNORECASE | re.UNICODE,
)

_CO_QUAN_BASE_PREFIXES = (
    "TÒA ÁN|CƠ QUAN|ỦY BAN|BAN|TRUNG TÂM|VĂN PHÒNG|SỞ|PHÒNG|CỤC|BỘ|VIỆN|"
    "BHXH|BẢO HIỂM XÃ HỘI|BẢO HIỂM Y TẾ|UBND|HĐND|CHI CỤC|TỔNG CỤC|VỤ|BAN QUẢN LÝ|"
    "CÔNG TY|DOANH NGHIỆP|HỢP TÁC XÃ|QUỸ|NGÂN HÀNG|NGÂN HÀN|TRƯỜNG|BỆNH VIỆN|TRẠM|ĐỘI|"
    "LIÊN ĐOÀN|HỘI|ĐOÀN|TỔNG LIÊN ĐOÀN|HIỆP HỘI|LIÊN HIỆP|CHI NHÁNH|"
    "TRUNG ƯƠNG|ĐỊA PHƯƠNG|NHÂN DÂN|CHÍNH PHỦ|THỦ TƯỚNG|BỘ TRƯỞNG|"
    "NGÂN HÀNG NHÀ NƯỚC|NGÂN HÀN NHÀ NƯỚC|NGÂN HÀNG|NGÂN HÀN|BHXH|BẢO HIỂM XÃ HỘI|TÒA ÁN"
)

_CO_QUAN_RE = re.compile(r"^(" + _CO_QUAN_BASE_PREFIXES + r")[\s\w\/\-\(\),\.]*$", re.IGNORECASE | re.UNICODE)

def _rebuild_co_quan_re(extra_prefixes: List[str]) -> re.Pattern:
    if not extra_prefixes: return _CO_QUAN_RE
    extras = "|".join(re.escape(p.upper()) for p in extra_prefixes)
    pattern = r"^(" + _CO_QUAN_BASE_PREFIXES + r"|" + extras + r")[\s\w\/\-\(\),\.]*$"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

_GLINER_THRESHOLDS_DEFAULT: Dict[str, float] = {
    "ten_bi_cao": 0.55, "toi_danh": 0.50, "so_bhxh": 0.60,
    "ho_ten_nguoi_tham_gia": 0.55, "dia_chi": 0.45,
    "thoi_gian_dong_bhxh": 0.45, "_default": 0.42,
}
_GLINER_THRESHOLDS: Dict[str, float] = dict(_GLINER_THRESHOLDS_DEFAULT)

def _get_gliner_threshold(field_name: str) -> float:
    return _GLINER_THRESHOLDS.get(field_name, _GLINER_THRESHOLDS.get("_default", 0.42))

_DATE_SIGNATURE_RE = re.compile(
    r"(?:hà\s+nội|tp\.?\s*hồ\s*chí\s*minh|tp\.?\s*hcm|đà\s+nẵng|huế|"
    r"thành\s+phố\s+[\w\s]+|[\wÀ-ỹ]{2,20})\s*,\s*ngày\s+", re.IGNORECASE | re.UNICODE,
)
_DATE_DEADLINE_NOISE_RE = re.compile(
    r"\b(trước\s+ngày|đến\s+ngày|hạn\s+nộp|thời\s+hạn|chậm\s+đóng|"
    r"báo\s+cáo\s+trước|nộp\s+trước|hết\s+hạn)\b", re.IGNORECASE | re.UNICODE,
)
_DATE_BODY_NOISE_RE = re.compile(
    r"\b(xét\s+thấy|căn\s+cứ\s+điều|đương\s+sự|mở\s+phiên|theo\s+đề\s+nghị)\b", re.IGNORECASE | re.UNICODE,
)
_SO_LINE_RE = re.compile(r"(?i)(?:^|\b)(?:Số|S[Ôô6Oo]|SO)\s*[:\.]")

_WORD_TO_NUM = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
    "mười một": 11, "mười hai": 12, "mười ba": 13, "mười bốn": 14, "mười lăm": 15, "mười sáu": 16, 
    "mười bảy": 17, "mười tám": 18, "mười chín": 19, "hai mươi": 20, "hai mươi mốt": 21,
    "hai mươi hai": 22, "hai mươi ba": 23, "hai mươi bốn": 24, "hai mươi lăm": 25, "hai mươi sáu": 26, 
    "hai mươi bảy": 27, "hai mươi tám": 28, "hai mươi chín": 29, "ba mươi": 30, "ba mươi mốt": 31,
}
_WORD_NUM_PATTERN = "|".join(sorted(_WORD_TO_NUM.keys(), key=len, reverse=True))
_DATE_WORD_RE = re.compile(r"ngày\s+(" + _WORD_NUM_PATTERN + r")\s+tháng\s+(" + _WORD_NUM_PATTERN + r")\s+năm\s+(\d{4})", re.IGNORECASE)
_DATE_NUMERIC_WORD_RE = re.compile(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", re.IGNORECASE)

def _line_near_so_block(line_idx: int, text_lines: List[Dict[str, Any]], window: int = 4) -> bool:
    start = max(0, line_idx - window)
    end = min(len(text_lines), line_idx + window + 1)
    for i in range(start, end):
        if _SO_LINE_RE.search(text_lines[i].get("text", "")): return True
    return False

def _collect_ngay_thang_nam_candidates(text_string: str, line_idx: int, text_lines: List[Dict[str, Any]], bbox: List, line_conf: float) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    lower = text_string.lower()

    def _append(raw: str, iso: str, method: str, score: float) -> None:
        candidates.append({
            "field_name": "ngay_thang_nam", "raw_value": raw, "value": iso,
            "confidence": round(min(1.0, max(0.0, score)), 3), "bbox": bbox, "method": method, "_score": score,
        })

    is_signature = bool(_DATE_SIGNATURE_RE.search(text_string))
    near_so = _line_near_so_block(line_idx, text_lines)
    header_region = line_idx < 15

    num_match = _DATE_NUMERIC_WORD_RE.search(text_string)
    if num_match:
        d, m, y = num_match.groups()
        score = line_conf + 0.30
        if is_signature: score += 0.35
        elif near_so: score += 0.22
        elif header_region: score += 0.12
        if _DATE_BODY_NOISE_RE.search(lower): score -= 0.15
        _append(num_match.group(0), f"{y}-{int(m):02d}-{int(d):02d}", "regex", score)

    word_match = _DATE_WORD_RE.search(text_string)
    if word_match:
        d_str, m_str, y_str = word_match.groups()
        d_val = _WORD_TO_NUM.get(d_str.lower(), 0)
        m_val = _WORD_TO_NUM.get(m_str.lower(), 0)
        if d_val and m_val:
            score = line_conf + 0.28
            if is_signature: score += 0.38
            elif near_so: score += 0.25
            elif header_region: score += 0.15
            _append(word_match.group(0), f"{y_str}-{m_val:02d}-{d_val:02d}", "regex_word_date", score)

    if _DATE_DEADLINE_NOISE_RE.search(lower): return candidates

    slash_match = re.search(r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b", text_string)
    if slash_match and "Số:" not in text_string and "Số " not in text_string.upper():
        d, m, y = slash_match.groups()
        score = line_conf - 0.05
        if is_signature: score += 0.30
        elif near_so: score += 0.20
        elif header_region: score += 0.10
        else: score -= 0.25
        if _DATE_BODY_NOISE_RE.search(lower): score -= 0.40
        if score >= line_conf - 0.20:
            _append(slash_match.group(0), f"{y}-{int(m):02d}-{int(d):02d}", "regex", score)

    return candidates

def _pick_best_ngay_thang_nam(text_lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    all_candidates: List[Dict[str, Any]] = []
    for idx, line in enumerate(text_lines):
        text_string = line.get("text", "").strip()
        if not text_string: continue
        bbox = line.get("bbox", [0, 0, 0, 0])
        line_conf = float(line.get("confidence", 0.0))
        all_candidates.extend(_collect_ngay_thang_nam_candidates(text_string, idx, text_lines, bbox, line_conf))
    
    # [Tối ưu 2]: Giải quyết fallback toàn văn ngay khi không lấy được candidates ưu tiên
    if not all_candidates:
        full_text_concat = " ".join(l.get("text", "") for l in text_lines)
        date_match = re.search(r'(?i)ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', full_text_concat)
        
        if date_match:
            d, m, y = date_match.groups()
            return {
                "field_name": "ngay_thang_nam", "raw_value": date_match.group(0),
                "value": f"{y}-{int(m):02d}-{int(d):02d}", "confidence": 0.45,
                "bbox": [0, 0, 0, 0], "method": "regex_fallback"
            }
        slash_match = re.search(r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b', full_text_concat)
        if slash_match:
            d, m, y = slash_match.groups()
            return {
                "field_name": "ngay_thang_nam", "raw_value": slash_match.group(0),
                "value": f"{y}-{int(m):02d}-{int(d):02d}", "confidence": 0.40,
                "bbox": [0, 0, 0, 0], "method": "regex_fallback"
            }
        return None
        
    best = max(all_candidates, key=lambda c: c["_score"])
    return {k: v for k, v in best.items() if k != "_score"}

def _extract_structural(text_string: str, bbox: List, line_conf: float, confusions: Dict[str, str]) -> List[Dict[str, Any]]:
    results = []
    match = _SO_VAN_BAN_FULL_RE.search(text_string)
    if match:
        raw_full = match.group(1)
        fixed_full = apply_ocr_correction(raw_full, confusions)
        parts = fixed_full.split("/")
        if len(parts) >= 2:
            if parts[1].isdigit(): so_qd = f"{parts[0]}/{parts[1]}"; ky_hieu_idx = 2
            else: so_qd = parts[0]; ky_hieu_idx = 1
        else: so_qd = parts[0]; ky_hieu_idx = -1
            
        results.append({
            "field_name": "so_quyet_dinh", "raw_value": raw_full, "value": post_process_value(so_qd),
            "confidence": line_conf, "bbox": bbox, "method": "regex",
        })
        if ky_hieu_idx != -1 and len(parts) > ky_hieu_idx:
            results.append({
                "field_name": "ky_hieu", "raw_value": raw_full, "value": post_process_value("/".join(parts[ky_hieu_idx:])),
                "confidence": line_conf, "bbox": bbox, "method": "regex",
            })
    else:
        simple = _SO_VAN_BAN_SIMPLE_RE.search(text_string)
        if simple:
            results.append({
                "field_name": "so_quyet_dinh", "raw_value": simple.group(1), "value": post_process_value(simple.group(1)),
                "confidence": line_conf - 0.05, "bbox": bbox, "method": "regex_simple",
            })
    return results

def _extract_config_driven(text_string: str, bbox: List, line_conf: float, field_configs: Dict[str, Any], confusions: Dict[str, str], all_text_lines: Optional[List[Dict[str, Any]]] = None, allowed_fields: Optional[frozenset] = None) -> List[Dict[str, Any]]:
    results = []
    for field_name, rule in field_configs.items():
        if field_name in _STRUCTURAL_FIELDS: continue
        if allowed_fields is not None and field_name not in allowed_fields: continue
        if allowed_fields is None and field_name in _FIXED_HEADER_FIELDS: continue
        
        label_regexes = rule.get("label_regexes") or []
        value_regexes = rule.get("value_regexes") or []
        strategy = rule.get("value_strategy", "same_line")
        
        if strategy == "pattern_match" or not label_regexes: continue
        if not any(re.search(rx, text_string, re.IGNORECASE) for rx in label_regexes): continue
            
        value = None
        for v_rx in value_regexes:
            try:
                m = re.search(v_rx, text_string, re.IGNORECASE)
                if m: value = m.group(1).strip() if m.lastindex else m.group(0).strip(); break
            except re.error: continue
                
        if not value and strategy == "same_line":
            for l_rx in label_regexes:
                m = re.search(l_rx, text_string, re.IGNORECASE)
                if m:
                    after = text_string[m.end():].strip()
                    if after:
                        value = re.sub(r"^[:\-\s]+", "", after).strip()
                        value = _truncate_at_other_labels(value, field_name, field_configs)
                        break
                        
        if not value and strategy == "next_line" and all_text_lines:
            for i, line in enumerate(all_text_lines):
                if any(re.search(rx, line.get("text", ""), re.IGNORECASE) for rx in label_regexes):
                    if i + 1 < len(all_text_lines):
                        next_text = all_text_lines[i + 1].get("text", "").strip()
                        if next_text: value = re.sub(r"^[:\-\s]+", "", next_text).strip()
                    break

        if not value and strategy == "next_n_lines" and all_text_lines:
            n = rule.get("next_n_lines", 2)
            for i, line in enumerate(all_text_lines):
                line_text = line.get("text", "")
                if any(re.search(rx, line_text, re.IGNORECASE) for rx in label_regexes):
                    parts = []
                    for l_rx in label_regexes:
                        m = re.search(l_rx, line_text, re.IGNORECASE)
                        if m:
                            after = line_text[m.end():].strip()
                            if after: parts.append(re.sub(r"^[:\-\s]+", "", after).strip())
                            break
                    for j in range(i + 1, min(i + 1 + n, len(all_text_lines))):
                        t = all_text_lines[j].get("text", "").strip()
                        if t:
                            if _SECTION_BREAK_RE.search(t): break
                            other_label_hit = False
                            for fname, frule in field_configs.items():
                                if fname == field_name: continue
                                for rx in (frule.get("label_regexes") or []):
                                    try:
                                        if re.search(rx, t, re.IGNORECASE): other_label_hit = True; break
                                    except re.error: continue
                                if other_label_hit: break
                            if other_label_hit: break
                            parts.append(re.sub(r"^[:\-\s]+", "", t).strip())
                    value = " ".join(parts) if parts else None
                    if value: value = _truncate_at_other_labels(value, field_name, field_configs)
                    break

        if not value and strategy == "until_blank" and all_text_lines:
            _all_labels = [rx for r in field_configs.values() for rx in r.get("label_regexes", [])]
            for i, line in enumerate(all_text_lines):
                if any(re.search(rx, line.get("text", ""), re.IGNORECASE) for rx in label_regexes):
                    parts = []
                    for j in range(i + 1, len(all_text_lines)):
                        t = all_text_lines[j].get("text", "").strip()
                        if not t: break
                        if any(re.search(rx, t, re.IGNORECASE) for rx in _all_labels): break
                        parts.append(re.sub(r"^[:\-\s]+", "", t).strip())
                    value = " ".join(parts) if parts else None
                    break
                    
        if not value: continue
            
        raw_val_for_log = value
        value = apply_ocr_correction(value, confusions if field_name in _NUMERIC_FIELD_NAMES else {}, field_name=field_name)
        value = post_process_value(value, field_name)
        conf = compute_field_confidence(line_conf, value, value, rule, text_string)
        
        results.append({
            "field_name": field_name, "raw_value": raw_val_for_log, "value": value, 
            "confidence": conf, "bbox": bbox, "method": "regex"
        })
    return results

def _extract_ten_co_quan(text_lines: List[Dict[str, Any]], confusions: Dict[str, str], co_quan_re: Optional[re.Pattern] = None) -> Optional[Dict[str, Any]]:
    co_quan_re = co_quan_re or _CO_QUAN_RE
    SKIP_RE = re.compile(r"CỘNG HÒA|ĐỘC LẬP|TỰ DO|HẠNH PHÚC", re.IGNORECASE)
    cong_hoa_idx = -1
    
    for i, line in enumerate(text_lines):
        if re.search(r"CỘNG HÒA|CONG HOA", line.get("text", "").upper()): cong_hoa_idx = i; break

    def _try_match(lines, confidence_default):
        for line in lines:
            raw = line.get("text", "").strip()
            text = apply_ocr_correction(raw, {})
            if '\n' in text: text = text.split('\n')[0].strip()
            text = re.split(r'(?i)(CỘNG HÒA|CONG HOA)', text)[0].strip()
            if re.match(r"(?i)^\s*(Số|SO|S[Ôô6Oo])\s*[:\.]", text): continue
            if re.search(r"\d+\/[A-ZĐa-zđ0-9\-]+", text): continue
            if not text or SKIP_RE.search(text) or len(text.split()) > 15: continue
            if co_quan_re.match(text.upper()):  
                return {
                    "field_name": "ten_co_quan_to_chuc", "raw_value": raw,
                    "value": post_process_value(normalize_name_strip(text), "ten_co_quan_to_chuc"),
                    "confidence": confidence_default, "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
                }
        return None

    start_idx = max(0, cong_hoa_idx - 5) if cong_hoa_idx != -1 else 0
    end_idx = (cong_hoa_idx + 15) if cong_hoa_idx != -1 else min(20, len(text_lines))
    
    result = _try_match(text_lines[start_idx: end_idx], 0.9)
    if result: return result

    strong_header_re = re.compile(r"(?i)\b(TÒA ÁN|ỦY BAN NHÂN DÂN|UBND|BHXH|BẢO HIỂM XÃ HỘI|NGÂN HÀNG NHÀ NƯỚC)\b")
    for line in text_lines[:8]:
        raw = line.get("text", "").strip()
        text = apply_ocr_correction(raw, {})
        if "\n" in text: text = text.split("\n")[0].strip()
        text = re.split(r"(?i)(CỘNG HÒA|CONG HOA)", text)[0].strip()
        if re.match(r"(?i)^\s*(Số|SO|S[Ôô6Oo])\s*[:\.]", text): continue
        if re.search(r"\d+\/[A-ZĐa-zđ0-9\-]+", text): continue
        if not text or SKIP_RE.search(text): continue
        if strong_header_re.search(text):
            return {
                "field_name": "ten_co_quan_to_chuc", "raw_value": raw,
                "value": post_process_value(normalize_name_strip(text), "ten_co_quan_to_chuc"),
                "confidence": 0.86, "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }

    all_text = " ".join(apply_ocr_correction(line.get("text", ""), {}) for line in text_lines[:10])
    if all_text.strip():
        doc = nlp(all_text[:500])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "FAC"):
                ent_text = ent.text.strip()
                if len(ent_text) > 5 and not re.search(r"CỘNG HÒA|ĐỘC LẬP|TỰ DO|HẠNH PHÚC|VIỆT NAM", ent_text, re.IGNORECASE):
                    return {
                        "field_name": "ten_co_quan_to_chuc", "value": post_process_value(normalize_name_strip(ent_text), "ten_co_quan_to_chuc"),
                        "confidence": 0.65, "bbox": [0, 0, 0, 0], "method": "spacy_ner",
                    }
    return None

def _extract_ten_loai_van_ban(text_lines: List[Dict[str, Any]], confusions: Dict[str, str]) -> Optional[Dict[str, Any]]:
    title_keyword_re = re.compile(r"^(QUYẾT ĐỊNH|THÔNG BÁO|CÔNG VĂN|BẢN ÁN|THÔNG TƯ|NGHỊ ĐỊNH|NGHỊ QUYẾT)\b", re.IGNORECASE)
    cong_hoa_idx = -1
    so_idx = len(text_lines)
    for i, line in enumerate(text_lines):
        txt = line.get("text", "").upper()
        if "CỘNG HÒA" in txt and cong_hoa_idx == -1: cong_hoa_idx = i
        if re.search(r"\bSỐ\s*:", txt) and i > cong_hoa_idx: so_idx = i; break

    search_end = min(len(text_lines), so_idx + 5)
    search_start = max(0, cong_hoa_idx) if cong_hoa_idx != -1 else 0
    search_lines = text_lines[search_start:search_end] if cong_hoa_idx != -1 else text_lines[:min(20, len(text_lines))]
    for line in search_lines:
        raw = line.get("text", "").strip()
        text = apply_ocr_correction(raw, {})
        if not text: continue
        if title_keyword_re.search(text.strip()):
            return {
                "field_name": "ten_loai_van_ban", "raw_value": raw, "value": post_process_value(text.strip(), "ten_loai_van_ban"),
                "confidence": round(float(line.get("confidence", 0.92)), 3), "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }
        if _VAN_BAN_TITLE_RE.match(text.upper()):
            return {
                "field_name": "ten_loai_van_ban", "raw_value": raw, "value": post_process_value(text.strip(), "ten_loai_van_ban"),
                "confidence": round(float(line.get("confidence", 0.9)), 3), "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }

    for line in text_lines:
        raw = line.get("text", "").strip()
        text = apply_ocr_correction(raw, {})
        if not text: continue
        if title_keyword_re.search(text.strip()):
            return {
                "field_name": "ten_loai_van_ban", "raw_value": raw, "value": post_process_value(text.strip(), "ten_loai_van_ban"),
                "confidence": round(float(line.get("confidence", 0.87)), 3), "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }
        if _VAN_BAN_TITLE_RE.match(text.upper()):
            return {
                "field_name": "ten_loai_van_ban", "value": post_process_value(text.strip(), "ten_loai_van_ban"),
                "confidence": round(float(line.get("confidence", 0.85)), 3), "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }
    return None

def extract_fixed_fields(text_lines: List[Dict[str, Any]], confusions: Dict[str, str] = None, field_configs: Dict[str, Any] = None, extra_org_prefixes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if confusions is None: confusions = {}
    if field_configs is None: field_configs = {}

    co_quan_re = _rebuild_co_quan_re(extra_org_prefixes or [])
    results = []
    has_strict_header_so = any(re.search(r"(?i)^Số\s*[:\.]", l.get("text", "").strip()) for l in text_lines[:15])

    for idx, line in enumerate(text_lines):
        raw_text = line.get("text", "").strip()
        text_string = apply_ocr_correction(raw_text, confusions)
        bbox = line.get("bbox", [0, 0, 0, 0])
        line_conf = float(line.get("confidence", 0.0))

        structural_results = _extract_structural(text_string, bbox, line_conf, confusions)
        for r in structural_results:
            if r["field_name"] in ["so_quyet_dinh", "ky_hieu", "ngay_thang_nam"]:
                if re.search(r"^(Căn cứ|Theo|Tại|Xét|Về việc|Điều.*của|Thông báo số|Thông tư số|Nghị định số|ban hành)\b", raw_text, re.IGNORECASE):
                    if r["field_name"] != "ngay_thang_nam": r["confidence"] = max(0.0, r["confidence"] - 0.7)
                elif idx > 15:
                    is_so_line = bool(re.search(r"(?i)(Số|S[Ôô6Oo]|SO)\s*[:\.]", raw_text))
                    r["confidence"] = min(1.0, r["confidence"] + 0.2) if is_so_line else max(0.0, r["confidence"] - 0.2)
                elif idx < 10:
                    if re.search(r"(?i)(Số|S[Ôô6Oo]|SO)\s*[:\.]", raw_text): r["confidence"] = min(1.0, r["confidence"] + 0.2)
                    elif has_strict_header_so: r["confidence"] = max(0.0, r["confidence"] - 0.4)
                    else: r["confidence"] = min(1.0, r["confidence"] + 0.1)
            results.append(r)

    best_date = _pick_best_ngay_thang_nam(text_lines)
    if best_date: results.append(best_date)

    co_quan = _extract_ten_co_quan(text_lines, confusions, co_quan_re=co_quan_re)
    if co_quan: results.append(co_quan)

    ten_loai = _extract_ten_loai_van_ban(text_lines, confusions)
    if ten_loai: results.append(ten_loai)

    return dedup_fixed_fields(results)

def _split_text_into_chunks(text: str, max_chunk_size: int = 400) -> list:
    """
    Chia nhỏ văn bản thành các đoạn (chunks) dựa trên số lượng từ.
    Đã đổi tên (bỏ _spacy) để phản ánh đúng logic không phụ thuộc thư viện ngoài.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Kiểm tra xem thêm từ hiện tại có vượt quá kích thước chunk không
        if current_length + len(word) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 cho khoảng trắng

    # Push chunk cuối cùng nếu còn
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def _find_bbox_for_value(corrected_value: str, raw_value: str, text_lines: List[Dict]) -> List:
    if raw_value:
        for line in text_lines:
            if raw_value in line.get("text", ""): return line.get("bbox", [0, 0, 0, 0])
    for line in text_lines:
        if corrected_value in line.get("text", ""): return line.get("bbox", [0, 0, 0, 0])

    corr_tokens = set(corrected_value.lower().split())
    if not corr_tokens: return [0, 0, 0, 0]

    best_line = None
    max_overlap = 0
    for line in text_lines:
        line_tokens = set(line.get("text", "").lower().split())
        overlap = len(corr_tokens.intersection(line_tokens))
        if overlap > max_overlap:
            max_overlap = overlap; best_line = line

    if best_line and max_overlap >= len(corr_tokens) * 0.4:
        return best_line.get("bbox", [0, 0, 0, 0])
    return [0, 0, 0, 0]

_GLINER_LABEL_FALLBACK: Dict[str, str] = {
    "tên bị cáo": "ten_bi_cao", "tội danh": "toi_danh", "cơ quan ban hành": "ten_co_quan_to_chuc",
    "họ tên người tham gia": "ho_ten_nguoi_tham_gia", "địa chỉ": "dia_chi",
    "thời gian đóng": "thoi_gian_dong_bhxh", "giá trị tài sản chiếm đoạt": "gia_tri_tai_san_chiem_doat",
    "tài sản chiếm đoạt": "gia_tri_tai_san_chiem_doat",
}

def build_gliner_label_map(field_configs: Dict[str, Any]) -> Dict[str, str]:
    label_map: Dict[str, str] = dict(_GLINER_LABEL_FALLBACK)
    _GLINER_THRESHOLDS.clear()
    _GLINER_THRESHOLDS.update(_GLINER_THRESHOLDS_DEFAULT)
    for field_name, rule in field_configs.items():
        for gliner_label in rule.get("gliner_labels", []): label_map[gliner_label.lower().strip()] = field_name
        if "gliner_threshold" in rule: _GLINER_THRESHOLDS[field_name] = float(rule["gliner_threshold"])
    return label_map

def map_gliner_to_config_fields(gliner_label: str, label_map: Optional[Dict[str, str]] = None) -> str:
    if label_map: return label_map.get(gliner_label.lower().strip(), gliner_label)
    return _GLINER_LABEL_FALLBACK.get(gliner_label, gliner_label)

def _spacy_ner_fallback(text: str, field_configs: dict) -> dict:
    """
    Hàm fallback trích xuất dữ liệu khi mô hình chính miss.
    Đã gỡ bỏ pipeline spaCy NER, chỉ giữ lại xử lý Regex tốc độ cao.
    """
    results = {}
    
    for field_name, config in field_configs.items():
        fallback_rules = config.get('fallback_rules', [])
        if not fallback_rules:
            continue
            
        for pattern in fallback_rules:
            try:
                # Chạy regex tìm kiếm trên toàn bộ text
                match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    # Lấy capture group 1 (nếu có định nghĩa trong regex), ngược lại lấy toàn bộ match
                    extracted_value = match.group(1) if match.lastindex else match.group(0)
                    results[field_name] = extracted_value.strip()
                    break  # Đã tìm thấy dữ liệu cho field này, bỏ qua các pattern regex sau đó
            except re.error as e:
                print(f"Lỗi cú pháp Regex tại field '{field_name}', pattern '{pattern}': {e}")
                continue
                
    return results

def extract_dynamic_fields(
    full_text: str, required_fields: List[str], extracted_fixed_fields: List[Dict[str, Any]], 
    text_lines: List[Dict], confusions: Dict[str, str] = None, field_configs: Dict[str, Any] = None, 
    zero_shot_configs: List[Dict[str, str]] = None, gliner_label_map: Dict[str, str] = None # Đưa gliner map ra ngoài
) -> List[Dict[str, Any]]:
    if confusions is None: confusions = {}
    if field_configs is None: field_configs = {}
    if zero_shot_configs is None: zero_shot_configs = []

    results = []
    REGEX_HANDLED_FIELDS = {"so_quyet_dinh", "ten_co_quan_to_chuc", "ten_loai_van_ban"}
    existing_fields = [f["field_name"] for f in extracted_fixed_fields]
    forbidden_gliner_values = set()
    for f in extracted_fixed_fields:
        val = str(f.get("value", "")).lower().strip()
        if len(val) > 3 and f["field_name"] in ["ten_bi_cao", "ho_ten_nguoi_tham_gia"]: forbidden_gliner_values.add(val)

    template_fields = set(required_fields)
    all_possible_fields = [f for f in (field_configs.keys() if field_configs else []) if f in template_fields]
    zero_shot_names = [z["name"] for z in zero_shot_configs if z["name"] in template_fields]
    all_possible_fields.extend(zero_shot_names)

    missing_fields = [f for f in all_possible_fields if f not in existing_fields and f not in REGEX_HANDLED_FIELDS]
    if not missing_fields: return results

    existing_after_label = {f["field_name"] for f in results}
    label_missing = [f for f in missing_fields if f not in existing_after_label]
    if label_missing:
        dynamic_allowed = frozenset(label_missing)
        for idx, line in enumerate(text_lines):
            raw_text = line.get("text", "").strip()
            text_string = apply_ocr_correction(raw_text, confusions)
            bbox = line.get("bbox", [0, 0, 0, 0])
            line_conf = float(line.get("confidence", 0.0))
            label_hits = _extract_config_driven(text_string, bbox, line_conf, field_configs, confusions, all_text_lines=text_lines, allowed_fields=dynamic_allowed)
            for hit in label_hits:
                fname = hit["field_name"]
                if fname not in dynamic_allowed: continue
                if fname not in {r["field_name"] for r in results}:
                    hit["method"] = hit.get("method", "regex") + "_label"
                    results.append(hit)

    missing_fields = [f for f in missing_fields if f not in {r["field_name"] for r in results}]
    if not missing_fields: return results

    label_map = gliner_label_map if gliner_label_map else build_gliner_label_map(field_configs)
    
    target_labels = []
    for field in missing_fields:
        rule = field_configs.get(field, {})
        gliner_labels = rule.get("gliner_labels", [])
        if gliner_labels: target_labels.extend(gliner_labels)
        else:
            z_config = next((z for z in zero_shot_configs if z["name"] == field), None)
            if z_config:
                target_labels.append(z_config["description"])
                label_map[z_config["description"].lower().strip()] = field
            else:
                for glabel, fname in _GLINER_LABEL_FALLBACK.items():
                    if fname == field: target_labels.append(glabel)
    
    target_labels = list(set(target_labels))
    
    # [Tối ưu 1]: GLiNER Batching Inference thay vì vòng lặp tĩnh 
    if target_labels:
        clean_full_text = apply_ocr_correction(full_text, confusions)
        raw_gliner_results = []
        
        # [Guard Clause]: Bỏ qua nếu text quá ngắn
        if len(clean_full_text) >= 50:
            chunks = _split_text_into_chunks(clean_full_text, max_chunk_size=500)
            is_short_text = len(clean_full_text) < 200 
            
            try:
                with torch.inference_mode():
                    # Ưu tiên dùng API mới 'inference' để tránh warning
                    if hasattr(gliner_model, 'inference'):
                        batch_predictions = gliner_model.inference(chunks, target_labels, flat_ner=True, threshold=0.25)
                    elif hasattr(gliner_model, 'batch_predict_entities'):
                        batch_predictions = gliner_model.batch_predict_entities(chunks, target_labels, flat_ner=True, threshold=0.25)
                    else:
                        batch_predictions = [gliner_model.predict_entities(c, target_labels, flat_ner=True, threshold=0.25) for c in chunks]

                    for chunk_idx, predictions in enumerate(batch_predictions):
                        for p in predictions:
                            label = p["label"].lower().strip()
                            text_val = p["text"].strip()
                            score = float(p["score"])
                            field_name = map_gliner_to_config_fields(label, label_map)
                            
                            if field_name not in missing_fields: continue
                            threshold = _get_gliner_threshold(field_name)
                            if score < (threshold * 0.8 if is_short_text else threshold): continue
                            if text_val.lower() in forbidden_gliner_values or len(text_val) < 2: continue
                                
                            raw_gliner_results.append({
                                "field_name": field_name, "raw_value": text_val, "value": post_process_value(text_val, field_name),
                                "confidence": round(score, 3), "bbox": _find_bbox_for_value(text_val, text_val, text_lines), "method": "gliner_ner",
                            })
            except Exception as e:
                logging.getLogger("module3").warning(f"Lỗi khi chạy GLiNER Inference Batch: {e}")

        best_results = {}
        for r in raw_gliner_results:
            fname = r["field_name"]
            if fname not in best_results or r["confidence"] > best_results[fname]["confidence"]:
                best_results[fname] = r
                
        results.extend(list(best_results.values()))

    still_missing = [f for f in missing_fields if f not in [res["field_name"] for res in results]]
    if still_missing:
        # Fallback regex sẽ chạy cho mọi trường hợp (kể cả text ngắn bị skip bên trên)
        fallback_results = _spacy_ner_fallback(full_text, field_configs)
        
        # Map lại định dạng dictionary của spacy fallback về chuẩn list kết quả
        for fname, val in fallback_results.items():
            if fname in still_missing:
                results.append({
                    "field_name": fname, "raw_value": val, "value": val, 
                    "confidence": 0.40, "bbox": [0, 0, 0, 0], "method": "spacy_ner_fallback"
                })

    return results

def _process_single_file(
    p: Path, cfg: Dict[str, Any], out_schema: Dict, output_dir: Path, logger: logging.Logger, 
    TEXT_FIELDS_FOR_CORRECTION: List[str], field_configs: Dict[str, Any], global_gliner_map: Dict[str, str] = None
) -> Dict[str, Any]:
    file_start_time = time.time()
    stat = {"file": p.name, "status": "error", "fields": 0, "elapsed": 0.0, "doc_type": "unknown"}

    try:
        in_json = load_json(str(p))
        doc_id = in_json.get("document_id", "unknown")
        req_id = in_json.get("request_id", str(uuid.uuid4()))
        payload = in_json.get("payload", {})
        full_text = payload.get("text", "")

        text_lines = []
        blocks = payload.get("blocks") or []
        
        def _normalize_for_match(s: str) -> str: return re.sub(r'\s+', '', s).lower() if s else ""
        last_bbox = [52, 30, 620, 50] 
        
        # [Tối ưu 2]: Lấy ngưỡng fallback từ tham số yaml (mặc định 6)
        min_blocks = cfg.get("extraction", {}).get("min_blocks_threshold", 6)
        if len(blocks) < min_blocks:
            for raw_line in full_text.split("\n"):
                raw_line = raw_line.strip()
                if not raw_line: continue
                h = 20; new_y1 = last_bbox[3] + 4; new_y2 = new_y1 + h
                interpolated_bbox = [last_bbox[0], new_y1, last_bbox[2], new_y2]
                text_lines.append({"text": raw_line, "bbox": interpolated_bbox, "confidence": 0.8})
                last_bbox = interpolated_bbox
        else:
            for raw_line in full_text.split("\n"):
                raw_line = raw_line.strip()
                if not raw_line: continue
                raw_clean = _normalize_for_match(raw_line)
                matched_block = next((b for b in blocks if raw_clean in _normalize_for_match(b.get("text", "")) or _normalize_for_match(b.get("text", "")) in raw_clean), None)
                if matched_block:
                    raw_conf = float(matched_block.get("confidence", 0.85))
                    normalized_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
                    bbox = matched_block.get("bbox", [0, 0, 0, 0])
                    text_lines.append({"text": matched_block.get("text", ""), "bbox": bbox, "confidence": normalized_conf})
                    if bbox != [0, 0, 0, 0]: last_bbox = bbox
                    blocks.remove(matched_block)
                else:
                    h = last_bbox[3] - last_bbox[1] if last_bbox[3] > last_bbox[1] else 20
                    new_y1 = last_bbox[3] + 4; new_y2 = new_y1 + h
                    interpolated_bbox = [last_bbox[0], new_y1, last_bbox[2], new_y2]
                    text_lines.append({"text": raw_line, "bbox": interpolated_bbox, "confidence": 0.8})
                    last_bbox = interpolated_bbox

        templates = cfg.get("router", {}).get("templates") or []
        doc_type, routing_conf = route_template(text_lines, templates)
        
        if doc_type == "unknown":
            logger.warning(f"⚠️ {p.name}: routing_confidence=0 — Fallback về 'unknown', chỉ trích xuất các trường cấu trúc")
            required_fields = ["so_quyet_dinh", "ngay_thang_nam", "ten_co_quan_to_chuc", "ten_loai_van_ban"]
        else:
            matched_tpl = next((t for t in templates if t["template_id"] == doc_type), {})
            required_fields = matched_tpl.get("required_fields", [])

        auto_enabled = cfg.get("auto_correction", {}).get("enabled", False)
        extra_org_prefixes = cfg.get("extraction", {}).get("extra_org_prefixes", [])
        auto_corr_cfg = cfg.get("auto_correction", {})
        confusions_global = auto_corr_cfg.get("ocr_confusions", {}) if auto_corr_cfg.get("enabled") else {}

        fixed_fields = extract_fixed_fields(text_lines, confusions_global, field_configs=field_configs, extra_org_prefixes=extra_org_prefixes)
        zero_shot_configs = cfg.get("zero_shot_fields", [])
        
        dynamic_fields = extract_dynamic_fields(
            full_text, required_fields, fixed_fields, text_lines, confusions_global, 
            field_configs=field_configs, zero_shot_configs=zero_shot_configs, gliner_label_map=global_gliner_map
        )

        dynamic_fields_final = []
        for field in dynamic_fields:
            rule = field_configs.get(field["field_name"], {})
            raw_val = field["value"]
            corrected_val = apply_normalizer(rule, raw_val, cfg)
            field["value"] = corrected_val
            dynamic_fields_final.append(field)

        raw_fixed_fields = [f for f in fixed_fields if f["field_name"] in required_fields]
        allowed_dynamic_fields = set(required_fields + list(field_configs.keys()))
        raw_dynamic_fields = [f for f in dynamic_fields_final if f["field_name"] in allowed_dynamic_fields]
        
        validated_fixed_fields = [f for f in raw_fixed_fields if validate_field_value(f["field_name"], f["value"], f.get("method", ""), f.get("confidence", 1.0), full_text)]
        validated_dynamic_fields = [f for f in raw_dynamic_fields if validate_field_value(f["field_name"], f["value"], f.get("method", ""), f.get("confidence", 1.0), full_text)]

        correction_sub_methods: Dict[str, Optional[str]] = {}
        if auto_enabled:
            all_extracted_fields = validated_fixed_fields + validated_dynamic_fields
            correction_targets = []

            for field in all_extracted_fields:
                fname = field["field_name"]
                fval = str(field["value"])

                if fname == "ten_loai_van_ban" and fval.upper() in LEGAL_DOCUMENT_TITLES_WHITELIST:
                    field["correction_skip"] = True
                    continue
                
                if fname == "ten_co_quan_to_chuc":
                    corrected_hard = apply_ocr_correction(fval, {}, field_name=fname)
                    clean_val = re.sub(r'[^\w\s\-\.,&]', '', corrected_hard)
                    field["value"] = _sanitize_org_text(clean_val).upper().strip()
                    field["correction_skip"] = True
                    continue

                if fname in TEXT_FIELDS_FOR_CORRECTION and fname not in _NUMERIC_FIELD_NAMES:
                    correction_targets.append(field)

            if correction_targets:
                target_values = [field["value"] for field in correction_targets]
                target_fields = [field["field_name"] for field in correction_targets]

                corrected_values, sub_methods = apply_vietnamese_correction_batch(
                    target_values, field_names=target_fields, confusions={}, return_sub_methods=True,
                )

                for field, after, sub_m in zip(correction_targets, corrected_values, sub_methods):
                    before = str(field.get("value", ""))
                    if _is_safe_text_correction(field["field_name"], before, after):
                        field["value"] = after
                        if sub_m and before.strip() != str(after).strip():
                            correction_sub_methods[field["field_name"]] = sub_m

        changes_log = []
        all_validated = validated_fixed_fields + validated_dynamic_fields
        for field in all_validated:
            raw_val = field.get("raw_value", str(field["value"]))
            final_val = str(field["value"])
            clean_raw = raw_val.strip(".,:; \n").lower()
            clean_final = final_val.strip(".,:; \n").lower()

            if clean_raw != clean_final and clean_raw != "":
                fname = field["field_name"]
                sub_method = correction_sub_methods.get(fname) if auto_enabled else None
                if not sub_method and field.get("correction_skip"):
                    sub_method = "hard_ocr" if clean_raw != clean_final else None
                
                entry = {"original": raw_val, "corrected": final_val, "field": fname, "method": "pipeline_correction"}
                if sub_method: entry["sub_method"] = sub_method
                changes_log.append(entry)

        def _strip_internal_keys(f: Dict[str, Any]) -> Dict[str, Any]: return {k: v for k, v in f.items() if k not in {"correction_skip", "raw_value"}}

        merged_by_name: Dict[str, Dict[str, Any]] = {}
        for f in validated_fixed_fields + validated_dynamic_fields: merged_by_name[f["field_name"]] = f

        final_fixed = [_strip_internal_keys(merged_by_name[fname]) for fname in merged_by_name if fname in _FIXED_HEADER_FIELDS]
        final_dynamic = [_strip_internal_keys(merged_by_name[fname]) for fname in merged_by_name if fname not in _FIXED_HEADER_FIELDS]

        found_fields = {f["field_name"] for f in validated_fixed_fields + validated_dynamic_fields}
        found_required_fields = {f for f in found_fields if f in required_fields}
        missing_required = []
        full_text_lower = full_text.lower()
        
        for f in required_fields:
            if f not in found_required_fields:
                reason = "extraction_failed"
                rule = field_configs.get(f, {})
                presence_rx = rule.get("presence_check_regex")

                if presence_rx:
                    if not re.search(presence_rx, full_text_lower, re.IGNORECASE): reason = "not_in_source"
                else:
                    if f == "ngay_thang_nam" and not re.search(r'\b(ngày|tháng|năm|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b', full_text_lower): reason = "not_in_source"
                
                missing_required.append({"field": f, "reason": reason})
        
        coverage = round(len(found_required_fields) / len(required_fields), 2) if required_fields else 1.0

        out_json = {
            "request_id": req_id, "document_id": doc_id, "timestamp": now_iso_utc(), "status": "success", "error": None,
            "payload": {
                "document_type": doc_type, "routing_confidence": routing_conf, "field_coverage": coverage, "missing_required_fields": missing_required,
                "extracted_fields": {"fixed": final_fixed, "dynamic": final_dynamic},
                "correction_log": {"enabled": auto_enabled, "changes": changes_log},
            },
        }

        validate(instance=out_json, schema=out_schema)

        out_path = output_dir / f"{p.stem}_extracted.json"
        out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = time.time() - file_start_time
        total_fields = len(validated_fixed_fields) + len(validated_dynamic_fields)
        stat.update({"status": "success", "fields": total_fields, "elapsed": elapsed, "doc_type": doc_type})
        logger.info(f"✅ {p.name} -> {doc_type} | Fields: {total_fields} | Time: {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.time() - file_start_time
        stat["elapsed"] = elapsed
        logger.error(f"❌ Lỗi {p.name}: {e} | Time: {elapsed:.2f}s")

    return stat

def process_folder(
    input_dir: Path, output_dir: Path, config_path: str, schema_path: str, 
    config_schema_path: str, dry_run: bool = False, max_workers: int = None
) -> None:
    logger = setup_logger()
    cfg = load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    # [Tối ưu 1]: Tăng số lượng process song song tốt nhất theo số lượng nhân CPU.
    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 1) + 4)

    total_start_time = time.time()
    field_configs: Dict[str, Any] = cfg.get("fields", {})

    _HARDCODE_TEXT_FIELDS = {
        "ten_bi_cao", "toi_danh", "ten_co_quan_to_chuc", "ten_loai_van_ban",
        "ho_ten_nguoi_tham_gia", "dia_chi",
    }
    TEXT_FIELDS_FOR_CORRECTION: List[str] = []
    for fname, frule in field_configs.items():
        if "correction" in frule:
            if frule["correction"]: TEXT_FIELDS_FOR_CORRECTION.append(fname)
        else:
            if fname not in _NUMERIC_FIELD_NAMES: TEXT_FIELDS_FOR_CORRECTION.append(fname)
    for f in _HARDCODE_TEXT_FIELDS:
        if f not in TEXT_FIELDS_FOR_CORRECTION: TEXT_FIELDS_FOR_CORRECTION.append(f)

    json_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".json")

    if dry_run:
        logger.info(f"🔍 Dry-run: tìm thấy {len(json_files)} file JSON. Không xử lý thực tế.")
        for p in json_files: logger.info(f" - {p.name}")
        return

    # [Tối ưu 4]: Cache label map ra global
    global_gliner_map = build_gliner_label_map(field_configs)

    all_stats = []
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_file,
                    p, cfg, out_schema, output_dir, logger,
                    TEXT_FIELDS_FOR_CORRECTION, field_configs, global_gliner_map
                ): p for p in json_files
            }
            for future in concurrent.futures.as_completed(futures):
                all_stats.append(future.result())
    else:
        for p in json_files:
            stat = _process_single_file(
                p, cfg, out_schema, output_dir, logger,
                TEXT_FIELDS_FOR_CORRECTION, field_configs, global_gliner_map
            )
            all_stats.append(stat)

    total_elapsed = time.time() - total_start_time
    success_stats = [s for s in all_stats if s["status"] == "success"]
    error_stats = [s for s in all_stats if s["status"] == "error"]

    logger.info(f"{'='*50}")
    logger.info(f"🎉 Hoàn thành: {len(success_stats)}/{len(all_stats)} file thành công")
    
    total_fields_extracted = 0
    total_corrections_made = 0
    
    for s in success_stats:
        out_path = output_dir / f"{Path(s['file']).stem}_extracted.json"
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f: out_data = json.load(f)
                payload = out_data.get("payload", {})
                extracted = payload.get("extracted_fields", {})
                fixed_count = len(extracted.get("fixed", []))
                dynamic_count = len(extracted.get("dynamic", []))
                
                total_fields_extracted += (fixed_count + dynamic_count)
                changes = payload.get("correction_log", {}).get("changes", [])
                total_corrections_made += len(changes)
            except Exception: pass

    if success_stats:
        avg_time = sum(s["elapsed"] for s in success_stats) / len(success_stats)
        avg_fields = sum(s["fields"] for s in success_stats) / len(success_stats)
        logger.info(f"   Thời gian trung bình / file: {avg_time:.2f}s")
        logger.info(f"   Số fields trung bình / file: {avg_fields:.1f}")
        logger.info(f"   Tổng số fields trích xuất: {total_fields_extracted}")
        if total_fields_extracted > 0:
            correction_rate = (total_corrections_made / total_fields_extracted) * 100
            logger.info(f"   Tỷ lệ Correction tác động (Correction Impact Rate): {correction_rate:.2f}% ({total_corrections_made} fields)")
    if error_stats: logger.info(f"   ❌ Lỗi: {[s['file'] for s in error_stats]}")
    logger.info(f"   Tổng thời gian: {total_elapsed:.2f}s")