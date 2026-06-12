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
try:
    from rapidfuzz.distance import Levenshtein as _LevDist
    def _similarity_ratio(a: str, b: str) -> float:
        return 1.0 - _LevDist.normalized_distance(a, b)
except ImportError:
    from difflib import SequenceMatcher
    def _similarity_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

import spacy
import torch
from gliner import GLiNER
from jsonschema import validate
from symspellpy.symspellpy import SymSpell, Verbosity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as hf_logging

from src.utils import (
    load_json,
    load_yaml,
    now_iso_utc,
    normalize_bhxh,
    normalize_date_dmy_to_iso,
    normalize_name_strip,
    normalize_number_generic,
)

# ==========================================
# THIẾT KẾ ĐỐI TƯỢNG ĐIỀU PHỐI HỆ THỐNG (ENGINE CONTEXT)
# ==========================================
class ExtractionEngineContext:
    """
    Quản lý toàn bộ vòng đời, trạng thái và các biểu thức đã compiled động
    từ file cấu hình config.yaml để loại bỏ hoàn toàn các cấu trúc hằng số toàn cục.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        
        # 1. Trích xuất hằng số và cấu hình chung
        self.nlp_const = cfg.get("nlp_constants", {})
        self.auto_corr = cfg.get("auto_correction", {})
        self.fields_cfg = cfg.get("fields", {})
        
        self.proper_noun_allowlist = {w.lower() for w in self.nlp_const.get("proper_noun_allowlist", [])}
        self.legal_doc_titles_whitelist = set(self.nlp_const.get("legal_document_titles_whitelist", []))
        self.org_unit_field_names = frozenset(self.nlp_const.get("org_unit_field_names", []))
        self.legal_abbreviation_protected = frozenset(self.nlp_const.get("legal_abbreviation_protected", []))
        
        # Phân nhóm động các fields từ thuộc tính config
        self.numeric_fields = frozenset({k for k, v in self.fields_cfg.items() if v.get("is_numeric")})
        self.fixed_header_fields = frozenset({k for k, v in self.fields_cfg.items() if v.get("category") == "fixed"})
        self.colon_allowed_fields = frozenset({k for k, v in self.fields_cfg.items() if v.get("colon_allowed")})
        
        # 2. Compile Regex Cấu trúc hệ thống từ file yaml
        patterns = self.nlp_const.get("patterns", {})
        self.section_break_re = re.compile(patterns.get("section_break", r"^$"), re.IGNORECASE | re.UNICODE)
        self.so_prefix_re = patterns.get("so_prefix", "")
        self.so_van_ban_full_re = re.compile(patterns.get("so_van_ban_full", r"^$"), re.IGNORECASE | re.UNICODE)
        self.so_van_ban_simple_re = re.compile(patterns.get("so_van_ban_simple", r"^$"), re.IGNORECASE)
        self.van_ban_title_re = re.compile(patterns.get("van_ban_title", r"^$"), re.IGNORECASE | re.UNICODE)
        self.date_signature_re = re.compile(patterns.get("date_signature", r"^$"), re.IGNORECASE | re.UNICODE)
        self.date_deadline_noise_re = re.compile(patterns.get("date_deadline_noise", r"^$"), re.IGNORECASE | re.UNICODE)
        self.date_body_noise_re = re.compile(patterns.get("date_body_noise", r"^$"), re.IGNORECASE | re.UNICODE)
        self.so_line_re = re.compile(patterns.get("so_line", r"^$"), re.IGNORECASE)
        self.co_quan_base_prefixes = patterns.get("co_quan_base_prefixes", "")
        
        # 3. Bản đồ Whitelist Patterns (để mask thực thể số/ngày trước khi sửa lỗi)
        self.whitelist_patterns = [
            (item["name"], re.compile(item["regex"], re.IGNORECASE))
            for item in self.nlp_const.get("whitelist_patterns", [])
        ]
        
        # 4. Bản đồ Sửa nhiễu OCR Cố định (Hard Corrections)
        self.hard_corrections_compiled = [
            (re.compile(re.escape(wrong)), correct)
            for wrong, correct in self.auto_corr.get("hard_corrections", {}).items()
        ]
        
        # 5. Bản đồ Mở rộng Từ viết tắt (Abbreviation Expansions)
        self.abbrev_expansions_compiled = [
            (re.compile(r"(?<![\w\-])" + re.escape(wrong) + r"(?![\w\-])"), correct)
            for wrong, correct in self.auto_corr.get("abbrev_expansions", {}).items()
        ]
        
        # 6. Pre-compile toàn bộ Regex của nhãn (Label regexes) phục vụ tăng tốc cắt cụm dữ liệu
        self.compiled_labels_cache = []
        for fname, rule in self.fields_cfg.items():
            for rx in rule.get("label_regexes", []):
                try:
                    self.compiled_labels_cache.append((fname, re.compile(rx, re.IGNORECASE)))
                except re.error:
                    continue

        # 7. [FIX-P1-A] word_to_num — compile một lần, không rebuild mỗi dòng
        _w2n_cfg = self.nlp_const.get("word_to_num", {
            "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
            "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
            "mười một": 11, "mười hai": 12, "mười ba": 13, "mười bốn": 14,
            "mười lăm": 15, "mười sáu": 16, "mười bảy": 17, "mười tám": 18,
            "mười chín": 19, "hai mươi": 20, "hai mươi mốt": 21,
            "hai mươi hai": 22, "hai mươi ba": 23, "hai mươi bốn": 24,
            "hai mươi lăm": 25, "hai mươi sáu": 26, "hai mươi bảy": 27,
            "hai mươi tám": 28, "hai mươi chín": 29, "ba mươi": 30,
            "ba mươi mốt": 31,
        })
        self.word_to_num: Dict[str, int] = _w2n_cfg
        _w2n_pattern = "|".join(sorted(_w2n_cfg.keys(), key=len, reverse=True))
        self.word_date_re = re.compile(
            r"ngày\s+(" + _w2n_pattern + r")\s+tháng\s+(" + _w2n_pattern + r")\s+năm\s+(\d{4})",
            re.IGNORECASE
        )

        # 8. [FIX-P1-B] title_keyword_re — từ legal_document_titles_whitelist, không hardcode
        _title_kws = "|".join(re.escape(t) for t in sorted(
            self.legal_doc_titles_whitelist, key=len, reverse=True
        )) or "QUYẾT ĐỊNH"
        self.title_keyword_re = re.compile(r"^(" + _title_kws + r")\b", re.IGNORECASE)

        # 9. [FIX-P1-C] SKIP_RE — từ config noise_keywords, không hardcode
        _noise_kws = self.nlp_const.get(
            "header_noise_keywords",
            ["CỘNG HÒA", "ĐỘC LẬP", "TỰ DO", "HẠNH PHÚC"]
        )
        self.header_noise_re = re.compile(
            "|".join(re.escape(k) for k in _noise_kws), re.IGNORECASE
        )
        # pattern riêng chỉ để tìm dòng CỘNG HÒA (dùng xác định cong_hoa_idx)
        _cong_hoa_kws = self.nlp_const.get("cong_hoa_keywords", ["CỘNG HÒA", "CONG HOA"])
        self.cong_hoa_re = re.compile(
            "|".join(re.escape(k) for k in _cong_hoa_kws), re.IGNORECASE
        )

        # 10. [FIX-P1-D] strong_header_re — từ co_quan_base_prefixes, không hardcode
        _strong_kws = self.nlp_const.get(
            "strong_org_keywords",
            ["TÒA ÁN", "ỦY BAN NHÂN DÂN", "UBND", "BHXH", "BẢO HIỂM XÃ HỘI", "NGÂN HÀNG NHÀ NƯỚC"]
        )
        self.strong_header_re = re.compile(
            r"(?i)\b(" + "|".join(re.escape(k) for k in _strong_kws) + r")\b"
        )

        # 11. [FIX-P1-E] co_quan_re — compile một lần từ config, không rebuild mỗi lần gọi
        _extra_prefixes = cfg.get("extraction", {}).get("extra_org_prefixes", [])
        _extra_part = (r"|(?:" + "|".join(re.escape(p.upper()) for p in _extra_prefixes) + r")") if _extra_prefixes else ""
        self.co_quan_re = re.compile(
            r"^(" + self.co_quan_base_prefixes + _extra_part + r")[\s\w\/\-\(\),\.]*$",
            re.IGNORECASE | re.UNICODE
        )

        # 12. [FIX-G] GLiNER global inference threshold — từ config, không hardcode 0.25
        self.gliner_inference_threshold = float(
            cfg.get("models", {}).get("gliner", {}).get("inference_threshold", 0.25)
        )

        # 13. [FIX-F] Routing target score — từ config, không hardcode 7.0
        self.routing_target_score = float(
            cfg.get("engine_settings", {}).get("routing_target_score", 7.0)
        )

        # 14. [FIX-H] Ngưỡng số từ tối thiểu để chạy correction model — từ config
        self.correction_min_words = int(
            cfg.get("auto_correction", {}).get("min_words_for_model", 3)
        )

        # 15. [FIX-I/J/K/L] Các magic-number window — từ config
        extraction_cfg = cfg.get("extraction", {})
        self.header_line_limit    = int(extraction_cfg.get("header_line_limit", 15))
        self.so_line_boundary     = int(extraction_cfg.get("so_line_boundary", 10))
        self.date_context_window  = int(extraction_cfg.get("date_context_window", 5))
        self.cong_hoa_search_window = int(extraction_cfg.get("cong_hoa_search_window", 15))
        self.spacy_max_chars      = int(extraction_cfg.get("spacy_max_chars", 500))
        self.interpolated_conf    = float(extraction_cfg.get("interpolated_line_confidence", 0.8))


# ==========================================
# Cache LRU an toàn (Thread-safe)
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


def setup_system_resources(cfg: Dict[str, Any]) -> Tuple[SafeLRUCache, SafeLRUCache, SafeLRUCache]:
    set_ratio = cfg.get("engine_settings", {}).get("max_workers_ratio", 0.5)
    _CPU_PHYSICAL = max(1, int((os.cpu_count() or 2) * set_ratio))
    torch.set_num_threads(_CPU_PHYSICAL)
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message="Asking to truncate to max_length but no maximum length is provided.*")
    
    cap = cfg.get("engine_settings", {}).get("caches", {})
    return (
        SafeLRUCache(cap.get("ocr_capacity", 10000)),
        SafeLRUCache(cap.get("route_capacity", 2000)),
        SafeLRUCache(cap.get("correction_capacity", 10000))
    )

# Gọi tải tài nguyên ban đầu từ file config mặc định
BASE_DIR = Path(__file__).resolve().parent.parent
INITIAL_CFG = load_yaml(str(BASE_DIR / "config.yaml"))
_OCR_CACHE, _ROUTE_CACHE, _CORRECTION_CACHE = setup_system_resources(INITIAL_CFG)

print("🚀 Đang khởi tạo AI models từ cấu hình động...")
_GLINER_CFG = INITIAL_CFG.get("models", {}).get("gliner", {})
try:
    gliner_model = GLiNER.from_pretrained(_GLINER_CFG.get("local_path"), local_files_only=True)
except Exception as e:
    print(f"⚠️ Không thể load GLiNER local: {e}. Đang tải từ HF...")
    gliner_model = GLiNER.from_pretrained(_GLINER_CFG.get("fallback_hf_id", "urchade/gliner_multi-v2.1"))

def _load_spacy_model(cfg: Dict[str, Any]) -> spacy.Language:
    spacy_cfg = cfg.get("models", {}).get("spacy", {})
    disabled_pipes = spacy_cfg.get("disabled_pipes", [])
    for model_name in spacy_cfg.get("models_priority", []):
        try:
            nlp_obj = spacy.load(model_name, disable=disabled_pipes)
            print(f"✅ Loaded spaCy model: {model_name}")
            return nlp_obj
        except OSError:
            continue
    return spacy.blank("vi")

nlp = _load_spacy_model(INITIAL_CFG)

print("📚 Đang khởi tạo SymSpell...")
_SYM_CFG = INITIAL_CFG.get("models", {}).get("symspell", {})
DICT_PATH = BASE_DIR / _SYM_CFG.get("dict_path", "dictionary/Viet74K.txt")
PICKLE_PATH = BASE_DIR / _SYM_CFG.get("pickle_path", "dictionary/symspell_cache.pkl")

if PICKLE_PATH.exists():
    with open(PICKLE_PATH, "rb") as f:
        sym_spell = pickle.load(f)
    print("✅ Loaded SymSpell từ pickle cache thành công!")
else:
    sym_spell = SymSpell(
        max_dictionary_edit_distance=_SYM_CFG.get("max_dictionary_edit_distance", 2),
        prefix_length=_SYM_CFG.get("prefix_length", 7)
    )
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file từ điển: {DICT_PATH}")

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                sym_spell.create_dictionary_entry(word, 1)

    domain_words = INITIAL_CFG.get("auto_correction", {}).get("domain_words", {})
    for word, freq in domain_words.items():
        sym_spell.create_dictionary_entry(word, freq)
        
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(sym_spell, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Sinh và Lưu mới thành công SymSpell pickle cache!")

print("🤖 Đang khởi tạo Vietnamese Correction Model...")
_CORR_CFG = INITIAL_CFG.get("models", {}).get("correction", {})
_correction_tokenizer = None
_correction_model = None

def _load_correction_model(model_path: str, local_only: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
    try:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only, use_fast=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, local_files_only=local_only,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        mdl.eval()
        return tok, mdl
    except Exception:
        return None, None

if (BASE_DIR / _CORR_CFG.get("local_path") / "config.json").exists():
    _correction_tokenizer, _correction_model = _load_correction_model(str(BASE_DIR / _CORR_CFG.get("local_path")), local_only=True)

if _correction_tokenizer is None:
    _correction_tokenizer, _correction_model = _load_correction_model(_CORR_CFG.get("hf_id"), local_only=False)

if _correction_model is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _correction_model.to(device)
    try:
        _correction_model = torch.compile(_correction_model, mode="reduce-overhead")
    except Exception:
        pass

CORRECTION_MODEL_AVAILABLE = _correction_tokenizer is not None and _correction_model is not None

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("module3")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


# ==========================================
# PIPELINE XỬ LÝ SỬA LỖI VÀ CHUẨN HÓA OCR (CONFIG-DRIVEN)
# ==========================================
def mask_whitelisted_spans(text: str, ctx: ExtractionEngineContext) -> Tuple[str, List[Tuple[str, str]]]:
    masked = text
    replacements: List[Tuple[str, str]] = []
    idx = 0
    for _, pat in ctx.whitelist_patterns:
        for m in reversed(list(pat.finditer(masked))):
            token = f"__WLMASK{idx}__"
            replacements.append((token, m.group(0)))
            masked = masked[:m.start()] + token + masked[m.end():]
            idx += 1
    return masked, replacements


def restore_masked_spans(text: str, replacements: List[Tuple[str, str]]) -> str:
    for token, original in replacements:
        idx_match = re.search(r'\d+', token)
        if idx_match:
            idx = idx_match.group(0)
            flex_pattern = r"_*\s*W\s*L\s*M\s*A\s*S\s*K\s*" + idx + r"\s*_*"
            text = re.sub(flex_pattern, lambda _, orig=original: orig, text, flags=re.IGNORECASE)
        else:
            text = re.sub(re.escape(token), lambda _, orig=original: orig, text, flags=re.IGNORECASE)
    return text


def apply_ocr_correction(text: str, ctx: ExtractionEngineContext, confusions: Dict[str, str] = None, field_name: Optional[str] = None) -> str:
    if not text: return text
    confusions = confusions or {}
    cache_key = (text, tuple(sorted(confusions.items())), field_name or "")
    
    cached = _OCR_CACHE.get(cache_key)
    if cached is not None: return cached
    
    out = text
    for pattern, correct in ctx.hard_corrections_compiled:
        out = pattern.sub(correct, out)
        
    if not (field_name and field_name in ctx.org_unit_field_names):
        for pattern, correct in ctx.abbrev_expansions_compiled:
            def _abbrev_repl(m: re.Match, _correct=correct) -> str:
                token = m.group(0)
                if token.upper() in ctx.legal_abbreviation_protected: return token
                return _correct
            out = pattern.sub(_abbrev_repl, out)
            
    for wrong, correct in confusions.items():
        out = out.replace(wrong, correct)
    
    _OCR_CACHE.set(cache_key, out)
    return out


@lru_cache(maxsize=50000)
def _symspell_word_correction_cached(word: str, allowlist_frozen: frozenset) -> str:
    """Cache per (word, allowlist). allowlist_frozen là frozenset để hashable."""
    if not word.strip() or re.fullmatch(r"\d+", word): return word
    if word.lower() in allowlist_frozen: return word
    if word.istitle() and len(word) <= 3: return word

    suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
    if not suggestions: return word
    best = suggestions[0].term
    if word.isupper(): return best.upper()
    if word.istitle(): return best.title()
    return best


def apply_symspell_correction(text: str, ctx: ExtractionEngineContext) -> str:
    if not text: return text
    tokens = re.findall(r"\w+|\S", text, re.UNICODE)
    corrected_tokens = []
    allowlist_frozen = frozenset(ctx.proper_noun_allowlist)
    for token in tokens:
        if re.fullmatch(r"\w+", token, re.UNICODE):
            corrected_tokens.append(_symspell_word_correction_cached(token, allowlist_frozen))
        else:
            corrected_tokens.append(token)
    out = " ".join(corrected_tokens)
    return re.sub(r"\s+([,.;:])", r"\1", out).strip()


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
    ctx: ExtractionEngineContext, texts: List[str], field_names: Optional[List[str]] = None,
    use_model: bool = True, batch_size: int = 64, confusions: Optional[Dict[str, str]] = None,
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
        if fname in ctx.numeric_fields:
            masked_list.append(text)
            replacements_list.append([])
            hard_stage.append(text)
            continue

        text_hard = apply_ocr_correction(text, ctx, confusions, field_name=fname)
        hard_stage.append(text_hard)
        masked, reps = mask_whitelisted_spans(text_hard, ctx)
        sym_out = apply_symspell_correction(masked, ctx)
        masked_list.append(sym_out)
        replacements_list.append(reps)

    if not use_model or not CORRECTION_MODEL_AVAILABLE:
        final = [
            post_process_value(ctx, restore_masked_spans(sym_out, reps), field_names[idx] if field_names else "")
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
    cache_keys = [hashlib.md5(s.encode("utf-8")).hexdigest() for s in masked_list]
    miss_indices = []
    
    for i, k in enumerate(cache_keys):
        if k not in _CORRECTION_CACHE:
            if len(masked_list[i].split()) < ctx.correction_min_words:
                _CORRECTION_CACHE.set(k, masked_list[i])
            else:
                miss_indices.append(i)

    if miss_indices:
        texts_to_infer = []
        indices_to_infer = []
        for i in miss_indices:
            text = masked_list[i]
            if text == hard_stage[i] and re.fullmatch(r"[\w\s\.,\-]+", text, re.UNICODE):
                _CORRECTION_CACHE.set(cache_keys[i], text)
            else:
                texts_to_infer.append(text)
                indices_to_infer.append(i)

        for batch_start in range(0, len(texts_to_infer), batch_size):
            batch_texts = texts_to_infer[batch_start: batch_start + batch_size]
            batch_idx = indices_to_infer[batch_start: batch_start + batch_size]
            try:
                inputs = _correction_tokenizer(batch_texts, return_tensors="pt", max_length=128, truncation=True, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    output_ids = _correction_model.generate(**inputs, max_length=128, num_beams=1, do_sample=False, pad_token_id=_correction_tokenizer.pad_token_id)

                for local_i, (orig_i, out_id) in enumerate(zip(batch_idx, output_ids)):
                    corrected = _correction_tokenizer.decode(out_id, skip_special_tokens=True).strip()
                    sym_out = masked_list[orig_i]
                    
                    similarity = _similarity_ratio(sym_out.lower(), corrected.lower())
                    is_all_upper = sym_out.isupper()
                    is_title_case = sym_out.istitle()
                    
                    if len(corrected.split()) < len(sym_out.split()) and len(sym_out.split()) >= 3: result = sym_out
                    elif is_all_upper and similarity < 0.85: result = sym_out
                    elif is_title_case and similarity < 0.90: result = sym_out
                    elif similarity < 0.75: result = sym_out
                    elif corrected and len(corrected) > len(sym_out) * 0.6:
                        result = _restore_uppercase_tokens(sym_out, corrected)
                    else:
                        result = sym_out
                    
                    _CORRECTION_CACHE.set(cache_keys[orig_i], result)
            except Exception as e:
                for orig_i in batch_idx:
                    if cache_keys[orig_i] not in _CORRECTION_CACHE:
                        _CORRECTION_CACHE.set(cache_keys[orig_i], masked_list[orig_i])

    results = []
    model_stage = []
    for i, (sym_out, reps, key) in enumerate(zip(masked_list, replacements_list, cache_keys)):
        cached = _CORRECTION_CACHE.get(key, sym_out)
        restored = restore_masked_spans(cached, reps)
        final_val = post_process_value(ctx, restored, field_names[i] if field_names else "")
        results.append(final_val)
        model_stage.append(restored)

    if not return_sub_methods: return results

    sub_methods = []
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


def post_process_value(ctx: ExtractionEngineContext, value: str, field_name: str = "") -> str:
    if not value: return value
    
    # 1. Bổ sung danh sách cắt rác cho các trường nhân danh
    # Thêm nguyen_don, bi_don, nguoi_dai_dien để loại bỏ triệt để phần " - sinh năm..."
    if field_name in ["ho_ten_nguoi_tham_gia", "ten_bi_cao", "nguyen_don", "bi_don", "nguoi_dai_dien"]:
        # Tự động cắt ngay tại chữ "sinh", "CMND", "CCCD" hoặc dấu phẩy
        value = re.split(r'(?i)\s*(?:,|\bsinh\b|\bCMND\b|\bCCCD\b)', value)[0]
        
    if field_name not in ctx.colon_allowed_fields:
        value = re.sub(r'^:\s*', "", value)
        
    # Xóa các ký tự thừa ở đầu và cuối chuỗi (bao gồm cả dấu gạch ngang còn sót lại do cắt chữ sinh)
    value = re.sub(r'^[\s:;\-!,\.]+', "", value)
    value = re.sub(r'[\s:;\-!,\.]+$', "", value)
    value = re.sub(r'(?<!\w)!(?!\w)', "", value)
    
    # 2. Tinh chỉnh Regex khoảng trắng để bảo vệ Số tiền
    # Chỉ thêm khoảng trắng (space) sau dấu câu NẾU ký tự liền sau nó KHÔNG phải là một chữ số (d) và KHÔNG phải khoảng trắng (s)
    # Nhờ vậy: "khóm5,phường1" -> "khóm5, phường1" nhưng "2.500.000" vẫn được giữ nguyên.
    value = re.sub(r'([,.;:])(?=[^\s\d])', r'\1 ', value)
    
    if field_name in ["ten_loai_van_ban", "ten_co_quan_to_chuc", "toi_danh"]:
        value = re.sub(r'([^\W\d_])\s*,\s*([^\W\d_])', r'\1 \2', value, flags=re.UNICODE)
        
    # [FIX]: Tách chữ và số bị dính liền đối với các từ khóa địa chỉ (VD: khóm5 -> khóm 5, phường1 -> phường 1)
    if field_name == "dia_chi":
        value = re.sub(r'(?i)\b(khóm|phường|quận|tổ|ấp|đường|lô|tầng|số)(\d+)\b', r'\1 \2', value)

    return re.sub(r'\s{2,}', ' ', value).strip()


# ==========================================
# HOẠT ĐỘNG ROUTING PHÂN LOẠI VĂN BẢN (ROUTER)
# ==========================================
def route_template(text_lines: List[Dict[str, Any]], templates: List[Dict[str, Any]], ctx: ExtractionEngineContext = None) -> Tuple[str, float]:
    full_text_upper = " ".join(x.get("upper_text") or str(x.get("text", "")).upper() for x in text_lines)
    cache_key = hashlib.md5(full_text_upper.encode("utf-8")).hexdigest()
    
    cached = _ROUTE_CACHE.get(cache_key)
    if cached is not None: return cached
        
    best_tpl_id = "unknown"
    best_conf = 0.0
    # [FIX-F] routing_target_score từ config thông qua ctx, không hardcode 7.0
    target_score = ctx.routing_target_score if ctx is not None else 7.0

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
        
        conf = total_score / target_score if total_score > 0 else 0.0
        
        if conf > best_conf:
            best_conf = conf; best_tpl_id = tpl_id

    result = ("unknown", 0.0) if best_conf == 0.0 else (best_tpl_id, round(min(1.0, best_conf), 3))
    _ROUTE_CACHE.set(cache_key, result)
    return result


# ==========================================
# TRÍCH XUẤT CÁC TRƯỜNG CỐ ĐỊNH (FIXED FIELDS)
# ==========================================
def _collect_ngay_thang_nam_candidates(ctx: ExtractionEngineContext, text_string: str, line_idx: int, text_lines: List[Dict[str, Any]], bbox: List, line_conf: float) -> List[Dict[str, Any]]:
    candidates = []
    lower = text_string.lower()

    if ctx.date_deadline_noise_re.search(lower): return candidates

    def _append(raw: str, iso: str, method: str, score: float) -> None:
        # [FIX]: Chặn đứng ngay lập tức các chuỗi ngày tháng chứa ký tự đại diện hoặc bị khuyết
        if "X" in iso or "." in iso or iso.startswith("-") or iso.endswith("-"): 
            return
            
        candidates.append({
            "field_name": "ngay_thang_nam", "raw_value": raw, "value": iso,
            "confidence": round(min(1.0, max(0.0, score)), 3), "bbox": bbox, "method": method, "_score": score,
        })

    is_signature = bool(ctx.date_signature_re.search(text_string))

    # [FIX-J] date_context_window từ config, không hardcode 4/5
    win = ctx.date_context_window
    start_w, end_w = max(0, line_idx - win), min(len(text_lines), line_idx + win + 1)
    near_so = False
    for i in range(start_w, end_w):
        if ctx.so_line_re.search(text_lines[i].get("text", "")): near_so = True; break

    # [FIX-I] header_line_limit từ config
    header_region = line_idx < ctx.header_line_limit

    # Phân tách qua các Regex động giữ dấu chấm lửng
    flex_match = re.search(r"ngày\s*(\d{1,2}|\.{2,}|[xX]{1,2})\s*tháng\s*(\d{1,2}|\.{2,}|[xX]{1,2})\s*năm\s*(\d{4}|\.{2,}|[xX]{2,4})", text_string, re.IGNORECASE)
    if flex_match:
        d, m, y = flex_match.groups()
        score = line_conf + 0.30
        if is_signature: score += 0.35
        elif near_so: score += 0.22
        elif header_region: score += 0.12
        if ctx.date_body_noise_re.search(lower): score -= 0.15
        iso_val = f"{y}-{f'{int(m):02d}' if m.isdigit() else m}-{f'{int(d):02d}' if d.isdigit() else d}"
        _append(flex_match.group(0), iso_val, "regex_flex", score)

    # [FIX-P1-A] Dùng ctx.word_date_re đã compile sẵn, ctx.word_to_num đã load sẵn
    word_match = ctx.word_date_re.search(text_string)
    if word_match:
        d_str, m_str, y_str = word_match.groups()
        d_val = ctx.word_to_num.get(d_str.lower(), 0)
        m_val = ctx.word_to_num.get(m_str.lower(), 0)
        if d_val and m_val:
            score = line_conf + 0.28
            if is_signature: score += 0.38
            elif near_so: score += 0.25
            elif header_region: score += 0.15
            _append(word_match.group(0), f"{y_str}-{m_val:02d}-{d_val:02d}", "regex_word_date", score)

    slash_match = re.search(r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b", text_string)
    if slash_match and f"{ctx.so_prefix_re}" not in text_string:
        d, m, y = slash_match.groups()
        score = line_conf - 0.05
        if is_signature: score += 0.30
        elif near_so: score += 0.20
        elif header_region: score += 0.10
        else: score -= 0.25
        if ctx.date_body_noise_re.search(lower): score -= 0.40
        if score >= line_conf - 0.20:
            _append(slash_match.group(0), f"{y}-{int(m):02d}-{int(d):02d}", "regex", score)

    short_noise_match = re.search(r"(?i)\bNgày\s*[:\-]?\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.]([A-Za-z0-9]+)\b", text_string)
    if short_noise_match and not slash_match:
        d, m, y_str = short_noise_match.groups()
        score = line_conf + 0.1
        if is_signature: score += 0.2
        y_val = y_str if y_str.isdigit() and len(y_str) == 4 else "XXXX"
        _append(short_noise_match.group(0), f"{y_val}-{int(m):02d}-{int(d):02d}", "regex_short_noise", score)

    return candidates


def _extract_structural(ctx: ExtractionEngineContext, text_string: str, bbox: List, line_conf: float, confusions: Dict[str, str]) -> List[Dict[str, Any]]:
    results = []
    match = ctx.so_van_ban_full_re.search(text_string)
    if match:
        raw_full = match.group(1)
        fixed_full = apply_ocr_correction(raw_full, ctx, confusions)
        parts = fixed_full.split("/")
        if len(parts) >= 2:
            if parts[1].isdigit(): so_qd = f"{parts[0]}/{parts[1]}"; ky_hieu_idx = 2
            else: so_qd = parts[0]; ky_hieu_idx = 1
        else: so_qd = parts[0]; ky_hieu_idx = -1
            
        results.append({
            "field_name": "so_quyet_dinh", "raw_value": raw_full, "value": post_process_value(ctx, so_qd),
            "confidence": line_conf, "bbox": bbox, "method": "regex",
        })
        if ky_hieu_idx != -1 and len(parts) > ky_hieu_idx:
            results.append({
                "field_name": "ky_hieu", "raw_value": raw_full, "value": post_process_value(ctx, "/".join(parts[ky_hieu_idx:])),
                "confidence": line_conf, "bbox": bbox, "method": "regex",
            })
    else:
        simple = ctx.so_van_ban_simple_re.search(text_string)
        if simple:
            results.append({
                "field_name": "so_quyet_dinh", "raw_value": simple.group(1), "value": post_process_value(ctx, simple.group(1)),
                "confidence": line_conf - 0.05, "bbox": bbox, "method": "regex_simple",
            })
    return results


def _extract_ten_co_quan(ctx: ExtractionEngineContext, text_lines: List[Dict[str, Any]], confusions: Dict[str, str]) -> Optional[Dict[str, Any]]:
    # [FIX-P1-E] dùng ctx.co_quan_re đã compile sẵn trong context
    # [FIX-P1-C] dùng ctx.header_noise_re, ctx.cong_hoa_re đã compile sẵn
    cong_hoa_idx = -1
    
    for i, line in enumerate(text_lines):
        if ctx.cong_hoa_re.search(line.get("text", "").upper()): cong_hoa_idx = i; break

    def _try_match(lines, confidence_default):
        for line in lines:
            raw = line.get("text", "").strip()
            # [FIX-P1-C] ưu tiên corrected_text precomputed
            text = line.get("corrected_text") or apply_ocr_correction(raw, ctx, {})
            if '\n' in text: text = text.split('\n')[0].strip()
            text = ctx.cong_hoa_re.split(text)[0].strip()
            if re.match(r"(?i)^\s*(Số|SO|S[Ôô6Oo])\s*[:\.]", text): continue
            if re.search(r"\d+\/[A-ZĐa-zđ0-9\-]+", text): continue
            if ctx.van_ban_title_re.match(text.upper()): continue
            if not text or ctx.header_noise_re.search(text) or len(text.split()) > 15: continue

            if ctx.co_quan_re.match(text.upper()):
                return {
                    "field_name": "ten_co_quan_to_chuc", "raw_value": raw,
                    "value": post_process_value(ctx, normalize_name_strip(text), "ten_co_quan_to_chuc"),
                    "confidence": confidence_default, "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
                }
        return None

    # [FIX-K] cong_hoa_search_window từ config, không hardcode 15
    start_idx = max(0, cong_hoa_idx - 5) if cong_hoa_idx != -1 else 0
    end_idx = (cong_hoa_idx + ctx.cong_hoa_search_window) if cong_hoa_idx != -1 else min(20, len(text_lines))

    result = _try_match(text_lines[start_idx: end_idx], 0.9)
    if result: return result

    # [FIX-P1-D] dùng ctx.strong_header_re đã compile sẵn
    for line in text_lines[:8]:
        raw = line.get("text", "").strip()
        text = line.get("corrected_text") or apply_ocr_correction(raw, ctx, {})
        if "\n" in text: text = text.split("\n")[0].strip()
        text = ctx.cong_hoa_re.split(text)[0].strip()
        if re.match(r"(?i)^\s*(Số|SO|S[Ôô6Oo])\s*[:\.]", text): continue
        if re.search(r"\d+\/[A-ZĐa-zđ0-9\-]+", text): continue
        if not text or ctx.header_noise_re.search(text): continue
        if ctx.strong_header_re.search(text):
            return {
                "field_name": "ten_co_quan_to_chuc", "raw_value": raw,
                "value": post_process_value(ctx, normalize_name_strip(text), "ten_co_quan_to_chuc"),
                "confidence": 0.86, "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }

    # [FIX-L] spacy_max_chars từ config, không hardcode 500
    all_text = " ".join(
        (line.get("corrected_text") or apply_ocr_correction(line.get("text", ""), ctx, {}))
        for line in text_lines[:10]
    )
    if all_text.strip():
        doc = nlp(all_text[:ctx.spacy_max_chars])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "FAC"):
                ent_text = ent.text.strip()
                if len(ent_text) > 5 and not re.search(r"CỘNG HÒA|ĐỘC LẬP|TỰ DO|HẠNH PHÚC|VIỆT NAM", ent_text, re.IGNORECASE):
                    return {
                        "field_name": "ten_co_quan_to_chuc", "value": post_process_value(ctx, normalize_name_strip(ent_text), "ten_co_quan_to_chuc"),
                        "confidence": 0.65, "bbox": [0, 0, 0, 0], "method": "spacy_ner",
                    }
    return None


def _extract_ten_loai_van_ban(ctx: ExtractionEngineContext, text_lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # [FIX-P1-B] dùng ctx.title_keyword_re đã compile sẵn, không hardcode danh sách
    cong_hoa_idx = -1
    so_idx = len(text_lines)
    for i, line in enumerate(text_lines):
        txt = line.get("text", "").upper()
        if "CỘNG HÒA" in txt and cong_hoa_idx == -1: cong_hoa_idx = i
        if re.search(r"\bSỐ\s*:", txt) and i > cong_hoa_idx: so_idx = i; break

    search_end = min(len(text_lines), so_idx + 5)
    search_start = max(0, cong_hoa_idx) if cong_hoa_idx != -1 else 0
    search_lines = text_lines[search_start:search_end] if cong_hoa_idx != -1 else text_lines[:min(ctx.header_line_limit + 5, len(text_lines))]
    for line in search_lines:
        raw = line.get("text", "").strip()
        text = line.get("corrected_text") or apply_ocr_correction(raw, ctx, {})
        if not text: continue
        if ctx.title_keyword_re.search(text.strip()) or ctx.van_ban_title_re.match(text.upper()):
            return {
                "field_name": "ten_loai_van_ban", "raw_value": raw, "value": post_process_value(ctx, text.strip(), "ten_loai_van_ban"),
                "confidence": round(float(line.get("confidence", 0.92)), 3), "bbox": line.get("bbox", [0, 0, 0, 0]), "method": "pattern_match",
            }
    return None


def extract_fixed_fields(ctx: ExtractionEngineContext, text_lines: List[Dict[str, Any]], confusions: Dict[str, str] = None) -> List[Dict[str, Any]]:
    confusions = confusions or {}
    results = []
    # [FIX-I] header_line_limit từ config, không hardcode 15
    has_strict_header_so = any(re.search(r"(?i)^Số\s*[:\.]", l.get("text", "").strip()) for l in text_lines[:ctx.header_line_limit])

    for idx, line in enumerate(text_lines):
        raw_text = line.get("text", "").strip()
        text_string = line.get("corrected_text") or apply_ocr_correction(raw_text, ctx, confusions)
        bbox = line.get("bbox", [0, 0, 0, 0])
        line_conf = float(line.get("confidence", 0.0))

        structural_results = _extract_structural(ctx, text_string, bbox, line_conf, confusions)
        for r in structural_results:
            if re.search(r"^(Căn cứ|Theo|Tại|Xét|Về việc|Điều.*của|Thông báo số|Thông tư số|Nghị định số|ban hành)\b", raw_text, re.IGNORECASE):
                r["confidence"] = 0.0
            elif idx > ctx.header_line_limit:
                r["confidence"] = min(1.0, r["confidence"] + 0.2) if bool(ctx.so_line_re.search(raw_text)) else max(0.0, r["confidence"] - 0.2)
            elif idx < ctx.so_line_boundary:
                if bool(ctx.so_line_re.search(raw_text)): r["confidence"] = min(1.0, r["confidence"] + 0.2)
                elif has_strict_header_so: r["confidence"] = max(0.0, r["confidence"] - 0.4)
                else: r["confidence"] = min(1.0, r["confidence"] + 0.1)
            results.append(r)

    best_date = _pick_best_ngay_thang_nam(ctx, text_lines)
    if best_date: results.append(best_date)

    co_quan = _extract_ten_co_quan(ctx, text_lines, confusions)
    if co_quan: results.append(co_quan)

    ten_loai = _extract_ten_loai_van_ban(ctx, text_lines)
    if ten_loai: results.append(ten_loai)

    # Dedup logic dựa trên độ tin cậy
    best_unique = {}
    for f in results:
        name = f["field_name"]
        if name not in best_unique or f["confidence"] > best_unique[name]["confidence"]:
            best_unique[name] = f
    return list(best_unique.values())


def _pick_best_ngay_thang_nam(ctx: ExtractionEngineContext, text_lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    all_candidates = []
    for idx, line in enumerate(text_lines):
        text_string = line.get("text", "").strip()
        if not text_string: continue
        bbox = line.get("bbox", [0, 0, 0, 0])
        line_conf = float(line.get("confidence", 0.0))
        all_candidates.extend(_collect_ngay_thang_nam_candidates(ctx, text_string, idx, text_lines, bbox, line_conf))
    
    if not all_candidates:
        full_text_concat = " ".join(l.get("text", "") for l in text_lines)
        date_match = re.search(r'(?i)ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', full_text_concat)
        if date_match:
            d, m, y = date_match.groups()
            return {"field_name": "ngay_thang_nam", "raw_value": date_match.group(0), "value": f"{y}-{int(m):02d}-{int(d):02d}", "confidence": 0.45, "bbox": [0, 0, 0, 0], "method": "regex_fallback"}
        return None
    return max(all_candidates, key=lambda c: c["_score"])


# ==========================================
# TRÍCH XUẤT CÁC TRƯỜNG BIẾN ĐỘNG (DYNAMIC FIELDS)
# ==========================================
def _truncate_at_other_labels(value: str, current_field: str, ctx: ExtractionEngineContext) -> str:
    if not value: return value
    cut_points = []
    for fname, pat in ctx.compiled_labels_cache:
        if fname == current_field: continue
        m = pat.search(value)
        if m and m.start() > 0: cut_points.append(m.start())
    if cut_points: return value[: min(cut_points)].strip()
    return value


def _extract_config_driven(text_string: str, bbox: List, line_conf: float, ctx: ExtractionEngineContext, confusions: Dict[str, str], all_text_lines: List[Dict[str, Any]] = None, current_idx: int = 0, allowed_fields: frozenset = None) -> List[Dict[str, Any]]:
    results = []
    for field_name, rule in ctx.fields_cfg.items():
        if field_name in ctx.fixed_header_fields or (allowed_fields is not None and field_name not in allowed_fields): continue
        
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
                        value = _truncate_at_other_labels(value, field_name, ctx)
                        break
                        
        if not value and strategy == "next_line" and all_text_lines and current_idx + 1 < len(all_text_lines):
            value = re.sub(r"^[:\-\s]+", "", all_text_lines[current_idx + 1].get("text", "").strip()).strip()

        if not value and strategy == "next_n_lines" and all_text_lines:
            n = rule.get("next_n_lines", 2)
            parts = []
            for l_rx in label_regexes:
                m = re.search(l_rx, text_string, re.IGNORECASE)
                if m and text_string[m.end():].strip():
                    parts.append(re.sub(r"^[:\-\s]+", "", text_string[m.end():].strip()).strip())
                    break
            for j in range(current_idx + 1, min(current_idx + 1 + n, len(all_text_lines))):
                t = all_text_lines[j].get("text", "").strip()
                if t:
                    if ctx.section_break_re.search(t): break
                    
                    # [FIX] Cắt rác đột ngột: Chặn các dòng bắt đầu bằng Số thứ tự, Bản án, Ngày tháng...
                    if re.match(r"(?i)^(\d+\s*[\/\.]|bản án|quyết định|ngày[\s\.]+tháng)", t): break

                    other_hit = False
                    for fn, pat in ctx.compiled_labels_cache:
                        if fn != field_name and pat.search(t): other_hit = True; break
                    if geopolitical_break := other_hit: break
                    parts.append(re.sub(r"^[:\-\s]+", "", t).strip())
            value = " ".join(parts) if parts else None
            if value: value = _truncate_at_other_labels(value, field_name, ctx)

        if not value and strategy == "until_blank" and all_text_lines:
            parts = []
            for j in range(current_idx + 1, len(all_text_lines)):
                t = all_text_lines[j].get("text", "").strip()
                if not t: break
                if any(pat.search(t) for _, pat in ctx.compiled_labels_cache): break
                parts.append(re.sub(r"^[:\-\s]+", "", t).strip())
            value = " ".join(parts) if parts else None
                    
        if not value: continue
        raw_val = value
        value = apply_ocr_correction(value, ctx, confusions if field_name in ctx.numeric_fields else {}, field_name=field_name)
        value = post_process_value(ctx, value, field_name)
        conf = compute_field_confidence(line_conf, value, value, rule, text_string)
        
        results.append({"field_name": field_name, "raw_value": raw_val, "value": value, "confidence": conf, "bbox": bbox, "method": "regex"})
    return results


def extract_dynamic_fields(
    ctx: ExtractionEngineContext, full_text: str, required_fields: List[str], extracted_fixed_fields: List[Dict[str, Any]], 
    text_lines: List[Dict], confusions: Dict[str, str] = None, zero_shot_configs: List[Dict[str, str]] = None, 
    gliner_label_map: Dict[str, str] = None, bbox_index: Dict[str, List] = None
) -> List[Dict[str, Any]]:
    confusions = confusions or {}
    results = []
    existing_fields = {f["field_name"] for f in extracted_fixed_fields}
    forbidden_gliner_values = {str(f.get("value", "")).lower().strip() for f in extracted_fixed_fields if len(str(f.get("value", ""))) > 3}

    all_possible_fields = list(ctx.fields_cfg.keys()) + [z["name"] for z in zero_shot_configs or []]
    missing_fields = [f for f in all_possible_fields if f not in existing_fields and f not in {"so_quyet_dinh", "ten_co_quan_to_chuc", "ten_loai_van_ban"}]
    if not missing_fields: return results

    # 1. Quét theo Regex Label động trước
    dynamic_allowed = frozenset(missing_fields)
    for idx, line in enumerate(text_lines):
        raw_text = line.get("text", "").strip()
        text_string = line.get("corrected_text") or apply_ocr_correction(raw_text, ctx, confusions)
        label_hits = _extract_config_driven(text_string, line.get("bbox", [0,0,0,0]), float(line.get("confidence", 0.0)), ctx, confusions, all_text_lines=text_lines, current_idx=idx, allowed_fields=dynamic_allowed)
        for hit in label_hits:
            if hit["field_name"] not in {r["field_name"] for r in results}:
                hit["method"] = hit.get("method", "regex") + "_label"
                results.append(hit)

    missing_fields = [f for f in missing_fields if f not in {r["field_name"] for r in results}]
    if not missing_fields: return results

    # 2. Xây nhãn GLiNER động từ cấu hình trường dữ liệu
    target_labels = []
    for field in missing_fields:
        rule = ctx.fields_cfg.get(field, {})
        gliner_labels = rule.get("gliner_labels", [])
        if gliner_labels: target_labels.extend(gliner_labels)
        else:
            z_config = next((z for z in zero_shot_configs or [] if z["name"] == field), None)
            if z_config:
                target_labels.append(z_config["description"])
                gliner_label_map[z_config["description"].lower().strip()] = field
    
    target_labels = list(set(target_labels))
    # [FIX-C] dùng corrected_text đã precompute từ text_lines thay vì gọi lại apply_ocr_correction(full_text)
    clean_text = "\n".join(l.get("corrected_text") or l.get("text", "") for l in text_lines)
    if target_labels and len(clean_text) >= 50:
        # Thực hiện chia cụm an toàn thông qua hàm tiện ích
        lines = clean_text.split('\n')
        chunks, current_chunk, current_len = [], [], 0
        for line in lines:
            words = line.split()
            if not words: continue
            if current_len + len(words) > 250 and current_chunk:
                chunks.append(" ".join(current_chunk)); current_chunk = words; current_len = len(words)
            else:
                current_chunk.extend(words); current_len += len(words)
        if current_chunk: chunks.append(" ".join(current_chunk))

        try:
            with torch.inference_mode():
                # [FIX-G] inference_threshold từ config, không hardcode 0.25
                batch_predictions = gliner_model.inference(chunks, target_labels, flat_ner=True, threshold=ctx.gliner_inference_threshold)
                best_gliner = {}
                for predictions in batch_predictions:
                    for p in predictions:
                        lbl = p["label"].lower().strip()
                        val = p["text"].strip()
                        fname = gliner_label_map.get(lbl, lbl)
                        
                        if fname not in missing_fields or val.lower() in forbidden_gliner_values or len(val) < 2: continue
                        
                        thresh = float(ctx.fields_cfg.get(fname, {}).get("gliner_threshold", 0.42))
                        score = float(p["score"])
                        if score < thresh: continue
                        
                        # SỬA LỖI: Tiền xử lý và chặn đứng nếu chuỗi trả về bị rỗng
                        processed_val = post_process_value(ctx, val, fname)
                        if not processed_val: continue
                        
                        # Định vị Bounding Box tối ưu thông qua bbox_index O(1)
                        # [FIX-D] loại bỏ double-check tautology `if bbox==[0,0,0,0] and val in bbox_index`
                        bbox = bbox_index.get(val, [0, 0, 0, 0])
                        
                        if fname not in best_gliner or score > best_gliner[fname]["confidence"]:
                            best_gliner[fname] = {
                                "field_name": fname, 
                                "raw_value": val, 
                                "value": processed_val, 
                                "confidence": round(score, 3), 
                                "bbox": bbox, 
                                "method": "gliner_ner"
                            }
                results.extend(list(best_gliner.values()))
        except Exception:
            pass

    return results


# ==========================================
# CƠ CHẾ VALIDATION VÀ KIỂM ĐỊNH HOÀN TOÀN ĐỘNG
# ==========================================
def validate_field_value_dynamic(field_name: str, value: str, ctx: ExtractionEngineContext, full_text: str = "") -> bool:
    if not value or value.strip() == "": return False
    rule = ctx.fields_cfg.get(field_name, {})
    v_rules = rule.get("validation", {})
    if not v_rules: return True

    # 1. Kiểm tra ký tự giữ chỗ đại diện (Placeholders)
    if "allow_placeholders" in v_rules:
        if any(p in value.upper() for p in v_rules["allow_placeholders"]): return True

    # 2. Kiểm tra độ dài từ (Min words)
    if "min_words" in v_rules and len(value.split()) < v_rules["min_words"]: return False

    # 3. Kiểm tra độ dài chuỗi (Min length)
    if "min_length" in v_rules and len(value) < v_rules["min_length"]: return False

    # 4. Kiểm tra qua định dạng Regex đầu ra bắt buộc
    if "regex" in v_rules:
        val_clean = value.replace(" ", "")
        if not bool(re.match(v_rules["regex"], val_clean)):
            if "fallback_year_regex" in v_rules and bool(re.search(v_rules["fallback_year_regex"], value)): return True
            return False

    # 5. Khớp các điều kiện Anti-regex loại trừ (Như loại MST lọt vào sổ BHXH)
    if "anti_regexes" in v_rules and full_text:
        val_escaped = re.escape(value.replace(" ", ""))
        for anti in v_rules["anti_regexes"]:
            if re.search(f"{anti}[^\n]{{0,30}}{val_escaped}", full_text, re.IGNORECASE) or \
               re.search(f"{val_escaped}[^\n]{{0,30}}{anti}", full_text, re.IGNORECASE):
                return False

    return True


def _is_safe_text_correction_dynamic(field_name: str, before: str, after: str, ctx: ExtractionEngineContext) -> bool:
    if not after: return False
    b_n, a_n = before.strip(), after.strip()
    if not b_n: return True

    rule = ctx.fields_cfg.get(field_name, {})
    safe_rules = rule.get("correction_safe_rules", {})
    if not safe_rules: return True
    if safe_rules.get("skip_model"): return False

    sim = _similarity_ratio(b_n.lower(), a_n.lower())
    if "min_similarity" in safe_rules and sim < safe_rules["min_similarity"]: return False

    # Khớp cặp từ khóa bắt buộc phải giữ lại sau hiệu chỉnh
    for req_b, req_a in safe_rules.get("required_phrase_pairs", []):
        if req_b in b_n.lower() and req_a not in a_n.lower(): return False

    # Kiểm tra mất mát thông tin số nhạy cảm (SỬA LỖI TẠI ĐÂY)
    if safe_rules.get("check_digit_loss"):
        b_digits = re.findall(r"\d", b_n)
        a_digits = re.findall(r"\d", a_n)
        # Bắt buộc số lượng chữ số sau khi sửa phải LỚN HƠN HOẶC BẰNG ban đầu
        if b_digits and len(a_digits) < len(b_digits): 
            return False

    return True


# ==========================================
# THỰC THI CHƯƠNG TRÌNH TRÊN FILE ĐƠN (CORE)
# ==========================================
def _process_single_file(p: Path, ctx: ExtractionEngineContext, out_schema: Dict, output_dir: Path, logger: logging.Logger, gliner_label_map: Dict[str, str]) -> Dict[str, Any]:
    file_start_time = time.time()
    stat = {"file": p.name, "status": "error", "fields": 0, "elapsed": 0.0, "doc_type": "unknown"}

    try:
        in_json = load_json(str(p))
        payload = in_json.get("payload", {})
        full_text = payload.get("text", "")
        blocks = payload.get("blocks") or []

        text_lines = []
        last_bbox = [52, 30, 620, 50]
        min_blocks = ctx.cfg.get("extraction", {}).get("min_blocks_threshold", 6)
        
        if len(blocks) < min_blocks:
            for raw_line in full_text.split("\n"):
                if not raw_line.strip(): continue
                interpolated = [last_bbox[0], last_bbox[3] + 4, last_bbox[2], last_bbox[3] + 24]
                # [FIX] interpolated_conf từ config, không hardcode 0.8
                text_lines.append({"text": raw_line.strip(), "bbox": interpolated, "confidence": ctx.interpolated_conf})
                last_bbox = interpolated
        else:
            # [FIX-G] dùng "".join(split()) thay re.sub(r'\s+','') — nhanh hơn cho whitespace
            block_lookup = {"".join(b.get("text", "").replace("\n", " ").split()).lower(): b for b in blocks if b.get("text")}
            for raw_line in full_text.split("\n"):
                if not raw_line.strip(): continue
                clean_key = "".join(raw_line.split()).lower()
                matched_block = block_lookup.pop(clean_key, None)
                if matched_block:
                    raw_conf = float(matched_block.get("confidence", 0.85))
                    conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
                    text_lines.append({"text": matched_block.get("text", ""), "bbox": matched_block.get("bbox", [0,0,0,0]), "confidence": conf})
                else:
                    interpolated = [last_bbox[0], last_bbox[3] + 4, last_bbox[2], last_bbox[3] + 24]
                    text_lines.append({"text": raw_line.strip(), "bbox": interpolated, "confidence": ctx.interpolated_conf})
                    last_bbox = interpolated

        # [FIX-P2-H] Precompute upper_text một lần cho routing — không tính lại trong route_template
        for line in text_lines:
            if "upper_text" not in line:
                line["upper_text"] = line.get("text", "").upper()

        # Xử lý Router và Fixed fields
        templates = ctx.cfg.get("router", {}).get("templates", [])
        # [FIX-F] truyền ctx vào route_template để dùng routing_target_score
        doc_type, routing_conf = route_template(text_lines, templates, ctx)
        matched_tpl = next((t for t in templates if t["template_id"] == doc_type), {})
        required_fields = matched_tpl.get("required_fields", ["so_quyet_dinh", "ky_hieu", "ngay_thang_nam", "ten_co_quan_to_chuc", "ten_loai_van_ban"])

        # Precompute OCR cho toàn văn bản giảm 50% thời gian xử lý chuỗi lặp
        for line in text_lines:
            if "corrected_text" not in line:
                line["corrected_text"] = apply_ocr_correction(line.get("text", ""), ctx, {})

        fixed_fields = extract_fixed_fields(ctx, text_lines, ctx.auto_corr.get("ocr_confusions", {}))
        
        # Tạo chỉ mục Bounding Box O(1) phục vụ phục hồi tọa độ hình học
        bbox_idx = {l.get("text", "").strip(): l.get("bbox", [0,0,0,0]) for l in text_lines if l.get("text")}
        
        dynamic_fields = extract_dynamic_fields(ctx, full_text, required_fields, fixed_fields, text_lines, ctx.auto_corr.get("ocr_confusions", {}), ctx.cfg.get("zero_shot_fields", []), gliner_label_map, bbox_idx)

        # Trộn chuẩn hóa dữ liệu động qua hàm Normalizers
        dynamic_fields_final = []
        for field in dynamic_fields:
            rule = ctx.fields_cfg.get(field["field_name"], {})
            norm = rule.get("normalizer")
            val = field["value"]
            if norm == "date_dmy_to_iso": val = normalize_date_dmy_to_iso(val) or val.strip()
            elif norm == "number_slash": val = normalize_number_generic(val) or val.strip()
            elif norm == "name_strip": val = normalize_name_strip(val)
            elif norm == "bhxh_fix_confusions": val = normalize_bhxh(val, ctx.auto_corr.get("ocr_confusions", {})) or val.strip()
            field["value"] = val
            dynamic_fields_final.append(field)

        # Gộp nhóm và thực hiện cơ chế Validation động đa tầng
        merged_extracted = {f["field_name"]: f for f in fixed_fields + dynamic_fields_final if f["field_name"] in required_fields}
        validated_fields = {k: v for k, v in merged_extracted.items() if validate_field_value_dynamic(k, str(v["value"]), ctx, full_text)}

        # Áp dụng AI Sửa lỗi chính tả hàng loạt tiếng Việt (Vietnamese Correction Mode Batch)
        changes_log = []
        if ctx.auto_corr.get("enabled"):
            targets = []
            for fname, field in validated_fields.items():
                rule = ctx.fields_cfg.get(fname, {})
                if rule.get("correction") and not rule.get("correction_safe_rules", {}).get("skip_model"):
                    targets.append(field)
                elif fname == "ten_co_quan_to_chuc":
                    # [FIX]: Bỏ regex xóa ký tự (nguyên nhân gây mất chữ có dấu), chỉ normalize khoảng trắng
                    field["value"] = re.sub(r'\s{2,}', ' ', str(field["value"])).upper().strip()

            if targets:
                t_vals = [f["value"] for f in targets]
                t_fnames = [f["field_name"] for f in targets]
                corr_vals, sub_m = apply_vietnamese_correction_batch(
                    ctx, t_vals, field_names=t_fnames, confusions={}, return_sub_methods=True
                )
                
                for field, after, method in zip(targets, corr_vals, sub_m):
                    before = str(field["value"])
                    if _is_safe_text_correction_dynamic(field["field_name"], before, after, ctx):
                        field["value"] = after
                        if method and before.strip() != str(after).strip():
                            changes_log.append({"original": before, "corrected": after, "field": field["field_name"], "method": "pipeline_correction", "sub_method": method})

        final_fixed = [{k: v for k, v in f.items() if k not in {"raw_value"}} for f in validated_fields.values() if f["field_name"] in ctx.fixed_header_fields]
        final_dynamic = [{k: v for k, v in f.items() if k not in {"raw_value"}} for f in validated_fields.values() if f["field_name"] not in ctx.fixed_header_fields]

        # Logic phán đoán nguyên nhân thiếu trường (Missing Check) HOÀN TOÀN TỰ ĐỘNG KHÔNG IF/ELSE
        missing_required = []
        for f in set(required_fields).union({"so_quyet_dinh", "ky_hieu", "ngay_thang_nam", "ten_co_quan_to_chuc", "ten_loai_van_ban"}):
            if f not in validated_fields:
                rule = ctx.fields_cfg.get(f, {})
                p_rx = rule.get("presence_check_regex", "")
                reason = "extraction_failed"
                if p_rx and not re.search(p_rx, full_text, re.IGNORECASE):
                    reason = "not_in_source"
                elif f == "ten_loai_van_ban" and not ctx.van_ban_title_re.search(full_text.upper()):
                    reason = "not_in_source"
                missing_required.append({"field": f, "reason": reason})

        coverage = round(len(validated_fields.keys() & set(required_fields)) / max(1, len(required_fields)), 2)

        out_json = {
            "request_id": in_json.get("request_id", str(uuid.uuid4())), "document_id": in_json.get("document_id", "unknown"),
            "timestamp": now_iso_utc(), "status": "success", "error": None,
            "payload": {
                "document_type": doc_type, "routing_confidence": routing_conf, "field_coverage": coverage, "missing_required_fields": missing_required,
                "extracted_fields": {"fixed": final_fixed, "dynamic": final_dynamic},
                "correction_log": {"enabled": ctx.auto_corr.get("enabled", False), "changes": changes_log},
            },
        }

        validate(instance=out_json, schema=out_schema)
        (output_dir / f"{p.stem}_extracted.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

        stat.update({"status": "success", "fields": len(validated_fields), "elapsed": time.time() - file_start_time, "doc_type": doc_type})
        logger.info(f"✅ {p.name} -> {doc_type} | Fields: {len(validated_fields)} | Time: {stat['elapsed']:.2f}s")
    except Exception as e:
        stat["elapsed"] = time.time() - file_start_time
        logger.error(f"❌ Lỗi xử lý {p.name}: {e}")

    return stat


# ==========================================
# ĐIỀU PHỐI BATCH THƯ MỤC SONG SONG (MULTIPROCESSING)
# ==========================================
def process_folder(input_dir: Path, output_dir: Path, config_path: str, schema_path: str, config_schema_path: str = None, dry_run: bool = False, max_workers: int = None) -> None:
    logger = setup_logger()
    cfg = load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    # Đọc thiết lập luồng song song từ config
    if max_workers is None:
        set_cap = cfg.get("engine_settings", {}).get("max_workers_cap", 4)
        set_ratio = cfg.get("engine_settings", {}).get("max_workers_ratio", 0.5)
        max_workers = min(set_cap, max(1, int((os.cpu_count() or 2) * set_ratio)))

    # Đóng gói và dựng Context Động toàn cục duy nhất một lần
    global INITIAL_ENGINE_CONTEXT
    INITIAL_ENGINE_CONTEXT = ExtractionEngineContext(cfg)
    
    # Sinh bản đồ nhãn GLiNER động từ tệp YAML cấu hình
    gliner_label_map = {}
    for fname, rule in cfg.get("fields", {}).items():
        for gl_lbl in rule.get("gliner_labels", []):
            gliner_label_map[gl_lbl.lower().strip()] = fname

    json_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".json")
    if dry_run:
        logger.info(f"🔍 Dry-run: Tìm thấy {len(json_files)} file JSON cần xử lý.")
        return

    total_start_time = time.time()
    all_stats = []

    if max_workers > 1:
        # NOTE [P3]: ThreadPoolExecutor an toàn với CUDA context và model state dùng chung.
        # ProcessPoolExecutor sẽ nhanh hơn cho CPU-only nhưng yêu cầu serialize model — xem xét
        # khi tách GLiNER thành service riêng (queue-based batch inference).
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_file, p, INITIAL_ENGINE_CONTEXT, out_schema, output_dir, logger, gliner_label_map
                ): p for p in json_files
            }
            for fut in concurrent.futures.as_completed(futures): 
                all_stats.append(fut.result())
    else:
        for p in json_files:
            all_stats.append(_process_single_file(p, INITIAL_ENGINE_CONTEXT, out_schema, output_dir, logger, gliner_label_map))

    # Tổng hợp thống kê hiệu năng vận hành hệ thống
    success_stats = [s for s in all_stats if s["status"] == "success"]
    logger.info(f"{'='*50}\n🎉 Hoàn thành xử lý batch: {len(success_stats)}/{len(all_stats)} tập tin trích xuất thành công.")
    logger.info(f"   Tổng thời gian vận hành hệ thống: {time.time() - total_start_time:.2f}s")