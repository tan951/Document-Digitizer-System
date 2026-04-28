# Document Digitizer System

## Repository Structure

```
project-root/
├─ packages/
│  ├─ preprocess/          # Module 1: Image Preprocessing ✅ DONE
│  ├─ ocr/                 # Module 2: OCR / Text Recognition
│  ├─ extraction/          # Module 3: Field Extraction
│  └─ validation/          # Module 4: Schema validation & postprocess
│
├─ pipeline/               # end-to-end flow runner
│  ├─ run_pipeline.py
│  ├─ config/
│  └─ schemas/
│
├─ docs/
│  ├─ architecture.md
│  ├─ data_contracts.md
│  └─ decisions.md
│
├─ CHANGELOG.md
└─ README.md
```

## Module Ownership
- **preprocess**: image cleaning, binarization, deskew, blank detection  
- **ocr**: text recognition / layout OCR  
- **extraction**: key-value extraction, tables, doc QA  
- **validation**: schema validation, error normalization, output postprocess

---

## Next Steps
- Add Module 2: OCR
- Define unified output schema in `pipeline/schemas/`