# German PDF Translation (Mac)

Pipeline per page (default):
1. OCR
2. Normalize German
3. Translate to English
4. Verify
5. Write outputs

## Setup

```bash
brew install poppler tesseract tesseract-lang
brew install --cask ollama
```

```bash
ollama pull glm-ocr:latest
ollama pull gemma3:4b
ollama pull qwen3:8b
```

Check OCR languages:

```bash
tesseract --list-langs | rg 'deu|frak'
```

## Run

```bash
cd ~/Desktop/German_To_English
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --force-ocr --ocr-engine glm-ocr --glm-ocr-model glm-ocr:latest \
  --ollama-normalize-model gemma3:4b \
  --ollama-translate-model qwen3:8b \
  --ollama-verify-model qwen3:8b
```

You will see progress like:
- `Stage OCR | Page X | i/N | Remaining R`
- `Stage NORMALIZE | Page X | i/N | Remaining R`
- `Stage TRANSLATE | Page X | i/N | Remaining R`
- `Stage VERIFY | Page X | i/N | Remaining R`

## Outputs

Written to `~/Desktop/German_To_English/output/`:

- `*.translated_english_final.html` (English-only final)
- `*.scholar_german_english.html` (German/English side-by-side)
- `*.review.html` (raw OCR + normalized German + English)
- `*.de.raw.txt`
- `*.de.normalized.txt`
- `*.en.txt`
- `*.pages.json`
- `*.work.json`
