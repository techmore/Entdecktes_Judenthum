# German PDF -> OCR, Normalized German, English Translation (Mac)

Use `/Users/seandolbec/Desktop/German_To_English/translate_pdf.py` for staged page-by-page processing.
Default behavior is the full pipeline in one run, processed page-by-page:
OCR(page 1) -> normalize(page 1) -> translate(page 1) -> verify(page 1), then page 2, etc.

Pipeline per page:

1. Extract text (`pdftotext`, OCR fallback with Tesseract)
2. Optional local normalization to readable modern German (Ollama)
3. Translation to English (Ollama local or OpenAI)
4. Optional local verification pass (Ollama)

Final deliverables generated automatically:

1. English-only, page-by-page output (`*.translated_english_final.html`)
2. German/English side-by-side output (`*.scholar_german_english.html`)
3. Full QA view with OCR/raw + normalized + English (`*.review.html`)

## Install requirements

```bash
brew install poppler tesseract tesseract-lang
```

Verify OCR languages:

```bash
tesseract --list-langs | rg 'deu|frak'
```

You want `deu` and ideally `deu_frak`.

## Local model recommendations (already on your machine)

- Normalize German: `gemma3:4b` (faster)
- Translate German -> English: `qwen3:8b`
- Verify/fix outputs: `glm-4.7-flash:latest` (stronger but slower)
- OCR alternative: `glm-ocr:latest` (via Ollama multimodal OCR)

## Optional: install GLM-OCR

```bash
ollama pull glm-ocr:latest
```

Then enable it with:

```bash
--ocr-engine glm-ocr --glm-ocr-model glm-ocr:latest
```

## Quick OCR-only check (no model translation)

```bash
cd /Users/seandolbec/Desktop/German_To_English
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --max-pages 3 --force-ocr --ocr-lang deu_frak+deu --skip-translation
```

## Default one-command run (recommended)

```bash
cd /Users/seandolbec/Desktop/German_To_English
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --force-ocr --ocr-engine glm-ocr --glm-ocr-model glm-ocr:latest \
  --ollama-normalize-model gemma3:4b \
  --ollama-translate-model qwen3:8b \
  --ollama-verify-model qwen3:8b
```

No `--stage` flag is required; full staged processing is already the native operation.

## Optional explicit staged workflow (memory-friendly)

```bash
# Stage 1: OCR only (loads OCR model only)
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --stage ocr \
  --force-ocr --ocr-engine glm-ocr --glm-ocr-model glm-ocr:latest

# Stage 2: Normalize German (loads normalize model only)
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --stage normalize \
  --normalize-with-ollama \
  --ollama-normalize-model gemma3:4b

# Stage 3: Translate to English (loads translation model only)
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --stage translate \
  --translator-backend ollama \
  --ollama-translate-model qwen3:8b

# Stage 4: Verify (optional; loads verifier model only)
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --stage verify \
  --ollama-verify-model qwen3:8b
```

Run one job at a time (avoid parallel script runs) to reduce Ollama connection drops.

## If verifier model fails with HTTP 500 / unsupported weight format

This can happen after an Ollama upgrade with older cached model blobs.

Repair by re-downloading the verifier model:

```bash
ollama rm glm-4.7-flash:latest
ollama pull glm-4.7-flash:latest
```

Temporary workaround: use a different verifier model (for example `qwen3:8b`):

```bash
--ollama-verify-model qwen3:8b
```

By default, verifier failures are warnings and the run continues. Add `--verify-strict` if you want failures to stop the run.

The script writes and reuses `*.work.json` between stages so page processing remains consistent.
You can disable passes explicitly with `--no-normalize-with-ollama` and/or `--no-verify-with-ollama`.

## OpenAI translation option (if desired)

```bash
export OPENAI_API_KEY="your_key_here"
./translate_pdf.py "Entdecktes Judenthum (Johann Andreas Eisenmenger) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --force-ocr --ocr-lang deu_frak+deu \
  --normalize-with-ollama \
  --translator-backend openai \
  --openai-model gpt-4.1-mini
```

## Outputs

Written to `/Users/seandolbec/Desktop/German_To_English/output/`:

- `*.de.raw.txt` raw German OCR text
- `*.de.normalized.txt` cleaned modern German
- `*.en.txt` English translation
- `*.translated_english_final.html` English-only, page-by-page deliverable
- `*.scholar_german_english.html` German/English side-by-side deliverable
- `*.review.html` 3-column review (Raw German / Normalized German / English)
- `*.pages.json` per-page structured data
- `*.work.json` stage state used between runs

Open `*.review.html` in a browser and Print -> Save as PDF for side-by-side scholarly review.
