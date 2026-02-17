#!/usr/bin/env python3
"""OCR + German->English translation pipeline for PDFs.

Outputs:
- raw extracted German text (OCR/direct)
- normalized German text (optional Ollama pass)
- English translation
- review HTML with columns for raw German, normalized German, and English

Backends:
- Ollama local models
- OpenAI Chat Completions API
"""

from __future__ import annotations

import argparse
import html
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd: list[str], *, capture: bool = True) -> str:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return proc.stdout if capture else ""


def has_command(name: str) -> bool:
    return subprocess.call(["bash", "-lc", f"command -v {name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def pdf_page_count(pdf_path: Path) -> int:
    out = run(["pdfinfo", str(pdf_path)])
    m = re.search(r"^Pages:\s+(\d+)", out, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not determine PDF page count via pdfinfo")
    return int(m.group(1))


def extract_text_page(pdf_path: Path, page: int) -> str:
    return run([
        "pdftotext",
        "-layout",
        "-f",
        str(page),
        "-l",
        str(page),
        str(pdf_path),
        "-",
    ])


def text_quality_score(text: str) -> float:
    if not text.strip():
        return 0.0
    printable = sum(ch.isprintable() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    weird = text.count("�") + text.count("\x0c")
    return (letters / max(1, len(text))) + (printable / max(1, len(text))) - (weird / max(1, len(text)))


def ocr_page(pdf_path: Path, page: int, ocr_lang: str, tmpdir: Path) -> str:
    png_path = page_to_png(pdf_path, page, tmpdir)

    txt_base = tmpdir / f"ocr_{page:04d}"
    run([
        "tesseract",
        str(png_path),
        str(txt_base),
        "-l",
        ocr_lang,
        "--psm",
        "1",
    ], capture=False)
    txt_path = txt_base.with_suffix(".txt")
    return txt_path.read_text(encoding="utf-8", errors="replace") if txt_path.exists() else ""


def page_to_png(pdf_path: Path, page: int, tmpdir: Path) -> Path:
    base = tmpdir / f"page_{page:04d}"
    run([
        "pdftoppm",
        "-f",
        str(page),
        "-l",
        str(page),
        "-r",
        "300",
        "-png",
        str(pdf_path),
        str(base),
    ])
    png_candidates = sorted(tmpdir.glob(f"page_{page:04d}-*.png"))
    if not png_candidates:
        raise RuntimeError(f"No PNG page image generated for page {page}")
    return png_candidates[0]


def glm_ocr_page(pdf_path: Path, page: int, tmpdir: Path, model: str) -> str:
    png_path = page_to_png(pdf_path, page, tmpdir)
    prompt = f"Text Recognition: {png_path}"
    return run(["ollama", "run", model, prompt]).strip()


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for p in paragraphs:
        add = len(p) + 2
        if current and size + add > max_chars:
            chunks.append("\n\n".join(current))
            current = [p]
            size = len(p)
        else:
            current.append(p)
            size += add

    if current:
        chunks.append("\n\n".join(current))

    if not chunks and text.strip():
        chunks = textwrap.wrap(text, width=max_chars)
    return chunks


def openai_chat(user_prompt: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "You are a careful historical text assistant."},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error: HTTP {e.code}: {detail}") from e

    try:
        return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenAI response: {body}") from e


def ollama_generate(prompt: str, model: str, url_base: str, timeout: int, retries: int = 3) -> str:
    url = f"{url_base.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error: HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionResetError, socket.timeout) as e:
            if attempt >= retries:
                raise RuntimeError(f"Ollama request failed after {retries} attempts: {e}") from e
            time.sleep(min(5 * attempt, 15))

    text = body.get("response", "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned empty response for model {model}")
    return text


def normalize_german(text: str, model: str, url_base: str, timeout: int) -> str:
    prompt = (
        "Du bekommst OCR-Text aus einem historischen deutschen Druck (teilweise Fraktur, OCR-Fehler). "
        "Aufgabe: Schreibe den Text in gut lesbares, modernes Deutsch um, aber ohne zu kuerzen oder zusammenzufassen. "
        "Erhalte Eigennamen, Zahlen, Zitate und Absatzstruktur. "
        "Wenn ein Wort unklar ist, nutze die wahrscheinlichste Lesart. "
        "Gib NUR den normalisierten deutschen Text aus. Keine Erklaerungen.\n\n"
        f"OCR-Text:\n{text}"
    )
    return ollama_generate(prompt, model=model, url_base=url_base, timeout=timeout)


def verify_normalized_german(raw: str, normalized: str, model: str, url_base: str, timeout: int) -> str:
    prompt = (
        "Vergleiche OCR-Rohtext und normalisiertes Deutsch. "
        "Korrigiere das normalisierte Deutsch nur dort, wo es den Sinn des OCR-Textes klar verfelscht. "
        "Keine Zusammenfassung, keine neuen Informationen. Behalte Absatzstruktur. Keine Erklaerungen. "
        "Gib NUR die korrigierte deutsche Fassung aus.\n\n"
        f"OCR-Rohtext:\n{raw}\n\nNormalisierte Fassung:\n{normalized}"
    )
    return ollama_generate(prompt, model=model, url_base=url_base, timeout=timeout)


def translate_german_to_english_ollama(text: str, model: str, url_base: str, timeout: int) -> str:
    prompt = (
        "Translate the following historical German text into clear scholarly English. "
        "Preserve names, dates, citations, and paragraph boundaries. Do not summarize. "
        "Output only the English translation.\n\n"
        f"German text:\n{text}"
    )
    return ollama_generate(prompt, model=model, url_base=url_base, timeout=timeout)


def verify_english_translation(german: str, english: str, model: str, url_base: str, timeout: int) -> str:
    prompt = (
        "Check this English translation against the German source. "
        "Fix only mistranslations or missing meaning. Keep scholarly tone and paragraph boundaries. "
        "Do not add commentary. Output only corrected English.\n\n"
        f"German source:\n{german}\n\nEnglish translation:\n{english}"
    )
    return ollama_generate(prompt, model=model, url_base=url_base, timeout=timeout)


def translate_german_to_english_openai(text: str, model: str) -> str:
    prompt = (
        "Translate this historical German text into clear scholarly English. "
        "Preserve names, dates, citations, and paragraph boundaries. "
        "Do not summarize. Output only the English translation.\n\n"
        f"German text:\n{text}"
    )
    return openai_chat(prompt, model=model)


def render_review_html(raw_pages: list[str], normalized_pages: list[str], english_pages: list[str], title: str) -> str:
    rows = []
    for i, (raw_de, norm_de, en) in enumerate(zip(raw_pages, normalized_pages, english_pages), start=1):
        rows.append(
            "<tr>"
            f"<td class='page'>Page {i}</td>"
            f"<td><pre>{html.escape(raw_de)}</pre></td>"
            f"<td><pre>{html.escape(norm_de)}</pre></td>"
            f"<td><pre>{html.escape(en)}</pre></td>"
            "</tr>"
        )
    body = "\n".join(rows)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f8f6ef;
  --card: #fffdf7;
  --ink: #1e1b18;
  --grid: #dfd5c0;
  --accent: #6b4b2a;
}}
body {{
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  background: radial-gradient(circle at 10% 10%, #fff8e6, var(--bg) 45%);
  color: var(--ink);
}}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 12px; color: var(--accent); }}
p.note {{ margin: 0 0 20px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--grid); }}
th, td {{ border: 1px solid var(--grid); vertical-align: top; padding: 10px; }}
th {{ position: sticky; top: 0; background: #f4ead3; z-index: 1; }}
pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; line-height: 1.35; font-size: 13px; }}
td.page {{ width: 80px; font-weight: 700; background: #f9f1de; }}
@media print {{
  th {{ position: static; }}
  body {{ background: #fff; }}
}}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="note">Raw OCR German, normalized German, and English translation side-by-side.</p>
  <table>
    <thead><tr><th>#</th><th>German (Raw OCR)</th><th>German (Normalized)</th><th>English</th></tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def render_bilingual_html(german_pages: list[str], english_pages: list[str], title: str) -> str:
    rows = []
    for i, (de, en) in enumerate(zip(german_pages, english_pages), start=1):
        rows.append(
            "<tr>"
            f"<td class='page'>Page {i}</td>"
            f"<td><pre>{html.escape(de)}</pre></td>"
            f"<td><pre>{html.escape(en)}</pre></td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f8f6ef;
  --card: #fffdf7;
  --ink: #1e1b18;
  --grid: #dfd5c0;
  --accent: #6b4b2a;
}}
body {{
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  background: radial-gradient(circle at 10% 10%, #fff8e6, var(--bg) 45%);
  color: var(--ink);
}}
main {{ max-width: 1300px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 12px; color: var(--accent); }}
p.note {{ margin: 0 0 20px; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--grid); }}
th, td {{ border: 1px solid var(--grid); vertical-align: top; padding: 10px; }}
th {{ position: sticky; top: 0; background: #f4ead3; z-index: 1; }}
pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; line-height: 1.35; font-size: 13px; }}
td.page {{ width: 80px; font-weight: 700; background: #f9f1de; }}
@media print {{
  th {{ position: static; }}
  body {{ background: #fff; }}
}}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="note">German on left, English translation on right (page-by-page).</p>
  <table>
    <thead><tr><th>#</th><th>German</th><th>English</th></tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
</main>
</body>
</html>
"""


def render_english_html(english_pages: list[str], title: str) -> str:
    sections = []
    for i, en in enumerate(english_pages, start=1):
        sections.append(
            f"<section><h2>Page {i}</h2><pre>{html.escape(en)}</pre></section>"
        )
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f8f6ef;
  --card: #fffdf7;
  --ink: #1e1b18;
  --grid: #dfd5c0;
  --accent: #6b4b2a;
}}
body {{
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  background: radial-gradient(circle at 10% 10%, #fff8e6, var(--bg) 45%);
  color: var(--ink);
}}
main {{ max-width: 900px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 16px; color: var(--accent); }}
h2 {{ margin: 20px 0 8px; font-size: 18px; }}
section {{ background: var(--card); border: 1px solid var(--grid); padding: 12px; margin-bottom: 14px; }}
pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; line-height: 1.45; font-size: 14px; }}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
{body}
</main>
</body>
</html>
"""


def selected_pages(total_pages: int, start_page: int, end_page: int, max_pages: int) -> list[int]:
    end = end_page if end_page > 0 else total_pages
    start = max(1, start_page)
    end = min(total_pages, end)
    pages = list(range(start, end + 1))
    if max_pages > 0:
        pages = pages[:max_pages]
    return pages


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Stage requires work file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in data:
        records.append(
            {
                "page_pdf": int(row.get("page_pdf", row.get("page", 0))),
                "german_raw": str(row.get("german_raw", "")),
                "german_normalized": str(row.get("german_normalized", "")),
                "english": str(row.get("english", "")),
            }
        )
    return records


def save_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_progress(stage: str, page_pdf: int, index: int, total: int) -> None:
    remaining = max(total - index, 0)
    print(
        f"Stage {stage.upper()} | Page {page_pdf} | {index}/{total} | Remaining {remaining}",
        file=sys.stderr,
    )


def maybe_stop_ollama_model(model: str) -> None:
    try:
        run(["ollama", "stop", model], capture=False)
    except Exception:
        pass


def write_outputs(out_dir: Path, stem: str, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_pages = [r["german_raw"] for r in records]
    norm_pages = [r["german_normalized"] or r["german_raw"] for r in records]
    en_pages = [r["english"] for r in records]

    raw_de_txt = out_dir / f"{stem}.de.raw.txt"
    norm_de_txt = out_dir / f"{stem}.de.normalized.txt"
    en_txt = out_dir / f"{stem}.en.txt"
    review_html = out_dir / f"{stem}.review.html"
    scholar_html = out_dir / f"{stem}.scholar_german_english.html"
    english_final_html = out_dir / f"{stem}.translated_english_final.html"
    work_json = out_dir / f"{stem}.work.json"
    pages_json = out_dir / f"{stem}.pages.json"

    raw_de_txt.write_text("\n\n\n".join(raw_pages), encoding="utf-8")
    norm_de_txt.write_text("\n\n\n".join(norm_pages), encoding="utf-8")
    en_txt.write_text("\n\n\n".join(en_pages), encoding="utf-8")
    review_html.write_text(
        render_review_html(raw_pages, norm_pages, en_pages, f"{stem} (DE OCR / DE Normalized / EN)"),
        encoding="utf-8",
    )
    scholar_html.write_text(
        render_bilingual_html(norm_pages, en_pages, f"{stem} (Scholar German/English)"),
        encoding="utf-8",
    )
    english_final_html.write_text(
        render_english_html(en_pages, f"{stem} (Translated English Final)"),
        encoding="utf-8",
    )
    pages_json.write_text(
        json.dumps(
            [
                {
                    "page_pdf": r["page_pdf"],
                    "german_raw": r["german_raw"],
                    "german_normalized": r["german_normalized"],
                    "english": r["english"],
                }
                for r in records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_records(work_json, records)

    print(f"Wrote raw German text: {raw_de_txt}")
    print(f"Wrote normalized German text: {norm_de_txt}")
    print(f"Wrote English translation: {en_txt}")
    print(f"Wrote review HTML: {review_html}")
    print(f"Wrote scholar bilingual HTML: {scholar_html}")
    print(f"Wrote English final HTML: {english_final_html}")
    print(f"Wrote pages JSON: {pages_json}")
    print(f"Wrote work JSON: {work_json}")
    print("Tip: Open HTML files in a browser and Print -> Save as PDF if needed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Staged German PDF OCR -> normalization -> translation pipeline")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--stage", choices=["all", "ocr", "normalize", "translate", "verify", "package"], default="all")
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=0, help="0 means last page")
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no limit")
    parser.add_argument("--ocr-engine", choices=["tesseract", "glm-ocr"], default="tesseract")
    parser.add_argument("--ocr-lang", default="deu_frak+deu")
    parser.add_argument("--glm-ocr-model", default="glm-ocr:latest")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--translator-backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--normalize-with-ollama", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-with-ollama", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-strict", action="store_true")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=int, default=600)
    parser.add_argument("--ollama-normalize-model", default="gemma3:4b")
    parser.add_argument("--ollama-translate-model", default="qwen3:8b")
    parser.add_argument("--ollama-verify-model", default="glm-4.7-flash:latest")
    args = parser.parse_args()

    if not args.input_pdf.exists():
        raise SystemExit(f"Input PDF not found: {args.input_pdf}")
    for cmd in ["pdfinfo", "pdftotext", "pdftoppm"]:
        if not has_command(cmd):
            raise SystemExit(f"Missing required command: {cmd}")

    need_ocr = args.stage in {"all", "ocr"}
    need_normalize = args.stage in {"all", "normalize"} and (args.normalize_with_ollama or args.stage == "normalize")
    need_translate = args.stage in {"all", "translate"} and not args.skip_translation
    need_verify = args.stage in {"all", "verify"} and (args.verify_with_ollama or args.stage == "verify")

    if need_ocr and args.ocr_engine == "tesseract" and not has_command("tesseract"):
        raise SystemExit("Missing required command: tesseract")
    if (need_ocr and args.ocr_engine == "glm-ocr") or need_normalize or need_verify or (need_translate and args.translator_backend == "ollama"):
        if not has_command("ollama"):
            raise SystemExit("Missing required command: ollama")

    total_pages = pdf_page_count(args.input_pdf)
    pages = selected_pages(total_pages, args.start_page, args.end_page, args.max_pages)
    selected = set(pages)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_pdf.stem
    work_json = args.out_dir / f"{stem}.work.json"

    if args.stage == "all":
        records: list[dict] = []
        total = len(pages)
        with tempfile.TemporaryDirectory(prefix="ocr_tmp_") as tmp:
            tmpdir = Path(tmp)
            for idx, page in enumerate(pages, start=1):
                direct = extract_text_page(args.input_pdf, page)
                use_ocr = args.force_ocr or len(direct.strip()) < 120 or text_quality_score(direct) < 0.80
                if use_ocr:
                    raw = glm_ocr_page(args.input_pdf, page, tmpdir, args.glm_ocr_model) if args.ocr_engine == "glm-ocr" else ocr_page(args.input_pdf, page, args.ocr_lang, tmpdir)
                else:
                    raw = direct
                raw = raw.strip() or "[No text extracted from this page]"
                stage_progress("ocr", page, idx, total)
                if args.ocr_engine == "glm-ocr":
                    maybe_stop_ollama_model(args.glm_ocr_model)

                rec = {"page_pdf": page, "german_raw": raw, "german_normalized": "", "english": ""}

                if need_normalize:
                    norm_chunks = [
                        normalize_german(ch, args.ollama_normalize_model, args.ollama_url, args.ollama_timeout)
                        for ch in chunk_text(rec["german_raw"])
                    ]
                    rec["german_normalized"] = "\n\n".join(norm_chunks).strip() or rec["german_raw"]
                    stage_progress("normalize", page, idx, total)
                    maybe_stop_ollama_model(args.ollama_normalize_model)

                if need_translate:
                    source = rec["german_normalized"] or rec["german_raw"]
                    chunks = chunk_text(source)
                    if args.translator_backend == "ollama":
                        translated = [translate_german_to_english_ollama(ch, args.ollama_translate_model, args.ollama_url, args.ollama_timeout) for ch in chunks]
                    else:
                        translated = [translate_german_to_english_openai(ch, args.openai_model) for ch in chunks]
                    rec["english"] = "\n\n".join(translated).strip()
                    stage_progress("translate", page, idx, total)
                    if args.translator_backend == "ollama":
                        maybe_stop_ollama_model(args.ollama_translate_model)

                if need_verify:
                    if rec["german_normalized"]:
                        raw_chunks = chunk_text(rec["german_raw"])
                        norm_chunks = chunk_text(rec["german_normalized"])
                        de_out = []
                        for i in range(min(len(raw_chunks), len(norm_chunks))):
                            try:
                                de_out.append(verify_normalized_german(raw_chunks[i], norm_chunks[i], args.ollama_verify_model, args.ollama_url, args.ollama_timeout))
                            except Exception as e:
                                if args.verify_strict:
                                    raise
                                print(f"WARN verifier failed (DE page {rec['page_pdf']} chunk {i+1}): {e}", file=sys.stderr)
                                de_out.append(norm_chunks[i])
                        rec["german_normalized"] = "\n\n".join(de_out).strip() or rec["german_normalized"]
                    if rec["english"]:
                        src = rec["german_normalized"] or rec["german_raw"]
                        src_chunks = chunk_text(src)
                        en_chunks = chunk_text(rec["english"])
                        en_out = []
                        for i in range(min(len(src_chunks), len(en_chunks))):
                            try:
                                en_out.append(verify_english_translation(src_chunks[i], en_chunks[i], args.ollama_verify_model, args.ollama_url, args.ollama_timeout))
                            except Exception as e:
                                if args.verify_strict:
                                    raise
                                print(f"WARN verifier failed (EN page {rec['page_pdf']} chunk {i+1}): {e}", file=sys.stderr)
                                en_out.append(en_chunks[i])
                        rec["english"] = "\n\n".join(en_out).strip() or rec["english"]
                    stage_progress("verify", page, idx, total)
                    maybe_stop_ollama_model(args.ollama_verify_model)

                records.append(rec)
                save_records(work_json, records)
                write_outputs(args.out_dir, stem, records)
        return 0

    if args.stage in {"all", "ocr"}:
        records: list[dict] = []
        total = len(pages)
        with tempfile.TemporaryDirectory(prefix="ocr_tmp_") as tmp:
            tmpdir = Path(tmp)
            for idx, page in enumerate(pages, start=1):
                direct = extract_text_page(args.input_pdf, page)
                use_ocr = args.force_ocr or len(direct.strip()) < 120 or text_quality_score(direct) < 0.80
                if use_ocr:
                    raw = glm_ocr_page(args.input_pdf, page, tmpdir, args.glm_ocr_model) if args.ocr_engine == "glm-ocr" else ocr_page(args.input_pdf, page, args.ocr_lang, tmpdir)
                else:
                    raw = direct
                records.append(
                    {
                        "page_pdf": page,
                        "german_raw": raw.strip() or "[No text extracted from this page]",
                        "german_normalized": "",
                        "english": "",
                    }
                )
                stage_progress("ocr", page, idx, total)
        if args.stage == "all" and args.ocr_engine == "glm-ocr":
            maybe_stop_ollama_model(args.glm_ocr_model)
    else:
        records = load_records(work_json)

    selected_records = [r for r in records if r["page_pdf"] in selected]
    total_selected = len(selected_records)

    if need_normalize:
        for idx, rec in enumerate(selected_records, start=1):
            norm_chunks = [
                normalize_german(ch, args.ollama_normalize_model, args.ollama_url, args.ollama_timeout)
                for ch in chunk_text(rec["german_raw"])
            ]
            rec["german_normalized"] = "\n\n".join(norm_chunks).strip() or rec["german_raw"]
            stage_progress("normalize", rec["page_pdf"], idx, total_selected)
        if args.stage == "all":
            maybe_stop_ollama_model(args.ollama_normalize_model)

    if need_translate:
        for idx, rec in enumerate(selected_records, start=1):
            source = rec["german_normalized"] or rec["german_raw"]
            chunks = chunk_text(source)
            if args.translator_backend == "ollama":
                translated = [translate_german_to_english_ollama(ch, args.ollama_translate_model, args.ollama_url, args.ollama_timeout) for ch in chunks]
            else:
                translated = [translate_german_to_english_openai(ch, args.openai_model) for ch in chunks]
            rec["english"] = "\n\n".join(translated).strip()
            stage_progress("translate", rec["page_pdf"], idx, total_selected)
        if args.stage == "all" and args.translator_backend == "ollama":
            maybe_stop_ollama_model(args.ollama_translate_model)

    if need_verify:
        for idx, rec in enumerate(selected_records, start=1):
            if rec["german_normalized"]:
                raw_chunks = chunk_text(rec["german_raw"])
                norm_chunks = chunk_text(rec["german_normalized"])
                de_out = []
                for i in range(min(len(raw_chunks), len(norm_chunks))):
                    try:
                        de_out.append(verify_normalized_german(raw_chunks[i], norm_chunks[i], args.ollama_verify_model, args.ollama_url, args.ollama_timeout))
                    except Exception as e:
                        if args.verify_strict:
                            raise
                        print(f"WARN verifier failed (DE page {rec['page_pdf']} chunk {i+1}): {e}", file=sys.stderr)
                        de_out.append(norm_chunks[i])
                rec["german_normalized"] = "\n\n".join(de_out).strip() or rec["german_normalized"]
            if rec["english"]:
                src = rec["german_normalized"] or rec["german_raw"]
                src_chunks = chunk_text(src)
                en_chunks = chunk_text(rec["english"])
                en_out = []
                for i in range(min(len(src_chunks), len(en_chunks))):
                    try:
                        en_out.append(verify_english_translation(src_chunks[i], en_chunks[i], args.ollama_verify_model, args.ollama_url, args.ollama_timeout))
                    except Exception as e:
                        if args.verify_strict:
                            raise
                        print(f"WARN verifier failed (EN page {rec['page_pdf']} chunk {i+1}): {e}", file=sys.stderr)
                        en_out.append(en_chunks[i])
                rec["english"] = "\n\n".join(en_out).strip() or rec["english"]
            stage_progress("verify", rec["page_pdf"], idx, total_selected)
        if args.stage == "all":
            maybe_stop_ollama_model(args.ollama_verify_model)

    save_records(work_json, records)
    if args.stage in {"all", "package"} or args.stage in {"translate", "verify", "ocr", "normalize"}:
        write_outputs(args.out_dir, stem, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
