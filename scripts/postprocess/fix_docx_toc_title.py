"""Post-process the generated DOCX TOC title.

Pandoc emits an English "Table of Contents" heading for DOCX even when the
source paper is Chinese.  This tiny post-processing step keeps the Word
deliverable consistent with the PDF title "目录" without changing the TOC field.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def replace_toc_title(docx_path: Path) -> None:
    if not docx_path.exists():
        raise FileNotFoundError(docx_path)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        with zipfile.ZipFile(docx_path, "r") as src:
            src.extractall(tmp_dir)

        document_xml = tmp_dir / "word" / "document.xml"
        text = document_xml.read_text(encoding="utf-8")
        text = text.replace(">Table of Contents<", ">目录<")
        document_xml.write_text(text, encoding="utf-8")

        temp_docx = docx_path.with_suffix(".tmp.docx")
        with zipfile.ZipFile(temp_docx, "w", zipfile.ZIP_DEFLATED) as dst:
            for path in tmp_dir.rglob("*"):
                if path.is_file():
                    dst.write(path, path.relative_to(tmp_dir))
        shutil.move(temp_docx, docx_path)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: fix_docx_toc_title.py <docx_path>")
    replace_toc_title(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
