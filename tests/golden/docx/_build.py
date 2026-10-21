"""Step D4 docx fixture builder — a minimal but REAL .docx container.

``uv run python tests/golden/docx/_build.py`` rewrites ``report.docx``.

The container mirrors the D3 adapter test's builder conventions (stdlib
``zipfile``, mixed compression methods, even-second DOS timestamps — the
zip date field stores seconds at 2-second granularity). The document is a
mixed-language three-paragraph note: a zh-CN title, an en-US sentence and
a zh-Hans sentence, each run carrying its own ``w:lang`` — the language
attributes the D4 e2e op must preserve while translating the zh-Hans
paragraph text into English.

The expected translated document is derived by the SAME independent
line-splice oracle discipline as the mixed_md corpus: the document XML's
lines with the declared ``<w:t>`` line swapped for its translation, every
other byte identical. It lives in ``expected_document.xml`` next to the
container so the test can assert the merged document text against it
before any container bytes are compared.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

DIR = Path(__file__).resolve().parent

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    "<Default Extension='rels' ContentType='x'/>"
    "<Default Extension='xml' ContentType='y'/></Types>\n"
)
RELS = '<?Relationships xmlns="http://x"><Relationship Id="r1"/></Relationships>\n'
STYLES = "<w:styles xmlns:w='http://w'></w:styles>\n"

DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
    "  <w:body>\n"
    "    <w:p>\n"
    "      <w:r>\n"
    '        <w:rPr><w:lang w:val="zh-CN"/></w:rPr>\n'
    '        <w:t xml:space="preserve">用户数据流水线说明</w:t>\n'
    "      </w:r>\n"
    "    </w:p>\n"
    "    <w:p>\n"
    "      <w:r>\n"
    '        <w:rPr><w:lang w:val="en-US"/></w:rPr>\n'
    '        <w:t xml:space="preserve">Pipeline field notes.</w:t>\n'
    "      </w:r>\n"
    "    </w:p>\n"
    "    <w:p>\n"
    "      <w:r>\n"
    '        <w:rPr><w:lang w:val="zh-Hans"/></w:rPr>\n'
    '        <w:t xml:space="preserve">所有 region 的数据先进入临时队列。</w:t>\n'
    "      </w:r>\n"
    "    </w:p>\n"
    "  </w:body>\n"
    "</w:document>\n"
)

# The declared translation the D4 e2e op applies (line-splice oracle). The
# shared ``xml:space="preserve"`` LHS gives the old and new <w:t> lines one
# replacement identity — the keyed deletion rule — so the op needs no
# preservation marker (measured: the real model echoes the # marker into
# XML output verbatim on every attempt, and a leaked marker is rightly
# rejected).
ZH_LINE = '        <w:t xml:space="preserve">所有 region 的数据先进入临时队列。</w:t>\n'
EN_LINE = (
    '        <w:t xml:space="preserve">'
    "All region data enters a temporary queue first.</w:t>\n"
)


def main() -> None:
    assert DOCUMENT.count(ZH_LINE) == 1, "fixture drift: the zh-Hans w:t line"
    expected = DOCUMENT.replace(ZH_LINE, EN_LINE)
    assert expected != DOCUMENT
    (DIR / "expected_document.xml").write_text(expected, encoding="utf-8")

    path = DIR / "report.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)  # deflate
        stored = zipfile.ZipInfo("_rels/.rels", date_time=(2024, 1, 2, 3, 4, 4))
        stored.compress_type = zipfile.ZIP_STORED
        zf.writestr(stored, RELS)
        zf.writestr("word/document.xml", DOCUMENT)  # deflate
        styles = zipfile.ZipInfo("word/styles.xml", date_time=(2024, 6, 7, 8, 9, 10))
        styles.compress_type = zipfile.ZIP_STORED
        zf.writestr(styles, STYLES)
    print(f"wrote {path} ({path.stat().st_size} bytes) + expected_document.xml")


if __name__ == "__main__":
    main()
