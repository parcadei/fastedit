"""Content-based text/binary detection.

Sniffs file content -- never trusts a filename or extension. Used by
`fastedit create` and `fastedit duplicate` to refuse writing/copying
binary data as if it were text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_BOM_TEXT_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)

_ALLOWED_CONTROL_BYTES = {0x09, 0x0A, 0x0D, 0x0C}
_SNIFF_WINDOW = 8192
_BINARY_CONTROL_RATIO = 0.30


@dataclass(frozen=True)
class FileTypeResult:
    """Outcome of a text/binary sniff, with a human-readable reason."""

    is_text: bool
    reason: str


def _control_byte_ratio_is_binary(sniff: bytes) -> bool:
    if not sniff:
        return False
    controls = sum(1 for b in sniff if b < 0x20 and b not in _ALLOWED_CONTROL_BYTES)
    return (controls / len(sniff)) > _BINARY_CONTROL_RATIO


def detect_file_type(data: bytes) -> FileTypeResult:
    """Classify raw bytes as text or binary by sniffing content, never a filename."""
    if not data:
        return FileTypeResult(True, "empty file")

    for bom, encoding in _BOM_TEXT_ENCODINGS:
        if data.startswith(bom):
            try:
                data.decode(encoding)
            except UnicodeDecodeError:
                return FileTypeResult(False, f"{encoding} BOM present but content is not valid {encoding}")
            return FileTypeResult(True, f"{encoding} BOM detected")

    sniff = data[:_SNIFF_WINDOW]
    # UTF-16 text with no BOM is caught right here by the NUL check, not by
    # the "decode utf-16" attempt further down: every ASCII codepoint
    # encodes as "<byte> 0x00" (or the reverse in big-endian UTF-16), so
    # real UTF-16-no-BOM content longer than a few characters is dense with
    # NUL bytes within any 8KB window. Verified against a real UTF-16LE
    # sample: both `file(1)` on macOS ("data", not "text") and
    # `git diff --numstat` (the "-\t-" binary marker) classify it as binary
    # too -- this is the same heuristic they use.
    if b"\x00" in sniff:
        return FileTypeResult(False, "NUL byte found in first 8KB")

    try:
        data.decode("utf-8")
        return FileTypeResult(True, "valid UTF-8")
    except UnicodeDecodeError:
        pass

    try:
        data.decode("utf-16")
        return FileTypeResult(True, "valid UTF-16 (no BOM)")
    except UnicodeDecodeError:
        pass

    if _control_byte_ratio_is_binary(sniff):
        return FileTypeResult(False, "over 30% non-printable control bytes in the first 8KB")
    return FileTypeResult(True, "decodable as latin-1 with a low control-byte ratio")


def is_text_file(path_or_bytes) -> FileTypeResult:
    """Sniff a path's content, or raw bytes directly, and classify it as text/binary."""
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = path_or_bytes
    return detect_file_type(data)
