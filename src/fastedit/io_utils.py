"""Encoding-safe source I/O: read_source / write_source.

Every read that feeds an edit used to decode with ``errors="replace"``
(B21): each undecodable byte became U+FFFD and was written back as
EF BF BD — permanent corruption of lines the edit never touched. Every
write used to encode UTF-8 regardless of how the file was decoded (B23).
This module is the single sanctioned read/write pair for source files:

* detection order: BOM sniff (reusing ``filetype``'s BOM table) → binary
  gate (reusing ``filetype.detect_file_type``'s classification) → strict
  UTF-8 → latin-1;
* decode is STRICT everywhere — no ``errors=`` argument survives here. A
  file is either decoded exactly (UTF-8, UTF-8-BOM, latin-1) or refused;
* UTF-16/UTF-32 and binary content raise
  :class:`UnsupportedEncodingError` with a message that says the file was
  not modified — a UTF-16 file must never reach tree-sitter as mojibake;
* ``write_source`` encodes with the SAME codec that decoded the file, so
  ``read_source`` → edit → ``write_source`` is a byte-exact round-trip
  outside the edited span (B23).
"""

from __future__ import annotations

import os
from pathlib import Path

# Reuse filetype.py's BOM table and text/binary classification verbatim --
# duplicating the heuristics here would let the two detectors drift apart.
from .filetype import _BOM_TEXT_ENCODINGS, detect_file_type

__all__ = [
    "UnsupportedEncodingError",
    "read_source",
    "write_all",
    "write_source",
]


def write_all(fd: int, data: bytes) -> None:
    """Write ALL of *data* to the raw file descriptor *fd*, looping on
    partial writes.

    A single ``os.write`` is allowed to write fewer bytes than requested
    and report the count without error — POSIX promises only the returned
    count (a >2 GiB single write is capped by the kernel's MAX_RW_COUNT on
    Linux; a signal interruption can stop one early), and Windows has its
    own per-call limits. Ignoring the return value silently TRUNCATES the
    payload in exactly those cases. Every descriptor-level write of
    unbounded-size content (whole-file backups, whole-file temp copies)
    must go through this loop: it completes only when every byte landed,
    and refuses to spin forever if the descriptor reports a zero-byte
    write for non-empty data.
    """
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - defensive: no spin on 0
            raise OSError(
                f"os.write reported {written} bytes written for "
                f"{len(view)} remaining"
            )
        view = view[written:]


class UnsupportedEncodingError(ValueError):
    """A file's encoding cannot be edited safely as text.

    Raised for UTF-16/UTF-32 files (NUL-byte encodings that would reach
    tree-sitter as mojibake), for binary content, and for content that
    cannot be re-encoded with the file's own codec on write. Subclasses
    :class:`ValueError` so every existing ``except ValueError`` refusal
    path treats it as a clean, user-facing refusal rather than a crash.
    """


def read_source(
    path: str | Path, return_stat: bool = False,
) -> tuple[str, str] | tuple[str, str, os.stat_result]:
    """Read a source file and return ``(text, encoding)``.

    With ``return_stat=True`` returns ``(text, encoding, stat)`` where
    *stat* is the ``os.stat_result`` captured via ``os.fstat`` from the
    SAME open the bytes were read through. Pass it to
    ``mcp.backup._atomic_write(expected_stat=...)`` so the write can refuse
    (B37 lost-update guard) when the file changed on disk between this read
    and the write. The default stays the 2-tuple so existing callers are
    untouched.

    Detection order:

    1. empty file → ``("", "utf-8")``;
    2. BOM sniff, in ``filetype._BOM_TEXT_ENCODINGS`` order: a UTF-8 BOM
       selects the ``utf-8-sig`` codec (the codec strips the BOM on decode
       and re-adds it on encode, which is what makes the round-trip exact);
       a UTF-16/UTF-32 BOM is refused;
    3. binary gate: ``filetype.detect_file_type`` must classify the content
       as text. This catches no-BOM UTF-16 (dense with NUL bytes, per
       filetype's heuristic) and true binary before any codec is chosen;
    4. strict UTF-8;
    5. latin-1 — filetype's remaining text class ("decodable as latin-1
       with a low control-byte ratio"). latin-1 maps every byte to the same
       codepoint, so the decode is total and re-encoding is byte-exact.
       (cp1252 is deliberately NOT used: it has five undefined code points
       that would not round-trip.)

    Raises:
        UnsupportedEncodingError: for UTF-16/UTF-32 and binary content.
            The message always states that the file was not modified.
        OSError: if the file cannot be read.
    """
    with open(path, "rb") as fh:
        st = os.fstat(fh.fileno())
        data = fh.read()
    text, encoding = _decode_source_bytes(data, path)
    if return_stat:
        return text, encoding, st
    return text, encoding


def _decode_source_bytes(data: bytes, path: str | Path) -> tuple[str, str]:
    """Decode *data* per read_source's detection order. Split out so the
    ``return_stat`` read path and the decode contract cannot drift apart.
    """
    if not data:
        return "", "utf-8"

    for bom, encoding in _BOM_TEXT_ENCODINGS:
        if not data.startswith(bom):
            continue
        if encoding != "utf-8-sig":
            # UTF-16/UTF-32 BOM: refused, never decoded. The file on disk
            # is untouched -- the caller has not written anything yet.
            # (encoding.upper(): the canonical UTF-16/UTF-32 display form.)
            raise UnsupportedEncodingError(
                f"{path}: {encoding.upper()}-encoded file (BOM detected). "
                f"FastEdit edits UTF-8 and latin-1 text only; convert the "
                f"file to UTF-8 first. The file was not modified."
            )
        try:
            return data.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as e:
            raise UnsupportedEncodingError(
                f"{path}: UTF-8 BOM present but the content is not valid "
                f"UTF-8 ({e}). The file was not modified."
            ) from e

    detection = detect_file_type(data)
    if not detection.is_text:
        raise UnsupportedEncodingError(
            f"{path}: not editable text ({detection.reason}). The file was "
            f"not modified."
        )

    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    # filetype.py's remaining text class. latin-1 decodes every byte
    # strictly (total codec), so no errors= fallback is ever needed.
    return data.decode("latin-1"), "latin-1"


def write_source(path: str | Path, text: str, encoding: str) -> None:
    """Write *text* back to *path* with the SAME codec that decoded it.

    Delegates to ``mcp.backup._atomic_write`` so the write stays atomic,
    keeps the BOM-restore policy, and saves the pre-edit backup. Passing
    the read-time encoding here is the B23 round-trip guarantee: whatever
    ``read_source`` returned as its codec must be handed back unchanged.

    Raises:
        UnsupportedEncodingError: if *text* contains characters *encoding*
            cannot represent (e.g. an emoji merged into a latin-1 file).
            Raised BEFORE anything is written; the file on disk is
            untouched.
    """
    # Imported lazily: backup.py belongs to the mcp package tree, and this
    # module must stay importable wherever filetype.py is (the CLI imports
    # io_utils without pulling in the `mcp` distribution).
    from .mcp.backup import _atomic_write

    _atomic_write(Path(path), text, encoding=encoding)
