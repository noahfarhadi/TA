"""
file_io.py

Reading uploaded delimited text files. Shared by oos_app.py and ta_app.py.
There is no Streamlit code in this file.

The delimiter and the decimal mark are detected for every file separately, so
a comma-separated weight file and a semicolon-separated price file with
decimal commas can be used together. Both can also be set by hand.
"""
from __future__ import annotations

import io
import re

import pandas as pd

SEPARATORS = {"Auto-detect": None, "Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t"}
DECIMALS = ["Auto", ".", ","]

_COMMA_NUMBER = re.compile(r"^-?\d+,\d+$")


def detect_separator(data: bytes) -> str:
    """Pick the delimiter that occurs most often in the header line."""
    lines = data[:20000].decode("utf-8-sig", errors="ignore").splitlines()
    header = lines[0] if lines else ""
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _has_decimal_comma(frame: pd.DataFrame, min_share: float = 0.8, sample: int = 200) -> bool:
    """True if a text column mostly holds numbers written like 12,5."""
    for col in frame.columns:
        if pd.api.types.is_string_dtype(frame[col]) or frame[col].dtype == object:
            values = frame[col].dropna().astype(str).head(sample)
            if len(values) and values.str.match(_COMMA_NUMBER).mean() >= min_share:
                return True
    return False


def read_table(data: bytes, separator=None, decimal: str = "Auto") -> pd.DataFrame:
    """
    Read a delimited file from bytes (UTF-8 first, then Latin-1).
    separator None = detect; decimal "Auto" = detect, otherwise "." or ",".
    """
    sep = separator if separator is not None else detect_separator(data)
    error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), sep=sep, decimal="." if decimal == "Auto" else decimal,
                                encoding=encoding)
            if decimal == "Auto" and _has_decimal_comma(frame):
                frame = pd.read_csv(io.BytesIO(data), sep=sep, decimal=",", encoding=encoding)
            return frame
        except UnicodeDecodeError as exc:
            error = exc
    raise error


def guess_column(columns, candidates):
    """First column whose lower-case name appears in the candidate list (or None)."""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-08)  New file. Added: SEPARATORS, DECIMALS, detect_separator,
#   read_table, guess_column.
