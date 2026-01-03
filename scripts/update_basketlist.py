#!/usr/bin/env python3
"""
Append an OPEC price row to data/Basketlist.csv using the date computed from the source's "Last Updated".

Key behaviors:
- Extract OPEC "Last" price and the raw token from data/Oilppricecharts.csv.
- Compute source_date from "Last Updated" (absolute or relative like "3 days Delay", "7 Hours Delay").
- Read data/Basketlist.csv as strings to preserve formatting and column order.
- Detect the Date column name and a likely date string format (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY).
- Append a new row with the Date formatted the same way and the OPEC_Price formatted to match the decimal places of the source token.
- If OPEC_Price column does not exist, it will be added as the last column.
- Appends even if an entry for that date already exists (no dedup by default).
"""

from pathlib import Path
import pandas as pd
import re
import sys
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser

# ---- Config ----
BASKET_PATH = Path("data/Basketlist.csv")
SOURCE_PATH = Path("data/Oilppricecharts.csv")
PRICE_COLUMN_NAME = "OPEC_Price"
DATE_COL_CANDIDATES = ["Date", "date", "DATE"]
# ----------------

def extract_first_number_and_raw(s: str):
    """Return (float_value, raw_token_string, decimal_places) or (None, None, None)."""
    if s is None:
        return None, None, None
    text = str(s).strip()
    m = re.search(r'[-+]?\d{1,3}(?:[,]\d{3})*(?:\.\d+)?', text)
    if not m:
        return None, None, None
    raw = m.group(0)
    num = raw.replace(",", "")
    try:
        val = float(num)
    except ValueError:
        return None, raw, None
    # decimal places detection
    dec = None
    if "." in raw:
        dec = len(raw.split(".")[1])
    else:
        dec = 0
    return val, raw, dec

def find_date_col(cols):
    for c in DATE_COL_CANDIDATES:
        if c in cols:
            return c
    for c in cols:
        if "date" in c.lower():
            return c
    return None

def parse_source_table_with_pandas(path: Path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] >= 2:
            return df
    except Exception:
        return None
    return None

def parse_source_table_fallback(path: Path):
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = re.split(r'\t| {2,}', line)
            parts = [p.strip() for p in parts if p.strip() != ""]
            if parts:
                rows.append(parts)
    if not rows:
        return None
    header = rows[0]
    use_header = any("last" in h.lower() or "change" in h.lower() or "futures" in h.lower() for h in header)
    if use_header:
        data = rows[1:]
        cols = header
    else:
        maxlen = max(len(r) for r in rows)
        cols = [f"col{idx}" for idx in range(maxlen)]
        data = [r + [""]*(maxlen - len(r)) for r in rows]
    df = pd.DataFrame(data, columns=cols)
    return df

def parse_last_updated_token(token: str):
    """
    Try to parse token as absolute datetime via dateutil or
    as relative delays like '3 days Delay', '7 Hours Delay', '16 Minutes Delay'
    Returns timezone-aware datetime (UTC) or None.
    """
    if token is None:
        return None
    s = str(token).strip().strip("()").strip()
    if not s:
        return None

    # Try absolute parse
    try:
        dt = dateparser.parse(s, default=datetime.utcnow())
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
    except Exception:
        pass

    # Try relative patterns like "3 days Delay", "7 Hours Delay"
    m = re.search(r'(\d+)\s*(day|days|hour|hours|minute|minutes)', s, flags=re.IGNORECASE)
    if m:
        val = int(m.group(1))
        unit = m.group(2).lower()
        now = datetime.now(timezone.utc)
        if 'day' in unit:
            return now - timedelta(days=val)
        if 'hour' in unit:
            return now - timedelta(hours=val)
        if 'minute' in unit:
            return now - timedelta(minutes=val)

    # Try short forms like '3d'
    m2 = re.search(r'(\d+)\s*d\b', s, flags=re.IGNORECASE)
    if m2:
        return datetime.now(timezone.utc) - timedelta(days=int(m2.group(1)))

    return None

def get_opec_price_and_token_from_source(path: Path):
    df = parse_source_table_with_pandas(path)
    if df is None:
        df = parse_source_table_fallback(path)
    if df is None:
        raise RuntimeError(f"Could not parse source file {path}")

    cols = list(df.columns)
    # pick name column heuristically
    name_col = None
    for c in cols:
        if "future" in c.lower() or "index" in c.lower() or "&" in c:
            name_col = c
            break
    if name_col is None:
        name_col = cols[0]

    # find 'Last' column
    last_col = None
    for c in cols:
        if "last" == c.lower() or "last " in c.lower() or "last" in c.lower():
            last_col = c
            break
    if last_col is None and len(cols) >= 2:
        last_col = cols[1]

    # find 'Last Updated' column
    last_updated_col = None
    for c in cols:
        if "last" in c.lower() and "updated" in c.lower():
            last_updated_col = c
            break
        if "updated" in c.lower() or "lastupdated" in c.lower() or "last updated" in c.lower():
            last_updated_col = c
            break

    # find opec row
    opec_row = None
    for idx, row in df.iterrows():
        cell = str(row.get(name_col, "")).lower()
        if "opec" in cell:
            opec_row = row
            break
    if opec_row is None:
        # try searching all cell values
        for idx, row in df.iterrows():
            joined = " ".join(str(v).lower() for v in row.tolist())
            if "opec" in joined:
                opec_row = row
                break

    if opec_row is None:
        raise RuntimeError("Could not find 'opec' row in source")

    # extract price and raw token
    price_token_source = None
    if last_col and last_col in df.columns:
        price_token_source = opec_row.get(last_col)
    if price_token_source is None:
        vals = [v for v in opec_row.tolist() if str(v).strip() != ""]
        if len(vals) >= 2:
            price_token_source = vals[1]
        else:
            raise RuntimeError("Couldn't extract 'Last' token from OPEC row")

    price_val, price_raw, price_decimals = extract_first_number_and_raw(price_token_source)
    if price_val is None:
        raise RuntimeError(f"Couldn't parse numeric price from token: {price_token_source}")

    # parse last updated
    source_dt = None
    if last_updated_col and last_updated_col in df.columns:
        raw = opec_row.get(last_updated_col)
        parsed = parse_last_updated_token(raw)
        if parsed:
            source_dt = parsed.date()
    else:
        # try any column looking for 'updated'-like token in the OPEC row
        for c in cols:
            if "updated" in c.lower() or "time" in c.lower() or "last" in c.lower():
                raw = opec_row.get(c)
                parsed = parse_last_updated_token(raw)
                if parsed:
                    source_dt = parsed.date()
                    break

    return price_val, price_raw, price_decimals, source_dt

def detect_date_string_format(sample: str):
    """Return a strftime format string for common patterns, or default '%Y-%m-%d'."""
    if sample is None:
        return "%Y-%m-%d"
    s = str(sample).strip()
    # YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return "%Y-%m-%d"
    # YYYY/MM/DD
    if re.match(r'^\d{4}/\d{2}/\d{2}$', s):
        return "%Y/%m/%d"
    # MM/DD/YYYY
    if re.match(r'^\d{2}/\d{2}/\d{4}$', s):
        return "%m/%d/%Y"
    # DD-MM-YYYY
    if re.match(r'^\d{2}-\d{2}-\d{4}$', s):
        return "%d-%m-%Y"
    # fallback
    return "%Y-%m-%d"

def atomic_write(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def main():
    if not BASKET_PATH.exists():
        print(f"Basketlist file not found at {BASKET_PATH}", file=sys.stderr)
        sys.exit(2)
    if not SOURCE_PATH.exists():
        print(f"Source price file not found at {SOURCE_PATH}", file=sys.stderr)
        sys.exit(2)

    try:
        price_val, price_raw, price_decimals, source_date = get_opec_price_and_token_from_source(SOURCE_PATH)
    except Exception as e:
        print(f"Error extracting price from source: {e}", file=sys.stderr)
        sys.exit(3)

    # fallback to today's UTC date if source_date missing
    if source_date is None:
        source_date = datetime.now(timezone.utc).date()

    # Read Basketlist as strings to preserve formatting
    basket_df = pd.read_csv(BASKET_PATH, dtype=str, keep_default_na=False)

    # detect date column
    date_col = find_date_col(basket_df.columns)
    if date_col is None:
        # no date column: create one named 'Date' and preserve original columns
        date_col = "Date"
        if date_col not in basket_df.columns:
            basket_df[date_col] = ""

    # pick a sample non-empty date value to detect format
    sample_date = None
    non_empty = [v for v in basket_df[date_col].tolist() if str(v).strip() != ""]
    if non_empty:
        sample_date = non_empty[0]
    detected_fmt = detect_date_string_format(sample_date)

    # format source_date to string using detected format
    try:
        formatted_date = source_date.strftime(detected_fmt)
    except Exception:
        formatted_date = source_date.isoformat()

    # ensure OPEC_Price column exists; if not, add it at the end preserving order
    if PRICE_COLUMN_NAME not in basket_df.columns:
        basket_df[PRICE_COLUMN_NAME] = ""

    # format price string to match decimals from source token if available
    if price_raw is not None and price_decimals is not None:
        if price_decimals > 0:
            price_str = f"{price_val:.{price_decimals}f}"
        else:
            # no decimals in source token
            price_str = f"{int(round(price_val))}"
    else:
        price_str = str(price_val)

    # build new row (strings for all columns to preserve formatting)
    new_row = {c: "" for c in basket_df.columns}
    new_row[date_col] = formatted_date
    new_row[PRICE_COLUMN_NAME] = price_str

    # Append the row
    appended = pd.concat([basket_df, pd.DataFrame([new_row])], ignore_index=True)

    # Write back atomically
    atomic_write(appended, BASKET_PATH)
    print(f"Appended row to {BASKET_PATH}: Date={formatted_date}, {PRICE_COLUMN_NAME}={price_str}")

if __name__ == "__main__":
    main()
