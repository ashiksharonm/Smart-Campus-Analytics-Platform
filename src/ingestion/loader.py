"""
Data Loader
Reads raw CSV and optionally persists to DuckDB.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_FILE = RAW_DIR / "students.csv"


def load_raw(filepath: Path = RAW_FILE) -> pd.DataFrame:
    """Load raw CSV into a pandas DataFrame."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data not found at {filepath}. "
            "Run `python src/ingestion/generate_data.py` first."
        )
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} rows from {filepath.name}")
    return df
