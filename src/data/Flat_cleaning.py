import numpy as np
import pandas as pd
import re
import os
from typing import Optional


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        if "area" in df.columns:
            df.rename(columns={"area": "price_per_sqft"}, inplace=True)

        if "price_per_sqft" not in df.columns:
            raise KeyError("Neither 'area' nor 'price_per_sqft' found in CSV")

        for c in ("link", "property_id"):
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        def _clean_society(name):
            try:
                s = str(name)
                return re.sub(r'\d+(\.\d+)?\s?★', '', s).strip().lower()
            except Exception:
                return ""

        if "society" in df.columns:
            df["society"] = df["society"].apply(_clean_society)
        else:
            df["society"] = ""

        df = df[df["price"].fillna("").astype(str).str.lower() != "price on request"]

        return df

    except Exception as e:
        raise ValueError(f"load_data failed: {e}") from e


def treat_price(x) -> Optional[float]:
    try:
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)

        if isinstance(x, str):
            tokens = x.split()
        else:
            tokens = list(x) if x is not None else []

        if len(tokens) == 0:
            return None

        first = str(tokens[0]).replace(",", "").strip()
        val = float(first)

        unit = tokens[1].lower() if len(tokens) > 1 and isinstance(tokens[1], str) else ""
        if unit.startswith("lac") or unit.startswith("lakh"):
            return round(val / 100.0, 2)
        elif unit.startswith("cr"):
            return round(val, 2)
        else:
            return round(val, 2)

    except Exception:
        return None


def clean_function(df: pd.DataFrame) -> pd.DataFrame:
    try:
        def _safe_split_price(v):
            try:
                if pd.isna(v):
                    return None
                if isinstance(v, (float, int, np.number)):
                    return v
                return str(v).split()
            except Exception:
                return None

        df["price"] = df["price"].apply(_safe_split_price).apply(treat_price)

        df["price_per_sqft"] = (
            df["price_per_sqft"]
            .astype(str)
            .str.split("/")
            .str.get(0)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["price_per_sqft"] = pd.to_numeric(df["price_per_sqft"], errors="coerce")

        df = df[~df["bedRoom"].isnull()]

        return df

    except Exception as e:
        raise ValueError(f"clean_function failed: {e}") from e


def change_data_type(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        df["bedRoom"] = (
            df["bedRoom"].astype(str).str.split().str.get(0).str.extract(r'(\d+)', expand=False)
        )
        df["bedRoom"] = pd.to_numeric(df["bedRoom"], errors="coerce").fillna(0).astype(int)

        if "bathroom" in df.columns:
            df["bathroom"] = (
                df["bathroom"].astype(str).str.split().str.get(0).str.extract(r'(\d+)', expand=False)
            )
            df["bathroom"] = pd.to_numeric(df["bathroom"], errors="coerce").fillna(0).astype(int)
        else:
            df["bathroom"] = 0

        if "balcony" in df.columns:
            b = df["balcony"].astype(str)
            b = b.replace(to_replace=r'(?i)no', value='0', regex=True)
            b = b.str.extract(r'(\d+)', expand=False)
            df["balcony"] = pd.to_numeric(b, errors="coerce").fillna(0).astype(int)
        else:
            df["balcony"] = 0

        if "additionalRoom" in df.columns:
            df["additionalRoom"] = df["additionalRoom"].fillna("not available").astype(str).str.lower()
        else:
            df["additionalRoom"] = "not available"

        if "floorNum" in df.columns:
            floor = df["floorNum"].astype(str).str.strip()
            floor = floor.replace({"Ground": "0", "Basement": "-1", "Lower": "0"}, regex=False)
            floor_num = floor.str.extract(r'(-?\d+)', expand=False)
            df["floorNum"] = pd.to_numeric(floor_num, errors="coerce").fillna(0).astype(int)
        else:
            df["floorNum"] = 0

        safe_pps = df["price_per_sqft"].replace({0: np.nan})
        df.insert(loc=4, column="area", value=np.round((df["price"] * 10000000) / safe_pps).astype("Int64"))

        df.insert(loc=1, column="property_type", value="flat")

        return df

    except Exception as e:
        raise ValueError(f"change_data_type failed: {e}") from e


def save_data(data_path: str, df: pd.DataFrame):
    try:
        os.makedirs(data_path, exist_ok=True)
        out_path = os.path.join(data_path, "flat_cleaned.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved cleaned data to: {out_path}")
    except Exception as e:
        raise IOError(f"save_data failed: {e}") from e


def main():
    try:
        in_file = os.path.join("data", "raw", "flats.csv")
        if not os.path.isfile(in_file):
            raise FileNotFoundError(f"Input file not found: {in_file}")

        df = pd.read_csv(in_file)
        df = load_data(df)
        df = clean_function(df)
        df = change_data_type(df)

        data_path = os.path.join("data", "flat_clean")
        save_data(data_path, df)

    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()

