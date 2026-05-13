"""Fetch Primeira Liga match data from football-data.co.uk and cache it locally."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODE = "P1"  # Primeira Liga

# Season codes are last-two-digits-of-each-year concatenated.
# "2122" = 2021-22 season, "2526" = 2025-26 season, etc.
SEASONS = ["2122", "2223", "2324", "2425", "2526"]


def fetch_season(season: str) -> pd.DataFrame:
    """Download one season's CSV and tag rows with the season code."""
    url = f"{BASE_URL}/{season}/{LEAGUE_CODE}.csv"
    print(f"  Fetching {season} from {url}")
    df = pd.read_csv(url)
    df = df.assign(Season=season)
    return df


def main():
    print(f"Fetching Primeira Liga seasons: {SEASONS}")
    frames = [fetch_season(s) for s in SEASONS]
    full = pd.concat(frames, ignore_index=True)

    # Date column comes as strings like "15/08/21" — parse to real dates.
    # dayfirst=True because UK source uses DD/MM/YY.
    full["Date"] = pd.to_datetime(full["Date"], dayfirst=True, errors="coerce")

    # Drop rows with no result (occasionally there are header artifacts or
    # postponed matches that haven't been replayed yet).
    full = full.dropna(subset=["FTHG", "FTAG"])

    output_path = DATA_DIR / "matches.parquet"
    full.to_parquet(output_path)

    print(f"\nSaved {len(full)} matches to {output_path}")
    print(f"\nSeasons covered:")
    print(full.groupby("Season").size())
    print(f"\nColumns available ({len(full.columns)}):")
    print(full.columns.tolist())
    print(f"\nFirst few rows:")
    print(full[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].head())


if __name__ == "__main__":
    main()