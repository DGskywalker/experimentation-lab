"""Download the public datasets (no API keys) and load them into a local DuckDB file.

    python scripts/fetch_data.py            # everything (Criteo is a 311 MB download)
    python scripts/fetch_data.py --no-criteo
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import duckdb

DATA = Path(__file__).resolve().parents[1] / "data"

SOURCES = {
    # Cookie Cats mobile-game experiment (originally from DataCamp / Kaggle)
    "cookie_cats.csv": "https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv",
    # Hotel Booking Demand (Antonio, Almeida & Nunes, 2019) via TidyTuesday
    "hotels.csv": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
    # Marketing A/B testing: ads vs public-service announcements (Kaggle, mirrored on GitHub)
    "marketing_AB.csv": "https://raw.githubusercontent.com/Shoyo0104/marketing-ab-test-kaggle-dataset/main/data/marketing_AB.csv",
    # Kevin Hillstrom's MineThatData e-mail experiment (2008)
    "hillstrom.csv": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
}
CRITEO = (
    "criteo-research-uplift-v2.1.csv.gz",
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
)
CRITEO_SAMPLE_ROWS = 1_000_000


def main(with_criteo: bool = True) -> None:
    DATA.mkdir(exist_ok=True)
    sources = dict(SOURCES)
    if with_criteo:
        sources[CRITEO[0]] = CRITEO[1]
    for name, url in sources.items():
        dest = DATA / name
        if not dest.exists():
            print(f"downloading {name}")
            urlretrieve(url, dest)

    con = duckdb.connect(str(DATA / "lab.duckdb"))
    con.execute(f"CREATE OR REPLACE TABLE cookie_cats AS SELECT * FROM read_csv_auto('{DATA / 'cookie_cats.csv'}')")
    con.execute(f"CREATE OR REPLACE TABLE hotels AS SELECT * FROM read_csv_auto('{DATA / 'hotels.csv'}', nullstr=['NA', 'NULL'])")
    con.execute(f"""
        CREATE OR REPLACE TABLE marketing AS
        SELECT user_id, test_group, converted::INT AS converted, total_ads, most_ads_day, most_ads_hour
        FROM read_csv_auto('{DATA / 'marketing_AB.csv'}')
    """)
    con.execute(f"CREATE OR REPLACE TABLE hillstrom AS SELECT * FROM read_csv_auto('{DATA / 'hillstrom.csv'}')")
    tables = ["cookie_cats", "hotels", "marketing", "hillstrom"]
    if with_criteo:
        # A reproducible 1M-row simple random sample of the 13.9M-row file
        con.execute(f"""
            CREATE OR REPLACE TABLE criteo AS
            SELECT * FROM read_csv_auto('{DATA / CRITEO[0]}')
            USING SAMPLE reservoir({CRITEO_SAMPLE_ROWS} ROWS) REPEATABLE (42)
        """)
        tables.append("criteo")
    for table in tables:
        print(table, con.execute(f"SELECT count(*) FROM {table}").fetchone()[0], "rows")
    con.close()


if __name__ == "__main__":
    main(with_criteo="--no-criteo" not in sys.argv)
