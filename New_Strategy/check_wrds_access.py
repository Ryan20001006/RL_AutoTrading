"""
Quick diagnostic — check dsp500list columns before running full pipeline.
  python3.11 New_Strategy/check_wrds_access.py
"""
import wrds
db = wrds.Connection()

print("\n=== crsp.dsp500list — first 3 rows ===")
sample = db.raw_sql("SELECT * FROM crsp.dsp500list LIMIT 3")
print(sample.to_string())
print(f"\nColumns: {list(sample.columns)}")

print("\n=== crsp.dsf — first 3 rows (price/market cap table) ===")
sample2 = db.raw_sql("""
    SELECT * FROM crsp.dsf
    WHERE date = '2020-01-02'
    LIMIT 3
""")
print(sample2[["permno", "permco", "date", "prc", "shrout"]].to_string())
print(f"\nColumns: {list(sample2.columns)}")

db.close()
