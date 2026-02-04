import pandas as pd
import json
import inflection
import duckdb

def parse_event_body(value, path=["el_eb"], delim="."):
    res = {}
    if isinstance(value, dict):
        for k, v in value.items():
            res.update(parse_event_body(value=v, path=path + [k], delim=delim))
    elif isinstance(value, list):
        for k, v in enumerate(value):
            res.update(parse_event_body(value=v, path=path + [str(k)], delim=delim))
    else:
        res[inflection.underscore(delim.join(path))] = value
    return res

if __name__ == "__main__":
    # df = pd.read_parquet()
    with duckdb.connect(database = "data/sync.duckdb", read_only=True) as conn:
        df = conn.sql("SELECT * FROM payby_anti_fraud_train_20241101_20250707;").fetchdf()
    # el_eb_feature_df = df["ev"]
    # print(df)
