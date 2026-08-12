import time
import ujson as json  # drop-in replacement, much faster C-based parser
import pandas as pd

start = time.time()
def load_jsonl_fast(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

df = pd.read_json(r"D:\CodingFiles\PersonalCoding\python\EmberClassification\data\raw\ember_2017_v1\train_features_0.jsonl", lines=True)
print(f"Loaded {len(df)} rows in {time.time()-start:.1f} seconds")
print(df.columns.tolist())
print(df.iloc[0])  # inspect first row structure