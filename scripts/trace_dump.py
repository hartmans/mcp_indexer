#!/usr/bin/env python3
"""Full trace of dump_table_to_dict."""

import lancedb
import pandas as pd

db = lancedb.connect('/tmp/test_lancedb_dump')
meta_table = db.open_table('test_col_meta')

# Step by step
print("1. count_rows:", meta_table.count_rows())

print("2. to_pandas...")
try:
    df = meta_table.to_pandas()
    print("   Success! Shape:", df.shape)
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("3. Iterating rows...")
try:
    for i, row in df.iterrows():
        print(f"   Row {i}:")
        for col in df.columns:
            val = row[col]
            # Check embedding
            if col == 'embedding':
                if val is None or (hasattr(val, '__len__') and len(val) == 0):
                    result = None
                elif hasattr(val, 'tolist'):
                    result = val.tolist()
                elif hasattr(val, '__iter__') and not isinstance(val, str):
                    result = list(val)
                else:
                    result = [float(x) for x in str(val).split(',')]
                print(f"     {col}: {type(result)} with {len(result)} elements")
            # Check keywords (metadata_str test)
            elif col == 'keywords':
                if val is None or (hasattr(val, '__len__') and len(val) == 0):
                    result = {}
                elif isinstance(val, dict):
                    result = val
                else:
                    result = json.loads(val) if isinstance(val, str) else {}
                print(f"     {col}: {result}")
            else:
                print(f"     {col}: {type(val)}")
except Exception as e:
    print(f"   Error during iteration: {e}")
    import traceback
    traceback.print_exc()