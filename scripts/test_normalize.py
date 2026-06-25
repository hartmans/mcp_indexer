#!/usr/bin/env python3
"""Test normalize_metadata with numpy arrays."""

import numpy as np
import json

# Test with empty numpy array (like keywords column)
empty_array = np.array([], dtype=object)
print(f"empty_array: {empty_array}")
print(f"len(empty_array): {len(empty_array)}")

# Test the normalize_metadata logic
def normalize_metadata(metadata):
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    # Handle numpy arrays (empty arrays should become empty dict)
    if hasattr(metadata, '__len__') and hasattr(metadata, 'item'):
        try:
            if len(metadata) == 0:
                return {}
        except Exception as e:
            print(f"Error in len check: {e}")
            return {}
    try:
        result = json.loads(metadata)
        if not isinstance(result, dict):
            return {}
        return result
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error in json.loads: {e}")
        return {}

print(f"Result: {normalize_metadata(empty_array)}")

# Now test with actual DataFrame iteration
import lancedb
db = lancedb.connect('/tmp/test_lancedb_dump')
meta_table = db.open_table('test_col_meta')
df = meta_table.to_pandas()

print("\nIterating:")
for i, row in df.iterrows():
    val = row['keywords']
    print(f"Row {i} keywords type: {type(val)}")
    print(f"hasattr '__len__': {hasattr(val, '__len__')}")
    print(f"hasattr 'item': {hasattr(val, 'item')}")
    try:
        print(f"len(val): {len(val)}")
    except Exception as e:
        print(f"len error: {e}")