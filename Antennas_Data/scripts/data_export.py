"""Simple exporter placeholder.

This script shows how to load exported CSV results from CST (or other EM tools) and combine
them into a single dataset for ML training. If you have CST Python automation available,
we can extend this to call CST and export results automatically.
"""
import os
import pandas as pd

def collect_csvs(src_dir, out_csv):
    frames = []
    for fn in os.listdir(src_dir):
        if fn.lower().endswith('.csv'):
            path = os.path.join(src_dir, fn)
            try:
                df = pd.read_csv(path)
                df['source_file'] = fn
                frames.append(df)
            except Exception as e:
                print(f"Skipping {fn}: {e}")
    if not frames:
        print('No CSV files found in', src_dir)
        return
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_csv, index=False)
    print('Saved combined dataset to', out_csv)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('src', help='directory containing CSV exports')
    p.add_argument('out', help='output CSV file path')
    args = p.parse_args()
    collect_csvs(args.src, args.out)
