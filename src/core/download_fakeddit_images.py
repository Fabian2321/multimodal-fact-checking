#!/usr/bin/env python3
"""
Download all images referenced in test_balanced_pairs_clean.csv to data/downloaded_fakeddit_images/.
Overwrites existing files. Prints a summary of successes and failures.
"""
import os
import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test_balanced_pairs_clean.csv')
IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'downloaded_fakeddit_images')

os.makedirs(IMG_DIR, exist_ok=True)

def download_image(url, out_path):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} rows.")
    
    success, fail = 0, 0
    failed_ids = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_url = row.get('image_url', None)
        img_id = row.get('id', None)
        if not img_url or not isinstance(img_url, str) or not img_id or not isinstance(img_id, str):
            fail += 1
            failed_ids.append(img_id)
            continue
        out_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        ok, err = download_image(img_url, out_path)
        if ok:
            success += 1
        else:
            fail += 1
            failed_ids.append(img_id)
            print(f"Failed to download {img_id}: {err}")
    print(f"\nDownload finished.")
    print(f"  Success: {success}")
    print(f"  Failed: {fail}")
    if failed_ids:
        print(f"  Failed IDs: {failed_ids}")

if __name__ == "__main__":
    main() 