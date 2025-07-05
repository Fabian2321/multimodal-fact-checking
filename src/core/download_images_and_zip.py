import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import zipfile

# Configuration
CSV_PATH = "data/processed/test_balanced_pairs_clean.csv"
IMAGES_DIR = "colab_images"
ZIP_PATH = "colab_images.zip"

# 1. Create directory
os.makedirs(IMAGES_DIR, exist_ok=True)

# 2. Load CSV
df = pd.read_csv(CSV_PATH)

# 3. Download images
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_url = row['image_url']
    img_id = row['id']
    ext = os.path.splitext(img_url.split("?")[0])[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"
    img_path = os.path.join(IMAGES_DIR, f"{img_id}{ext}")
    if os.path.exists(img_path):
        continue
    try:
        resp = requests.get(img_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(img_path)
    except Exception as e:
        # Optional: Save dummy image so all IDs are covered
        Image.new("RGB", (224, 224), color="gray").save(img_path)
        print(f"Warning: Image for {img_id} could not be loaded: {e}")

# 4. Create ZIP archive
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for fname in os.listdir(IMAGES_DIR):
        zipf.write(os.path.join(IMAGES_DIR, fname), arcname=fname)

print(f"All images saved as ZIP: {ZIP_PATH}") 