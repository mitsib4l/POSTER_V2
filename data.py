import os
import pandas as pd
import numpy as np
from PIL import Image

# Path to csv, relative to where the script is run (inside POSTER_V2)
csv_path = 'data/fer2013/fer2013.csv'
output_dir = 'data/fer2013'

print(f"Loading CSV from: {csv_path}")
df = pd.read_csv(csv_path)

print("Processing pixels to images...")
for idx, row in df.iterrows():
    emotion = str(row['emotion'])
    pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
    usage = row['Usage']
    if usage == 'Training':
        split = 'train'
    elif usage in ['PublicTest', 'Validation']:
        split = 'valid'
    else:
        split = 'test'  # If you want a test split

    out_dir = os.path.join(output_dir, split, emotion)
    os.makedirs(out_dir, exist_ok=True)
    img = Image.fromarray(pixels)
    img = img.convert('RGB')  # Convert to 3-channel if needed
    img.save(os.path.join(out_dir, f'{idx}.png'))

print("Data processing complete.")