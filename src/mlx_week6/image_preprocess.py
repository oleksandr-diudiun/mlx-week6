import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Create the prepared_data directory


def process_and_save_images(df, data_dir, batch_size=100, target_size=224):
    """
    Process a DataFrame of images and save them in batches as .npy files.
    Return the map from batch_id -> image_ids.
    """
    # Placeholder for batch data
    batch_data = []
    batch_indices = defaultdict(
        lambda: list()
    )  # To keep track of image IDs in each batch
    num_written = 0
    for idx, row in tqdm(df.iterrows()):
        batch_id = f"batch_{idx // batch_size}"
        image_id = int(row["img_id"])
        image_data = row["image"]["bytes"]

        # Decode the image
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Resize and maintain aspect ratio
        ratio = target_size / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.BICUBIC)

        # Pad the image to target_size
        padding = (target_size - new_size[0], target_size - new_size[1])
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        padded_image = np.zeros((target_size, target_size, 3), dtype=np.float32)
        padded_image[: new_size[1], : new_size[0], :] = image

        batch_data.append(padded_image)
        batch_indices[batch_id].append(image_id)

        # Save the batch if we've reached batch_size or end of DataFrame
        if len(batch_data) == batch_size or idx == len(df) - 1:
            batch_array = np.array(batch_data, dtype=np.float32)
            folder = data_dir / str(target_size)
            folder.mkdir(parents=True, exist_ok=True)
            batch_filename = folder / f"{batch_id}.npy"

            # Save the batch as a .npy file
            with open(batch_filename, "wb") as f:
                np.save(f, batch_array)

            # Reset for the next batch
            batch_data = []
            num_written += 1
    print(f"Written {num_written} batches")
    return batch_indices


# Assuming df is your DataFrame
# process_and_save_images(df)
