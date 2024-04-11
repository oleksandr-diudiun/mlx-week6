import itertools as its
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import Dataset
import numpy as np


class ImagePatchCaptionDataset(Dataset):
    def __init__(
        self, data_dir, batch_indices, caption_indices, patch_size=16, image_size=224
    ):
        """
        Initialize the dataset.
        :param data_dir: Directory containing the .npy batch files.
        :param batch_indices: A mapping from batch ID to list of image IDs.
        :param patch_size: Size of each patch (side length in pixels).
        :param image_size: Size of the images (assuming square images).
        """
        self.data_dir = Path(data_dir) / str(image_size)
        assert self.data_dir.exists(), f"Directory {self.data_dir} does not exist."
        num_images_in_batches = len(
            set(
                [
                    image_id
                    for image_ids in batch_indices.values()
                    for image_id in image_ids
                ]
            )
        )
        self.num_images_with_captions = len(caption_indices)

        assert (
            self.num_images_with_captions == num_images_in_batches
        ), f"Number of images in batch mappings {num_images_in_batches} and number of images with captions {self.num_images_with_captions} do not match"

        self.caption_cycler = {
            image_id: its.cycle(captions)
            for image_id, captions in caption_indices.items()
        }
        self.batch_indices = batch_indices
        self.caption_indices = caption_indices
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches_per_side = self.image_size // self.patch_size
        self.total_patches = self.num_patches_per_side**2

        # Flatten the list of image IDs to calculate total dataset size
        self.image_ids = [
            img_id for batch in batch_indices.values() for img_id in batch
        ]

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return self.num_images_with_captions

    def __getitem__(self, idx):
        """
        Fetch the idx-th image and preprocess it into patches.
        :param idx: Index of the image to fetch.
        """
        # Find the batch file and local index for the global index idx
        # print(idx)
        for batch_id, img_ids in self.batch_indices.items():
            if idx < len(img_ids):
                image_id = img_ids[idx]
                break
            idx -= len(img_ids)
        # print(idx)
        # Memory-map the batch file and access the specific image
        batch_path = self.data_dir / f"{batch_id}.npy"
        # print(f"batch_path: {batch_path}")
        images = np.load(batch_path, mmap_mode="r")
        # print(images.shape)
        image = images[idx]
        caption_tokens: Iterable[list[int]] = self.caption_cycler[image_id]
        # Reshape the image into patches and flatten
        # NOTE: This is numpy array not torch.
        patches = image.reshape(
            self.num_patches_per_side,
            self.patch_size,
            self.num_patches_per_side,
            self.patch_size,
            3,
        )
        # NOTE: This is numpy array not torch.
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(self.total_patches, -1)

        # Convert to a PyTorch tensor
        patches_tensor = torch.tensor(patches, dtype=torch.float32)
        # print(".")
        return image_id, patches_tensor, next(caption_tokens)


def reconstruct_image_from_patches(patches, original_image_size, patch_size):
    """
    Reconstruct the image from its patches for PyTorch tensors.

    Parameters:
    - patches: The tensor of patches with shape (N, P^2C).
    - original_image_size: The height/width of the square image before patching.
    - patch_size: The size of each patch (assuming square patches).

    Returns:
    - A reconstructed image as a PyTorch tensor.
    """
    # Determine the number of patches per side
    num_patches_side = original_image_size // patch_size

    # Reshape patches to (num_patches_side, num_patches_side, patch_size, patch_size, C)
    patches_reshaped = patches.view(
        num_patches_side, num_patches_side, patch_size, patch_size, 3
    )

    # Rearrange the patches into image layout
    # We need to do two permute operations to get to the original layout
    # First, permute to shape (num_patches_side, patch_size, num_patches_side, patch_size, 3)
    patches_permuted = patches_reshaped.permute(0, 2, 1, 3, 4)

    # Now, merge the first two and the last two dimensions to get the final image layout
    image_reconstructed = patches_permuted.contiguous().view(
        original_image_size, original_image_size, 3
    )

    return image_reconstructed
