import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class SpatioTemporalDataset(Dataset):
    def __init__(self, frame_dir, sequence_length=5, transform=None):
        """
        Args:
            frame_dir (str): Path to the directory containing chronological video frames/images.
            sequence_length (int): Number of consecutive frames to group into a single sample window.
            transform (callable, optional): Optional geometric or color transforms (e.g., Albumentations or torchvision).
        """
        self.frame_dir = frame_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Gather all image files and ensure they are sorted chronologically
        self.frame_files = sorted(
            [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if len(self.frame_files) == 0:
            raise RuntimeError(
                f"Found 0 frames in {frame_dir}. Ensure your preprocessing script extracts frames here."
            )

    def __len__(self):
        # The number of valid sliding windows we can extract from the total sequence length
        return len(self.frame_files) - self.sequence_length + 1

    def __getitem__(self, idx):
        """
        Returns:
            A 4D Tensor representing a sequence of frames: (Sequence_Length, Channels, Height, Width)
        """
        # Slice out a sequential window of frame paths
        window_paths = self.frame_files[idx : idx + self.sequence_length]

        sequence_frames = []
        for path in window_paths:
            # Read image in BGR format
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to load frame at: {path}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply augmentations if defined (passed down from training config)
            if self.transform:
                # Assuming standard torchvision transforms or basic numpy-to-tensor operations
                img = self.transform(img)
            else:
                # Fallback basic tensor conversion: Convert HWC to CHW and scale pixels to [0, 1]
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            sequence_frames.append(img)

        # Stack frames along a new temporal dimension: (Sequence_Length, Channels, Height, Width)
        return torch.stack(sequence_frames, dim=0)


def get_spatio_temporal_dataloader(
    frame_dir, batch_size=4, sequence_length=5, shuffle=True, num_workers=2
):
    """
    Factory function to easily build a data streaming worker pool.
    """
    dataset = SpatioTemporalDataset(
        frame_dir=frame_dir, sequence_length=sequence_length
    )

    # pin_memory=True speeds up tensor transfers from Host CPU to the M4's Unified Memory
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("DATA PIPELINE SANITY TEST")
    print("=" * 40)

    # Create a quick dummy mock directory structure to simulate video frames
    mock_dir = "ml_pipeline/data/mock_frames"
    os.makedirs(mock_dir, exist_ok=True)

    # Save 10 black placeholder images using OpenCV to verify code logic
    import numpy as np

    for i in range(10):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(mock_dir, f"frame_{i:04d}.jpg"), dummy_frame)
    print(f"Generated 10 mock video frames in: {mock_dir}")

    try:
        # Build dataloader with batch_size=2 and a sliding window of 5 frames
        loader = get_spatio_temporal_dataloader(
            frame_dir=mock_dir, batch_size=2, sequence_length=5, shuffle=False
        )

        # Fetch the very first batch
        first_batch = next(iter(loader))

        print(f"Successfully loaded a batch!")
        print(f"Batch Tensor Shape: {first_batch.shape}")
        print(" -> Shape Mapping: (BatchSize, SequenceLength, Channels, Height, Width)")
        print("Data pipeline integration looks production-ready!")

    finally:
        # Clean up mock files to keep the repo tidy after the sanity check
        for f in os.listdir(mock_dir):
            os.remove(os.path.join(mock_dir, f))
        os.rmdir(mock_dir)
        print("Cleaned up mock frame directories.")
        print("=" * 40 + "\n")
