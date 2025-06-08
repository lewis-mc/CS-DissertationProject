import os
import argparse
from torchvision import datasets, transforms
from PIL import Image

# Define a SafeImageFolder that catches errors and prints the file path.
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (OSError, ValueError) as e:
            file_path = self.samples[index][0]
            print(f"Warning: Skipping corrupted image at index {index} ({file_path}): {e}")
            return None

def main():

    # Define a simple transform that only loads the image.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    data_dir = '../../split_data/Evaluate'

    # Create the dataset using our SafeImageFolder
    dataset = SafeImageFolder(data_dir, transform=transform)
    print(f"Found {len(dataset.samples)} images in the dataset.")

    corrupted_files = []
    # Iterate through the dataset. If __getitem__ returns None, it means the image is corrupted.
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            # File path was printed in __getitem__; also collect it here.
            file_path = dataset.samples[idx][0]
            corrupted_files.append(file_path)
        if idx % 5000 == 0:
            print(f"Processed {idx} images...")

    print(f"\nTotal corrupted images found: {len(corrupted_files)}")
    if corrupted_files:
        print("List of corrupted files:")
        for f in corrupted_files:
            print(f)

if __name__ == '__main__':
    main()
