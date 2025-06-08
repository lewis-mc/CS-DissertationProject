from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config
import os

def data_prepare():

    train_dataset, eval_dataset = handle_corrupted_images()
    return train_dataset, eval_dataset

def handle_corrupted_images():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom Dataset class to handle corrupted images
    class SafeImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super(SafeImageFolder, self).__getitem__(index)
            except (OSError, ValueError) as e:
                print(f"Warning: Skipping corrupted image at index {index}: {e}")
                return None

    train_dataset = SafeImageFolder(root=config.train_dir, transform=transform)
    eval_dataset = SafeImageFolder(root=config.eval_dir, transform=transform)

    # Filter out None items resulting from corrupted images
    train_dataset.samples = [s for s in train_dataset.samples if os.path.exists(s[0])]
    eval_dataset.samples = [s for s in eval_dataset.samples if os.path.exists(s[0])]

    return train_dataset, eval_dataset
