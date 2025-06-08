import os
import random
from PIL import Image
import torchvision.transforms as transforms

def augment_images_for_class(class_dir, threshold, num_augmentations_per_image=3):
    """
    For a given class directory, if the number of images is below 'threshold',
    generate augmented images until the total count reaches the threshold.
    Each augmented image will have '_augment_<i>' appended to its base filename.
    """
    images = [f for f in os.listdir(class_dir) 
              if f.lower().endswith(('.jpg'))]
    current_count = len(images)
    
    if current_count >= threshold:
        print(f"{class_dir}: {current_count} images (meets threshold).")
        return
    
    # Calculate how many new images to generate
    num_to_generate = threshold - current_count
    print(f"{class_dir}: {current_count} images found. Generating {num_to_generate} augmented images.")

    # Define augmentation transform
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0))
    ])
    
    output_class_dir = class_dir.replace("split_data/Train", "data_augment_2")
    os.makedirs(output_class_dir, exist_ok=True)
    
    generated = 0
    # To generate enough images, loop until we've created the required number.
    while generated < num_to_generate:
        # Pick a random image from the class directory
        img_file = random.choice(images)
        img_path = os.path.join(class_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        # Generate one or more augmented variants from the image.
        for i in range(num_augmentations_per_image):
            if generated >= num_to_generate:
                break
            aug_image = aug_transform(image)
            # Create a new filename: original basename + _augment_<generated> + extension
            base, ext = os.path.splitext(img_file)
            new_filename = f"{base}_augment_{generated}{ext}"
            new_path = os.path.join(output_class_dir, new_filename)
            try:
                aug_image.save(new_path)
            except Exception as e:
                print(f"Error saving {new_path}: {e}")
                continue
            generated += 1

def augment_dataset(train_dir, threshold, num_augmentations_per_image=3):
    """
    Traverse each class subdirectory in train_dir and augment images if needed.
    """
    # List subdirectories (each corresponding to a class)
    class_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir)
                  if os.path.isdir(os.path.join(train_dir, d))]
    for class_dir in class_dirs:
        augment_images_for_class(class_dir, threshold, num_augmentations_per_image)

if __name__ == '__main__':
    
    train_dir = 'split_data/Train'
    augment_dataset(train_dir, threshold=7500, num_augmentations_per_image=3)
