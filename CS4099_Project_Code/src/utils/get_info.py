import os
import matplotlib.pyplot as plt
import numpy as np


# Define the top-level training directory
train_dir = os.path.join('split_data', 'Train')

# Get a list of class directories (subdirectories within Train)
class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

counts = {}
for cls in class_names:
    class_path = os.path.join(train_dir, cls)
    # List all files (ignore subdirectories)
    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    counts[cls] = len(files)

sorted_classes = sorted(counts.keys())
sorted_counts = [counts[cls] for cls in sorted_classes]

# Plotting the results in a bar chart
plt.figure(figsize=(12, 6))
plt.bar(sorted_classes, sorted_counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class in Training Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('dataset_info.png')

# Directories
train_dir = os.path.join('split_data', 'Train')
augment_dir = os.path.join('data_augment_2')

# Get the set of classes from both directories.
classes_train = {d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))}
classes_augment = {d for d in os.listdir(augment_dir) if os.path.isdir(os.path.join(augment_dir, d))}
all_classes = sorted(classes_train.union(classes_augment))

# Count images for each class.
original_counts = {}
augment_counts = {}

for cls in all_classes:
    # Count files in the training folder for this class.
    cls_train_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_train_path):
        original_counts[cls] = len([f for f in os.listdir(cls_train_path)
                                      if os.path.isfile(os.path.join(cls_train_path, f))])
    else:
        original_counts[cls] = 0
    # Count files in the data augmentation folder for this class.
    cls_aug_path = os.path.join(augment_dir, cls)
    if os.path.isdir(cls_aug_path):
        augment_counts[cls] = len([f for f in os.listdir(cls_aug_path)
                                   if os.path.isfile(os.path.join(cls_aug_path, f))])
    else:
        augment_counts[cls] = 0

# Prepare data for plotting.
x = np.arange(len(all_classes))
orig = [original_counts[cls] for cls in all_classes]
aug = [augment_counts[cls] for cls in all_classes]



# Plot stacked bar chart.
plt.figure(figsize=(12, 6))
plt.bar(x, orig, color='skyblue', label='Original (split_data/Train)')
plt.bar(x, aug, bottom=orig, color='orange', label='Augmented (data_augment)')
plt.xticks(x, all_classes, rotation=45)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Number of Images per Class in Training Data (after augmentation)")
plt.legend()
plt.tight_layout()


# Save the plot if desired.
output_dir = 'test5'
graphs_dir = os.path.join(output_dir, 'graphs')
os.makedirs(graphs_dir, exist_ok=True)
plt.savefig(os.path.join(graphs_dir, 'new_training_data_counts.png'))

