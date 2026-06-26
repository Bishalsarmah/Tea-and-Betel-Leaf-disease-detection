import fiftyone.zoo as foz
import os
import shutil

# 1. Download 800 completely random background/non-leaf images from COCO 
print("Downloading random background vector images from COCO...")
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",  # Validation split is small and fast to download
    max_samples=800,
    shuffle=True
)

# 2. Target your local stage 1 training directory
output_dir = "./Stage1_Data/Not_Leaf"
os.makedirs(output_dir, exist_ok=True)

# 3. Export them cleanly into your local folder as plain JPEGs
print(f"Exporting files cleanly to {output_dir}...")
for i, sample in enumerate(dataset):
    src_path = sample.filepath
    dest_path = os.path.join(output_dir, f"bg_sample_{i}.jpg")
    shutil.copy(src_path, dest_path)

print("Extraction complete. Your 'Not_Leaf' negative class is fully populated!")