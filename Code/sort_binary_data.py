import os
import shutil

# 1. Define your source path (Original Dataset)
source_root = r"C:\Users\99bis\Desktop\SDP\SDP\Datasets\Tea_Betel_Leaf_Final_Dataset(1)\Tea_Betel_Leaf_Final_Dataset"

# 2. Define your destination path (New Stage 1 Binary Dataset)
dest_root = r"C:\Users\99bis\Desktop\SDP\sdp\Stage1_Dataset"

# The 3 splits present in your data folders
splits = ['train', 'validation', 'test']

print("🚀 Commencing automated data aggregation pass...")

for split in splits:
    source_split_dir = os.path.join(source_root, split)
    dest_leaf_dir = os.path.join(dest_root, split, 'Leaf')
    
    # Ensure the target split directory exists
    os.makedirs(dest_leaf_dir, exist_ok=True)
    
    # Check if the source split folder actually exists before scanning
    if not os.path.exists(source_split_dir):
        print(f"⚠️ Warning: Source split folder path not found: {source_split_dir}")
        continue
        
    # Scan all 10 subfolders (disease folders) inside the current split
    disease_folders = os.listdir(source_split_dir)
    counter = 0
    
    for folder in disease_folders:
        folder_path = os.path.join(source_split_dir, folder)
        
        # Make sure it's a directory, not a stray file
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(folder_path, file_name)
                    
                    # Prefix file name with folder name to prevent overwriting duplicates
                    unique_name = f"{folder}_{file_name}"
                    dest_file = os.path.join(dest_leaf_dir, unique_name)
                    
                    # Copy the image file to the unified 'Leaf' repository
                    shutil.copy(src_file, dest_file)
                    counter += 1
                    
    print(f"✅ Synced {counter} total crop images into {split}/Leaf/")

print("\n🎉 Core data grouping pass successfully complete!")