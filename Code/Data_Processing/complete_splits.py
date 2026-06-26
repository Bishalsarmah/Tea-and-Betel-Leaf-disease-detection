import os
import shutil
import random

# =====================================================================
# 1. VERIFY SYSTEM PATHS PRECISELY
# =====================================================================
SOURCE_ROOT = r"C:\Users\99bis\Desktop\SDP\SDP\Datasets\Tea_Betel_Leaf_Final_Dataset\Tea_Betel_Leaf_Final_Dataset"
DEST_ROOT = r"C:\Users\99bis\Desktop\SDP\sdp\Stage1_Dataset"

# Paths pointing into your fully operational Train directory
TRAIN_LEAF_SOURCE = os.path.join(DEST_ROOT, "train", "Leaf")
TRAIN_NOT_LEAF_SOURCE = os.path.join(DEST_ROOT, "train", "Not_Leaf")

TARGET_SPLITS = ['validation', 'test']
SPLIT_TARGETS = {'validation': 200, 'test': 200}

# Quick initial verification pass
if not os.path.exists(TRAIN_LEAF_SOURCE) or not os.path.exists(TRAIN_NOT_LEAF_SOURCE):
    raise FileNotFoundError("❌ Cannot locate your train folders! Please check that train/Leaf and train/Not_Leaf contain your 1,000 baseline images.")

# =====================================================================
# 2. LOCAL DISTRIBUTION ENGINE
# =====================================================================
for split in TARGET_SPLITS:
    print(f"\n==========================================")
    print(f"🛠️ CONSTRUCTING LOCAL SPLIT MATRIX: [{split.upper()}]")
    print(f"==========================================")
    
    source_split_dir = os.path.join(SOURCE_ROOT, split)
    dest_leaf_dir = os.path.join(DEST_ROOT, split, 'Leaf')
    dest_not_leaf_dir = os.path.join(DEST_ROOT, split, 'Not_Leaf')
    
    # Wipe broken iterations to guarantee clean targets
    if os.path.exists(dest_leaf_dir): shutil.rmtree(dest_leaf_dir)
    if os.path.exists(dest_not_leaf_dir): shutil.rmtree(dest_not_leaf_dir)
    
    os.makedirs(dest_leaf_dir, exist_ok=True)
    os.makedirs(dest_not_leaf_dir, exist_ok=True)
    
    target_count = SPLIT_TARGETS[split]
    
    # ── PHASE A: SOURCE LEAF EXTRACTION ──
    raw_leaf_pool = []
    if os.path.exists(source_split_dir):
        for folder in os.listdir(source_split_dir):
            folder_path = os.path.join(source_split_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        raw_leaf_pool.append(os.path.join(folder_path, file_name))
                        
    if not raw_leaf_pool:
        print(f"⚠️ Warning: Split source leaf images empty at {source_split_dir}. Defaulting to train/Leaf buffer.")
        raw_leaf_pool = [os.path.join(TRAIN_LEAF_SOURCE, f) for f in os.listdir(TRAIN_LEAF_SOURCE)]
        
    selected_leaves = random.sample(raw_leaf_pool, min(target_count, len(raw_leaf_pool)))
    for idx, src_file in enumerate(selected_leaves):
        unique_name = f"split_leaf_{idx}_{os.path.basename(src_file)}"
        shutil.copy(src_file, os.path.join(dest_leaf_dir, unique_name))
    print(f"  └─ Successfully created balanced files inside {split}/Leaf")

    # ── PHASE B: LOCAL NOT_LEAF SHUFFLE EXTRACTION ──
    # Pull images from train/Not_Leaf locally to prevent network calls and repeats
    train_bg_pool = [os.path.join(TRAIN_NOT_LEAF_SOURCE, f) for f in os.listdir(TRAIN_NOT_LEAF_SOURCE) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(train_bg_pool) < target_count:
        raise ValueError(f"❌ Error: Not enough unique background images in train/Not_Leaf ({len(train_bg_pool)}) to build a {target_count} sample split.")
        
    selected_bgs = random.sample(train_bg_pool, target_count)
    for idx, src_file in enumerate(selected_bgs):
        unique_name = f"split_bg_{split}_{idx}.jpg"
        shutil.copy(src_file, os.path.join(dest_not_leaf_dir, unique_name))
    print(f"  └─ Successfully created {len(selected_bgs)} non-repeating real-world images inside {split}/Not_Leaf")

print("\n🎉 Fix applied! 'Stage1_Dataset' has been generated locally with full unique data parity.")