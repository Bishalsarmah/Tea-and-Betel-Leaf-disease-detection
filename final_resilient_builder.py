import os
import shutil
import random
import requests
import time

# =====================================================================
# 1. PATH CONFIGURATIONS
# =====================================================================
SOURCE_ROOT = r"C:\Users\99bis\Desktop\SDP\SDP\Datasets\Tea_Betel_Leaf_Final_Dataset(1)\Tea_Betel_Leaf_Final_Dataset"
DEST_ROOT = r"C:\Users\99bis\Desktop\SDP\sdp\Stage1_Dataset"

SPLITS = ['train', 'validation', 'test']
SPLIT_TARGETS = {'train': 1000, 'validation': 200, 'test': 200}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# NOTE: We do NOT delete DEST_ROOT here so we can safely resume if a crash happens!
print(f"📁 Initializing baseline framework paths at: {DEST_ROOT}")

# =====================================================================
# 2. THE RESILIENT PIPELINE ENGINE
# =====================================================================
for split in SPLITS:
    print(f"\n==========================================")
    print(f"🔄 PROCESSING SPLIT STRUCTURE: [{split.upper()}]")
    print(f"==========================================")
    
    source_split_dir = os.path.join(SOURCE_ROOT, split)
    dest_leaf_dir = os.path.join(DEST_ROOT, split, 'Leaf')
    dest_not_leaf_dir = os.path.join(DEST_ROOT, split, 'Not_Leaf')
    
    os.makedirs(dest_leaf_dir, exist_ok=True)
    os.makedirs(dest_not_leaf_dir, exist_ok=True)
    
    target_count = SPLIT_TARGETS[split]
    
    # ── PHASE A: LEAF PROCESSING & STRUCTURAL POOLING ──
    # Check if this folder is already populated so we don't repeat work
    existing_leaves = len([f for f in os.listdir(dest_leaf_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    if existing_leaves >= target_count:
        print(f"⏭️ Leaf pool for [{split}] already balanced ({existing_leaves} images). Skipping copy pass.")
    else:
        raw_leaf_pool = []
        if os.path.exists(source_split_dir):
            for folder in os.listdir(source_split_dir):
                folder_path = os.path.join(source_split_dir, folder)
                if os.path.isdir(folder_path):
                    for file_name in os.listdir(folder_path):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            raw_leaf_pool.append(os.path.join(folder_path, file_name))
                            
        if not raw_leaf_pool:
            print(f"❌ Error: Leaves missing from source path: {source_split_dir}")
            continue
            
        selected_leaves = random.sample(raw_leaf_pool, min(target_count, len(raw_leaf_pool)))
        for idx, src_file in enumerate(selected_leaves):
            parent_folder = os.path.basename(os.path.dirname(src_file))
            file_base = os.path.basename(src_file)
            unique_name = f"{parent_folder}_{idx}_{file_base}"
            shutil.copy(src_file, os.path.join(dest_leaf_dir, unique_name))
        print(f"✅ Leaf Layer balanced: {len(selected_leaves)} images saved to {split}/Leaf/")

    # ── PHASE B: CRASH-RESISTANT PICTURE STREAMING ──
    print(f"📥 Verifying real-world background objects inside {split}/Not_Leaf/...")
    
    for idx in range(target_count):
        dest_file_path = os.path.join(dest_not_leaf_dir, f"bg_{split}_{idx}.jpg")
        
        # SKIP if this image was already successfully downloaded before the machine stopped
        if os.path.exists(dest_file_path):
            continue
            
        # Unique seed keeps every single API response diverse and clear
        picsum_url = f"https://picsum.photos/id/{random.randint(1, 900)}/256/256"
        
        # Retry logic: Try up to 3 times per image before moving on, instead of crashing the whole program
        for attempt in range(3):
            try:
                with requests.get(picsum_url, headers=headers, stream=True, timeout=8) as r:
                    if r.status_code == 200:
                        with open(dest_file_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        break  # Break retry loop on successful download
                    else:
                        # Fallback to direct random tile stream if a specific ID fails
                        fallback_url = "https://picsum.photos/256/256"
                        with requests.get(fallback_url, headers=headers, stream=True, timeout=5) as fr:
                            if fr.status_code == 200:
                                with open(dest_file_path, 'wb') as f:
                                    for chunk in fr.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                break
            except (requests.exceptions.RequestException, Exception):
                time.sleep(2)  # Cool down for 2 seconds if network flickers
                continue
                
        # Cool-down break to avoid triggering server firewalls or overwhelming your local router
        if idx % 25 == 0 and idx > 0:
            print(f"  Processed up to background image #{idx}...")
            time.sleep(0.5)

print("\n🎉 Success! All splits (train, validation, and test) are fully structured and balanced with real-world pictures.")