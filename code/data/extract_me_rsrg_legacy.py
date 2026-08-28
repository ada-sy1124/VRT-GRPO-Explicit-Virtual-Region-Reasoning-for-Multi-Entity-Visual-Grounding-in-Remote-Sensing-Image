"""Legacy ME-RSRG extraction helper retained for reproducibility.

Use code/data/extract_me_rsrg.py for the current cleaned JSONL format."""

import os
import json
import re
import zipfile
from tqdm import tqdm

# ==========================================
# 1. Global config
# ==========================================
ZIP_PATH = "/root/autodl-tmp/model_cache/huggingface/hub/datasets--AlleyOop26--ME-RSRG/snapshots/09f1998517aa7f0382f72d535e84b7bbafc7b8db/ME-RSRG_datasets.zip"
TARGET_JSON_FILE = "RSRG_ME_datasets_ori/train.json"
OUTPUT_CLEAN_JSONL = "/root/autodl-tmp/SPIN/ME-RSRG-Data/train_cleaned_stage1.jsonl"

def direct_extract_and_clean():
    print(f"Reading JSONL data from archive member: {TARGET_JSON_FILE} ...")
    
    cleaned_data = []
    error_count = 0
    
    # Read the archive member line by line.
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        with zip_ref.open(TARGET_JSON_FILE) as f:
            lines = f.readlines()
            print(f"Loaded {len(lines)} raw training rows.")
            
            print("Starting structured extraction...")
            for idx, line in enumerate(tqdm(lines)):
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    error_count += 1
                    continue

                # --------------------------------------------------
                # 1. Extract image id.
                # --------------------------------------------------
                image_id = f"img_{idx}"
                if 'images' in item and len(item['images']) > 0:
                    # Example: "./data/RSRG_ME_datasets_ori/opt_rsvg/images/005508.jpg"
                    image_id = item['images'][0].split('/')[-1].split('.')[0]

                # --------------------------------------------------
                # 2. Extract and clean the prompt.
                # --------------------------------------------------
                user_text = ""
                for msg in item.get('messages', []):
                    if msg.get('role') == 'user':
                        user_text = msg.get('content', '')
                        break
                        
                prompt_match = re.search(r'description:\s*(.*?)\.\s*Generate', user_text, re.IGNORECASE)
                if prompt_match:
                    clean_instruction = prompt_match.group(1).strip()
                else:
                    clean_instruction = user_text.split("Generate")[0].replace("Localize the subject and objects in the image with description:", "").strip()

                # --------------------------------------------------
                # 3. Extract boxes from structured fields.
                # --------------------------------------------------
                gt_subject_box = None
                gt_anchor_boxes = []
                
                if 'objects' in item and 'ref' in item['objects'] and 'bbox' in item['objects']:
                    refs = item['objects']['ref']
                    bboxes = item['objects']['bbox']
                    
                    for i, ref_name in enumerate(refs):
                        if i < len(bboxes):
                            if ref_name == 'subject':
                                gt_subject_box = bboxes[i]
                            else:
                                gt_anchor_boxes.append(bboxes[i])

                # --------------------------------------------------
                # Assemble the output row.
                # --------------------------------------------------
                if not gt_subject_box:
                    error_count += 1
                    continue 

                clean_item = {
                    "image_id": image_id,
                    "instruction": clean_instruction, 
                    "gt_subject_box": gt_subject_box, 
                    "gt_anchor_boxes": gt_anchor_boxes 
                }
                cleaned_data.append(clean_item)

    print(f"\nExtraction completed: saved {len(cleaned_data)} rows, skipped {error_count} rows with missing data.")
    
    os.makedirs(os.path.dirname(OUTPUT_CLEAN_JSONL), exist_ok=True)
    with open(OUTPUT_CLEAN_JSONL, 'w', encoding='utf-8') as f:
        for data in cleaned_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print(f"Saved cleaned data to: {OUTPUT_CLEAN_JSONL}")
    
    if len(cleaned_data) > 0:
        print("\nFirst cleaned row:")
        print(json.dumps(cleaned_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    direct_extract_and_clean()




# python code/data/extract_me_rsrg_legacy.py
