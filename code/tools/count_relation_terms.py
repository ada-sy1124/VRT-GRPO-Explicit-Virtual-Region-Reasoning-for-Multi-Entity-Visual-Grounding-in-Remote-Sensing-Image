"""Stream Visual Genome relationships and summarize frequent spatial predicates."""

import json
from collections import Counter
from tqdm import tqdm
import ijson

# ==========================================
# Paths
# ==========================================
RELATIONSHIPS_PATH = "/root/autodl-tmp/SPIN/Visual_Genome_Data/relationships.json"
OUTPUT_STATS_JSON = "/root/autodl-tmp/SPIN/Visual_Genome_Data/predicate_frequencies.json"
OUTPUT_SPATIAL_TXT = "/root/autodl-tmp/SPIN/Visual_Genome_Data/spatial_predicates_filtered.txt"

# Spatial predicate roots used for filtering.
SPATIAL_ROOTS = ['left', 'right', 'above', 'below', 'under', 'over', 'top', 'bottom', 'front', 'behind']

def run_eda():
    print("Starting full relationship predicate frequency scan...")
    predicate_counter = Counter()
    
    # 1. Full streaming count.
    with open(RELATIONSHIPS_PATH, 'r', encoding='utf-8') as f:
        for img_rels in tqdm(ijson.items(f, 'item'), desc="scanning relationships"):
            for rel in img_rels.get('relationships', []):
                # Normalize predicates before counting.
                pred = str(rel.get('predicate', '')).strip().lower()
                if pred:
                    predicate_counter[pred] += 1
                    
    # Sort by frequency.
    sorted_preds = predicate_counter.most_common()
    total_unique_preds = len(sorted_preds)
    total_rel_count = sum(predicate_counter.values())
    
    print("\n" + "="*50)
    print("Visual Genome predicate frequency summary")
    print("="*50)
    print(f"total relationship edges: {total_rel_count:,}")
    print(f"unique predicate strings: {total_unique_preds:,}")
    
    # 2. Save the full frequency table.
    with open(OUTPUT_STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted_preds), f, indent=4, ensure_ascii=False)
    print(f"Saved full frequency table to: {OUTPUT_STATS_JSON}")

    # 3. Extract candidate spatial predicates.
    spatial_candidates = []
    print("\nExtracting candidate spatial predicates...")
    for pred, count in sorted_preds:
        # Keep predicates that occur often enough to avoid one-off typos.
        if count >= 5 and any(root in pred for root in SPATIAL_ROOTS):
            spatial_candidates.append(f"{count:7d} | {pred}")

    with open(OUTPUT_SPATIAL_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(spatial_candidates))
        
    print(f"Saved {len(spatial_candidates)} candidate spatial predicates to: {OUTPUT_SPATIAL_TXT}")
    print("\nTop candidate spatial predicates:")
    for line in spatial_candidates[:]:
        print(line)
    print("="*50)

if __name__ == "__main__":
    run_eda()


# python code/tools/count_relation_terms.py
