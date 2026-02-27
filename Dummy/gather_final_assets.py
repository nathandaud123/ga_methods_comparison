import os
import shutil
import glob

# Source Directories
SOURCE_IMG_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_polished_v7"
ROOT_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison"

# Destination Directory
DEST_DIR = r"e:\TESIS\01. Dokumen\03. Publikasi Komparasi\ga_method_comparison\final_submission_assets"

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)
    print(f"Created directory: {DEST_DIR}")

# 1. Copy Images from V7
png_files = glob.glob(os.path.join(SOURCE_IMG_DIR, "*.png"))
for f in png_files:
    shutil.copy2(f, DEST_DIR)
    print(f"Copied: {os.path.basename(f)}")

# 2. Copy Key CSVs
csv_files = [
    "top_combinations_summary.csv",
    "analyzed_sample_results.csv"
]

for csv in csv_files:
    src_path = os.path.join(ROOT_DIR, csv)
    if os.path.exists(src_path):
        shutil.copy2(src_path, DEST_DIR)
        print(f"Copied: {csv}")
    else:
        print(f"Warning: {csv} not found in root.")

# 3. Copy Text Draft
draft_md = "results_discussion_draft.md"
src_md = os.path.join(ROOT_DIR, draft_md)
if os.path.exists(src_md):
    shutil.copy2(src_md, DEST_DIR)
    print(f"Copied: {draft_md}")

print("\nAll final assets gathered in:", DEST_DIR)
