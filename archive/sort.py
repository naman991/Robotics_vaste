import os
import shutil

def sort_aitex_defects(base_dir="Defect_images"):
    # Unified mapping based on standard AITEX codes and provided names
    defect_mapping = {
        "002": "Broken_end",
        "006": "Broken_yarn",
        "010": "Broken_pick",
        "016": "Weft_curling",
        "019": "Fuzzyball",
        "022": "Cut_selvage",
        "023": "Crease",
        "025": "Warp_ball",
        "027": "Knots",
        "029": "Contamination",
        "030": "Nep",
        "036": "Weft_crack"
    }
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    # Create target directories
    for folder_name in defect_mapping.values():
        os.makedirs(os.path.join(base_dir, folder_name), exist_ok=True)

    # Process files
    moved_count = 0
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        
        # Ensure we only process files, not directories
        if not os.path.isfile(file_path):
            continue
            
        parts = filename.split('_')
        if len(parts) >= 2:
            code = parts[1] # Extracts '002', '016', etc.
            
            if code in defect_mapping:
                dest_folder = os.path.join(base_dir, defect_mapping[code])
                dest_path = os.path.join(dest_folder, filename)
                
                shutil.move(file_path, dest_path)
                moved_count += 1

    print(f"Successfully organized {moved_count} defect images.")

if __name__ == "__main__":
    sort_aitex_defects()

