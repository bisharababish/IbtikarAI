"""Check zip file contents"""
import zipfile
import os
import sys

def check_zip(zip_path):
    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        return
    
    print(f"\nChecking: {zip_path}")
    print(f"Zip file size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB\n")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("Contents:")
            print("-" * 70)
            
            model_files = []
            for info in zip_ref.infolist():
                size_mb = info.file_size / (1024*1024)
                print(f"{info.filename:60} {size_mb:8.2f} MB")
                
                if "model.safetensors" in info.filename:
                    model_files.append((info.filename, info.file_size))
            
            print("-" * 70)
            
            if model_files:
                print("\nModel files found:")
                for filename, size in model_files:
                    size_mb = size / (1024*1024)
                    if size_mb < 10:
                        print(f"  {filename}: {size_mb:.2f} MB - WARNING: Too small (likely placeholder)")
                    else:
                        print(f"  {filename}: {size_mb:.2f} MB - Looks good!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_zip(sys.argv[1])
    else:
        # Try common locations
        locations = [
            r"C:\Users\Leo\Desktop\ibtikar-backend-main (1).zip",
            r"C:\Users\Leo\Downloads\ibtikar-backend-main (1).zip",
        ]
        for loc in locations:
            if os.path.exists(loc):
                check_zip(loc)
                break
        else:
            print("Usage: python check_zip_simple.py <path_to_zip_file>")

