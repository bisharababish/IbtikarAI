"""
Check zip file contents and file sizes
"""
import zipfile
import os
import sys

def check_zip(zip_path):
    """Check contents of a zip file"""
    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        return
    
    print(f"\n📦 Checking: {zip_path}")
    print(f"📊 Zip file size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB\n")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("📋 Contents:")
            print("-" * 60)
            
            total_size = 0
            for info in zip_ref.infolist():
                size_mb = info.file_size / (1024*1024)
                total_size += info.file_size
                
                # Highlight model.safetensors
                marker = " ⭐" if "model.safetensors" in info.filename else ""
                print(f"{info.filename:50} {size_mb:8.2f} MB{marker}")
            
            print("-" * 60)
            print(f"{'Total (uncompressed)':50} {total_size/(1024*1024):8.2f} MB")
            print(f"\n✅ Zip file looks valid!")
            
            # Check if model.safetensors is reasonable size
            model_files = [f for f in zip_ref.namelist() if "model.safetensors" in f]
            if model_files:
                for model_file in model_files:
                    info = zip_ref.getinfo(model_file)
                    size_mb = info.file_size / (1024*1024)
                    if size_mb < 10:
                        print(f"\n⚠️  WARNING: {model_file} is only {size_mb:.2f} MB")
                        print("   This is likely a placeholder, not the actual model!")
                    else:
                        print(f"\n✅ {model_file} is {size_mb:.2f} MB - looks good!")
            
    except zipfile.BadZipFile:
        print("❌ Error: Not a valid zip file")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_zip(sys.argv[1])
    else:
        print("Usage: python check_zip.py <path_to_zip_file>")
        print("\nExample:")
        print("  python check_zip.py C:\\Users\\Leo\\Downloads\\arabert_toxic_classifier.zip")

