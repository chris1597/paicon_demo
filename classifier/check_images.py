import os
from PIL import Image

def check_corrupted_images(folder_path):
    corrupted_images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    with Image.open(file_path) as img:
                        img.load()
                        img = img.convert("RGB")
                except (IOError, SyntaxError, OSError, AttributeError) as e:
                    print(f"Corrupted image: {file_path} - {e}")
                    corrupted_images.append(file_path)
    return corrupted_images

folder_path = 'data/group_3'
corrupted_images = check_corrupted_images(folder_path)

print("Corrupted images found:")
for img_path in corrupted_images:
    print(img_path)
