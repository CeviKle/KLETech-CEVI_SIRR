from PIL import Image
import os

def check_images_with_size(folder, target_size=(638, 478)):
    matching_images = []
    
    for img_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, img_name)
        
        try:
            with Image.open(path) as image:
                size = image.size  # (width, height)
                print(f'size is {size}')
                
                if size == target_size:
                    matching_images.append(img_name)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
    
    if matching_images:
        print("Images with size 638x478:")
        for img in matching_images:
            print(img)
    else:
        print("No images found with the size 638x478.")

# Example usage:
folder = "/home/jatin/train_inp"
check_images_with_size(folder)
