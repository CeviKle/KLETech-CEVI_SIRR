import os
import cv2  # OpenCV for image loading and saving

output_dir = './output'
input_dir = './results/real20'

folder_list = os.listdir(input_dir)

for single in folder_list:
    single_output_path = os.path.join(input_dir, single, 'ytmt_ucs_sirs_t.png')
    
    # Check if the file exists before loading
    if os.path.exists(single_output_path):
        # Load the image
        image = cv2.imread(single_output_path)

        # Define the new output path with the new name
        new_filename = f"{single}.jpg"
        new_output_path = os.path.join(output_dir, new_filename)
        output = cv2.imread(new_output_path)

        # if output.shape != image.shape:
        #     print(f'error for single :{single}')

        # Save the image with the new name in output_dir
        cv2.imwrite(new_output_path, image)
        print(f"Saved: {new_output_path}")
    else:
        print(f"File not found: {single_output_path}")
