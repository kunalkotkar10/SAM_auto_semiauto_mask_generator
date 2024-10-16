# import os
# import torch
# import numpy as np
# from PIL import Image
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def generate_automatic_masks(image_dir, output_dir):

#     # Define paths
#     sam_checkpoint = 'sam_vit_h_4b8939.pth'  # Model checkpoint (assuming it is available)
#     model_type = 'vit_h'

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Check device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f'Using device: {device}')

#     # Load the SAM model
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)

#     # Verify model loading
#     if sam is None:
#         print('Failed to load SAM model.')
#     else:
#         print('SAM model loaded successfully.')

#     # Adjust mask generator parameters
#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=128,  # Increased from 32 to 128 for finer detail
#         pred_iou_thresh=0.9,  # Lowered threshold to 0.4 for more mask coverage
#         stability_score_thresh=0.9,
#         # min_mask_region_area=300  # Set minimum area to avoid small irrelevant masks
#     )

#     # Process each image in the directory
#     for filename in os.listdir(image_dir):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
#             image_path = os.path.join(image_dir, filename)
            
#             # Load the image as RGB
#             image = np.array(Image.open(image_path).convert('RGB'))

#             # Generate masks
#             masks = mask_generator.generate(image)

#             # Debugging: Print number of masks generated
#             print(f'Number of masks generated for {filename}: {len(masks)}')

#             if not masks:
#                 print(f'No masks found for {filename}')
#                 continue

#             # Continue with the largest mask but if area > 50000 or less than 2000, remove from masks list
#             masks = [mask for mask in masks if 3000 < mask['area'] < 15000]

#             # Optionally, print areas of all masks
#             for idx, mask in enumerate(masks):
#                 print(f'Mask {idx}: Area = {mask["area"]}')

#             # Visualize masks
#             fig, ax = plt.subplots(1)
#             ax.imshow(image)
#             for mask in masks:
#                 segmentation = mask['segmentation']
#                 y_coords, x_coords = np.where(segmentation)
#                 if y_coords.size > 0 and x_coords.size > 0:
#                     rect = patches.Rectangle(
#                         (np.min(x_coords), np.min(y_coords)),
#                         np.max(x_coords) - np.min(x_coords),
#                         np.max(y_coords) - np.min(y_coords),
#                         linewidth=1,
#                         edgecolor='r',
#                         facecolor='none'
#                     )
#                     ax.add_patch(rect)
#             plt.title(f'Masks for {filename}')
#             # plt.show()


#             sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
#             instrument_mask = sorted_masks[0]['segmentation']

#             # Create a binary mask image
#             mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#             mask_image[instrument_mask] = 255  # Set mask pixels to white

#             # Save the mask as a PNG image
#             mask_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.png')
#             mask_pil = Image.fromarray(mask_image)
#             mask_pil.save(mask_output_path)

#             print(f'Saved mask for {filename} to {mask_output_path}')

#     print('Mask generation complete.')

# if __name__ == '__main__':
#     image_dir = 'path/to/dir'  # Directory containing the images
#     output_dir = 'path/to/dir'  # Directory to save generated masks
#     generate_automatic_masks(image_dir, output_dir)



########## CODE FOR ONLY TOP POSITION IMAGES ##########

import os
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def generate_automatic_masks(image_dir, output_dir):

    # Define paths
    sam_checkpoint = 'sam_vit_h_4b8939.pth'  # Model checkpoint (assuming it is available)
    model_type = 'vit_h'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Verify model loading
    if sam is None:
        print('Failed to load SAM model.')
    else:
        print('SAM model loaded successfully.')

    # Adjust mask generator parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128,  # Increased from 32 to 128 for finer detail
        pred_iou_thresh=0.9,  # Lowered threshold to 0.4 for more mask coverage
        stability_score_thresh=0.9,
        # min_mask_region_area=300  # Set minimum area to avoid small irrelevant masks
    )

    # Process each image in the directory
    for filename in os.listdir(image_dir):
        start_timer = time.time()
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

            try:
                # Split filename from first underscore
                idx = filename.split('_')[0]
                
                # Convert the base name to an integer
                idx = int(idx)
                
                # Check if the number is between 0 and 100 (inclusive)
                if 0 <= idx <= 100:
                    # Process the file as needed
                    print(f"Processing file: {filename}")

                    image_path = os.path.join(image_dir, filename)
            
                    # Load the image as RGB
                    image = np.array(Image.open(image_path).convert('RGB'))

                    # Generate masks
                    masks = mask_generator.generate(image)

                    # Debugging: Print number of masks generated
                    print(f'Number of masks generated for {filename}: {len(masks)}')

                    if not masks:
                        print(f'No masks found for {filename}')
                        continue

                    # Continue with the largest mask but if area > 50000 or less than 2000, remove drom masks list
                    masks = [mask for mask in masks if 3000 < mask['area'] < 25000]

                    # Optionally, print areas of all masks
                    for idx, mask in enumerate(masks):
                        print(f'Mask {idx}: Area = {mask["area"]}')

                    # Visualize masks
                    fig, ax = plt.subplots(1)
                    ax.imshow(image)
                    for mask in masks:
                        segmentation = mask['segmentation']
                        y_coords, x_coords = np.where(segmentation)
                        if y_coords.size > 0 and x_coords.size > 0:
                            rect = patches.Rectangle(
                                (np.min(x_coords), np.min(y_coords)),
                                np.max(x_coords) - np.min(x_coords),
                                np.max(y_coords) - np.min(y_coords),
                                linewidth=1,
                                edgecolor='r',
                                facecolor='none'
                            )
                            ax.add_patch(rect)
                    plt.title(f'Masks for {filename}')
                    # plt.show()


                    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                    instrument_mask = sorted_masks[0]['segmentation']

                    # Create a binary mask image
                    mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    mask_image[instrument_mask] = 255  # Set mask pixels to white

                    # Save the mask as a PNG image
                    mask_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.png')
                    mask_pil = Image.fromarray(mask_image)
                    mask_pil.save(mask_output_path)

                    print(f'Saved mask for {filename} to {mask_output_path}')

                    end_timer = time.time()
                    print(f"Time taken for processing file: {filename} is {end_timer - start_timer} seconds")

                else:
                    # Skip files that are not in the range 0-100
                    print(f"Skipping file: {filename}")

            except Exception as e:
                # Skip files that don't have a numeric name
                print(f'Error processing file: {filename} - {e}')
                continue
            

    print('Mask generation complete.')

if __name__ == '__main__':
    image_dir = 'path/to/dir'  # Directory containing the images
    output_dir = 'path/to/dir'  # Directory to save generated masks
    generate_automatic_masks(image_dir, output_dir)