# import os
# import torch
# import numpy as np
# import cv2
# from segment_anything import sam_model_registry, SamPredictor

# # Define paths
# image_dir = '../small_kelly_clamp/small_kelly_clamp_normal/color_images'       # Replace with your images directory path
# output_dir = '../small_kelly_clamp/small_kelly_clamp_normal/masks'  # Replace with your desired output directory
# sam_checkpoint = 'sam_vit_h_4b8939.pth'  # Replace with the path to your SAM checkpoint
# model_type = 'vit_h'  # Model type: 'vit_h', 'vit_l', or 'vit_b'

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Check device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')

# # Load the SAM model
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# # Initialize the predictor
# predictor = SamPredictor(sam)

# # Variables to store rectangle coordinates
# ix, iy = -1, -1
# drawing = False
# rect_done = False

# # Function to draw rectangle and get coordinates
# def draw_rectangle(event, x, y, flags, param):
#     global ix, iy, drawing, rect_done, img_display

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#         rect_done = False
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             img_display = img.copy()
#             cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         rect_done = True
#         cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
#         # Store the rectangle coordinates
#         param['box'] = [ix, iy, x, y]

# # Process each image in the directory
# for filename in os.listdir(image_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
#         image_path = os.path.join(image_dir, filename)
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         img_display = img.copy()

#         # Reset rectangle parameters
#         rect_params = {}
#         drawing = False
#         rect_done = False

#         # Set up window and mouse callback
#         cv2.namedWindow('Image')
#         cv2.setMouseCallback('Image', draw_rectangle, rect_params)

#         print(f'\nDraw a rectangle around the instrument in the image: {filename}')
#         print('Press "r" to reset the selection, or "Esc" to skip this image.')

#         while True:
#             cv2.imshow('Image', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
#             k = cv2.waitKey(1) & 0xFF
#             if rect_done:
#                 break
#             elif k == ord('r'):
#                 img_display = img.copy()
#                 rect_params = {}
#                 drawing = False
#                 rect_done = False
#                 print('Selection reset. Draw the rectangle again.')
#             elif k == 27:  # Esc key to skip
#                 print('Skipping this image.')
#                 break
#         cv2.destroyAllWindows()

#         # Get the rectangle coordinates
#         if 'box' in rect_params:
#             x0, y0, x1, y1 = rect_params['box']
#             # Ensure coordinates are within image bounds
#             x0 = max(0, min(x0, img.shape[1]-1))
#             x1 = max(0, min(x1, img.shape[1]-1))
#             y0 = max(0, min(y0, img.shape[0]-1))
#             y1 = max(0, min(y1, img.shape[0]-1))
#             input_box = np.array([x0, y0, x1, y1])

#             # Prepare image for predictor
#             predictor.set_image(img)

#             # Use predictor to get masks
#             masks, _, _ = predictor.predict(
#                 point_coords=None,
#                 point_labels=None,
#                 box=input_box[None, :],
#                 multimask_output=False,
#             )

#             # Get the mask (since multimask_output=False, masks.shape[0] == 1)
#             mask = masks[0]

#             # Save the mask as a PNG image
#             mask_image = (mask * 255).astype(np.uint8)
#             mask_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.png')
#             cv2.imwrite(mask_output_path, mask_image)

#             print(f'Saved mask for {filename} to {mask_output_path}')
#         else:
#             print(f'No rectangle drawn for {filename}')

import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# Define paths
path = '../small_kelly_clamp/small_kelly_clamp_dark' 
input_path = f'{path}/color_images/105_small_kelly_clamp_dark_color_1280x720.png'  # Replace with your images directory path or single image path 
output_dir = f'{path}/masks'         # Replace with your desired output directory
sam_checkpoint = 'sam_vit_h_4b8939.pth'                                    # Replace with the path to your SAM checkpoint
model_type = 'vit_h'                                              

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize the predictor
predictor = SamPredictor(sam)

# Variables to store rectangle coordinates
ix, iy = -1, -1
drawing = False
rect_done = False

# Function to draw rectangle and get coordinates
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect_done, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect_done = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_done = True
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
        # Store the rectangle coordinates
        param['box'] = [ix, iy, x, y]

# Determine if the input path is a directory or a file
if os.path.isdir(input_path):
    # Process all images in the directory
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
elif os.path.isfile(input_path):
    # Process the single image file
    image_files = [input_path]
else:
    print(f'Error: {input_path} is neither a file nor a directory.')
    exit(1)

# Process each image
for image_path in image_files:
    filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f'Failed to load image: {image_path}')
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_display = img.copy()

    # Reset rectangle parameters
    rect_params = {}
    drawing = False
    rect_done = False

    # Set up window and mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle, rect_params)

    print(f'\nDraw a rectangle around the instrument in the image: {filename}')
    print('Press "r" to reset the selection, or "Esc" to skip this image.')

    while True:
        cv2.imshow('Image', cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        if rect_done:
            break
        elif k == ord('r'):
            img_display = img.copy()
            rect_params = {}
            drawing = False
            rect_done = False
            print('Selection reset. Draw the rectangle again.')
        elif k == 27:  # Esc key to skip
            print('Skipping this image.')
            break
    cv2.destroyAllWindows()

    # Get the rectangle coordinates
    if 'box' in rect_params:
        x0, y0, x1, y1 = rect_params['box']
        # Ensure coordinates are within image bounds
        x0 = max(0, min(x0, img.shape[1]-1))
        x1 = max(0, min(x1, img.shape[1]-1))
        y0 = max(0, min(y0, img.shape[0]-1))
        y1 = max(0, min(y1, img.shape[0]-1))
        input_box = np.array([x0, y0, x1, y1])

        # Prepare image for predictor
        predictor.set_image(img)

        # Use predictor to get masks
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Get the mask (since multimask_output=False, masks.shape[0] == 1)
        mask = masks[0]

        # Save the mask as a PNG image
        mask_image = (mask * 255).astype(np.uint8)
        mask_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.png')
        cv2.imwrite(mask_output_path, mask_image)

        print(f'Saved mask for {filename} to {mask_output_path}')
    else:
        print(f'No rectangle drawn for {filename}')
