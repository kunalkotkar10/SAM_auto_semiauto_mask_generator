# Code to replace a specific string in all filenames in a directory
import os

# Define the directory containing the files
directory = '../adson_forceps_with_teeth/adson_forceps_with_teeth_dark/color_images'

# Define the string to replace and the new string
old_string = 'adson_forceps_without_teeth'

new_string = 'adson_forceps_with_teeth'

# Replace the string in all filenames in the directory
for filename in os.listdir(directory):
    new_filename = filename.replace(old_string, new_string)
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    print(f'Renamed {filename} to {new_filename}')

# Print completion message
print('All files renamed successfully.')
