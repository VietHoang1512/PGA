import os

# Define the path to the metadata file and the base directory for the images
metadata_file = './data/image-clef/p.txt'
base_dir = './data/image-clef'

# Define class names in the order corresponding to their IDs
class_names = [
    'aeroplane', 'bike', 'bird', 'boat', 'boele', 'bus', 
    'car', 'dog', 'horse', 'monitor', 'motorbike', 'people'
]
domain_id = os.path.basename(metadata_file).split(".")[0]
# Ensure that a directory exists for each class
for class_name in class_names:
    class_dir = os.path.join(base_dir, domain_id, class_name)
    os.makedirs(class_dir, exist_ok=True)

# Read the metadata file and move files to the corresponding class folder
with open(metadata_file, 'r') as file:
    for line in file:
        # Parse the image path and class ID from each line
        parts = line.strip().split()
        if len(parts) != 2:
            print("Error", parts)
        image_path, class_id = parts
        class_id = int(class_id)

        # Determine the new path for the image
        class_name = class_names[class_id]
        
        
        new_path = os.path.join(base_dir, domain_id, class_name, os.path.basename(image_path))

        # Move the file to the new directory
        old_path = os.path.join(base_dir, image_path)
        print("old_path", old_path)
        print("new_path", new_path)
        os.rename(old_path, new_path)

print("Images have been reorganized by class.")