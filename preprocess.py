import PIL
from PIL import Image
import os

def find_images(root_path, extension = 'JPG'):
    from os import walk

    data_files = []
    for (dirpath, dirnames, filenames) in walk(path):
        print(dirpath, dirnames, filenames)
        # First add all files if the extension matches.
        for filename in filenames:
            extension = os.path.splitext(filename)[1][1:]
            if extension == 'JPG':
                data_files.append(os.path.join(dirpath, subdir, filename))
        # Then recursively add all subdirectories.
        for subdir in dirnames:
            data_files.append(find_images(os.path.join(dirpath, subdir), extension))


def resize_all_images(path, new_path, width, height):
    # Make sure the new path exists, creating it if necessary.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    images = find_images(path)
    for img in images:
        img = Image.open(path)
        img = img.resize((width, height), PIL.Image.ANTIALIAS)
         # Create a new path name, replacing the old path with the new path.
        updated_path = os.path.join(new_path, os.path.split(images)[1:])
        img.save(new_path)




