import PIL
from PIL import Image
import os
from pathlib2 import Path

verbose = False

def find_images(root_path, extension = 'jpg'):
    from os import walk

    global verbose
    if verbose:
        print("adding images in " + root_path + "..")

    data_files = []
    for (dirpath, dirnames, filenames) in walk(root_path):
        # First add all files if the extension matches.
        for filename in filenames:
            #print filename
            ext = os.path.splitext(filename)[1][1:]
            #print extension
            if ext == extension:
                found_file = os.path.join(dirpath, filename)
                #print("found " + found)
                data_files.append(found_file)

        # Then recursively add all subdirectories.
        for subdir in dirnames:
            found_files = find_images(os.path.join(dirpath, subdir), extension)
            if len(found_files) > 0:
                data_files = data_files + found_files
    return data_files


def resize_all_images(path, new_path, width, height):
    # Make sure the new path exists, creating it if necessary.
    global verbose
    if verbose:
        print("path: " + path)
        print("new_path: " + new_path)
        print("width: " + str(width))
        print("height: " + str(height))
    try:
        os.makedirs(new_path)
    except OSError:
        if not os.path.isdir(new_path):
            raise

    images = find_images(path)
    n_images = len(images)
    i_images = n_images / 100.0
    i = 0
    p = 0
    if verbose:
        print "found %i images.. start processing." % (n_images)

    if len(images) <= 0:
        if verbose:
            print("no images found.")
        return False
    for image_path in images:
        i = i + 1
        complete_per = i * 100.0 / n_images
        if int(complete_per) > p :
            if verbose:
                print "%i %%" % (p)
            p = int(complete_per)
        #print len(image_path)
        img = Image.open(image_path)
        img = img.resize((width, height), PIL.Image.ANTIALIAS)
        # Create a new path name, replacing the old path with the new path.
        updated_path = image_path.replace(path, new_path, 1)
        #print updated_path
        img.save(updated_path)

# Allow pre-processing to be called as a script.
def main():
    import argparse

    default_width = 64
    default_height = 64
    global verbose

    parser = argparse.ArgumentParser(
            description='preprocess a directory of images by resizing and moving to a new directory with the same structure')
    parser.add_argument('path', metavar='path', type=str,
                       help='path to search for images')
    parser.add_argument('new_path', metavar='new_path', type=str,
                       help='new path to place images (retains subdir structure)')
    parser.add_argument('--width', metavar='width', type=int, default=default_width,
                       help='new image width in pixels. [default:64]')
    parser.add_argument('--height', metavar='height',  type=int, default=default_height,
                       help='new image height in pixels. [default: 64]')
    parser.add_argument('--verbose', action='store_true', help='logs the progress of the preprocessing.')

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    resize_all_images(path=args.path, new_path=args.new_path, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
