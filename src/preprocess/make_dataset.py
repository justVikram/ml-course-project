import os
import shutil
import numpy


def traverse_dir(path: os.path, extension: str = 'jpg') -> None:
    # Traverse different class directories within test/train directory
    for directory in os.listdir(path):
        if directory.find('.') == -1:
            assign_seq_file_names(os.path.join(path, directory))
            change_img_extension(os.path.join(path, directory), extension)
            resize_img(os.path.join(path, directory))


def change_img_extension(path: os.path, extension: str) -> None:
    # Change the extension of images in the dataset to {extension}
    for img_name in os.listdir(path):
        if img_name.find('.') != 0:
            new_img_name = img_name[0:img_name.find('.')] + f'.{extension}'
            os.rename(os.path.join(path, img_name), os.path.join(path, new_img_name))


def assign_seq_file_names(path: os.path) -> None:
    # Rename images in the dataset so that they have sequential file names
    print(sorted(os.listdir(path)))
    for num, file_name in enumerate(sorted(os.listdir(path))):
        if file_name.find('.') != 0:
            new_file_name = f'{num}.jpg'
            os.rename(os.path.join(path, file_name), os.path.join(path, new_file_name))


def split_into_test_train(data_path: os.path, test_path: os.path, test_size: float) -> None:
    subcategories = ['mall', 'college', 'office']
    for subcategory in subcategories:
        source = os.path.join(data_path, subcategory)
        dest = os.path.join(test_path, subcategory)

        for file in os.listdir(source):
            if file.find('.') != 0:
                if numpy.random.rand(1) < test_size:
                    shutil.move(os.path.join(source, file), os.path.join(dest, file))


def resize_img(path: os.path, size: int = 32) -> None:
    # Resize images in the dataset to {size}
    for img_name in os.listdir(path):
        if img_name.find('.') != 0:
            # To force the size on both dimensions, use 'size x size!'
            os.system(f"convert {os.path.join(path, img_name)} -resize '{size}x{size}!' {os.path.join(path, img_name)}")


if __name__ == '__main__':
    train_path = './data/train'
    test_path = './data/test'
    extn = 'jpg'

    split_into_test_train(train_path, test_path, 0.2)
    traverse_dir(train_path, extn)
    traverse_dir(test_path, extn)
