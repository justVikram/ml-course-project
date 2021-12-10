import os


def traverse_dir(path: os.path, extension: str = 'jpg') -> None:
    # Traverse different class directories within test/train directory
    for directory in os.listdir(path):
        if directory.find('.') == -1:
            assign_seq_file_names(os.path.join(path, directory))
            change_img_extension(os.path.join(path, directory), extension)
            resize_img(os.path.join(path, directory), 512)


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


def resize_img(path: os.path, size: int) -> None:

    # Resize images in the dataset to {size}
    for img_name in os.listdir(path):
        if img_name.find('.') != 0:
            # To force the size on both dimensions, use 'size x size!'
            os.system(f"convert {os.path.join(path, img_name)} -resize '{size}x{size}' {os.path.join(path, img_name)}")

if __name__ == '__main__':
    train_path = './data/train'
    test_path = './data/test'
    extn = 'jpg'

    traverse_dir(train_path, extn)
    traverse_dir(test_path, extn)
