# augment training set only

# 1) Stratified Shuffle Split, with a lower percentage of training samples.

# count images

# get visage_base, Done
# check instagram script for age range images, Done
# add instagram to visage, Done
# count images again, Done
# add images to 9 y/o, Done
# crop images with DLIB, Done
# count images again, Done
# generate train and test dataset, Done
# augment training dataset, Done
# move files to single folder and rename, Done
# facenet vectors, Done: test_model.py


import os
import random
import shutil

import Augmentor as Augmentor

visage_instagram_base_path = os.path.expanduser('~/Documents/images/facenet/visage_instagram_train')
output_path = os.path.expanduser('~/Documents/images/facenet/visage_instagram_validation_test_single')
train_output_path = os.path.expanduser('~/Documents/images/facenet/visage_instagram_train')
count_images = False
generate_random_set = True
data_augmentation = False

if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(train_output_path):
    os.mkdir(train_output_path)

if count_images:

    N = 0  # total files

    min_count = 1000

    for dir_path, dir_names, file_names in os.walk(visage_instagram_base_path):

        N_c = len(file_names)
        N += N_c

        if N_c < min_count and N_c != 1:
            min_count = N_c

        print("Files in ", dir_path, N_c)
    print("Total Files ", N)

    print('minimum file in a folder', min_count)

# randomly obtain 500 images for Validation and Test 400 validation 100 test

# test and validation set of 9000 images


def get_random_dataset(my_path, number):

    list_directory = [x for x in os.listdir(my_path) if not x.startswith('.')]

    # change here if you already have the folder with a specific number or if you want it to be limited to.

    arr = [x for x in range(0, len(list_directory))]

    random.Random(42).shuffle(arr)

    random_list = []
    random_train_list = []

    for i in arr[:number]:
        file_path = os.path.join(my_path, list_directory[arr[i]])

        random_list.append(file_path)

    for i in arr[number:]:
        file_path = os.path.join(my_path, list_directory[arr[i]])
        random_train_list.append(file_path)

    return random_list, random_train_list


def augment_dataset(input_path, total_images):
    """
Augment a folder that has already been backed up
    :param input_path: path of the images
    :param total_images: number of images to augment to
    """
    # get length of input directory

    count_directory = len([x for x in os.listdir(input_path) if not x.startswith('.')])

    for d in range(1, count_directory + 1):

        image_path = os.path.expanduser(os.path.join(input_path, str(d)))
        output_directory = image_path

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        p = Augmentor.Pipeline(
            source_directory=image_path,
            output_directory=output_directory
        )

        p.flip_left_right(probability=0.5)
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.zoom_random(probability=0.5, percentage_area=0.95)
        p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
        p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
        p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
        p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
        p.random_erasing(probability=0.5, rectangle_area=0.2)

        count_images = len(os.listdir(image_path))

        generate_count = total_images + 1 - count_images

        print('\r\ngenerating {} images'.format(str(generate_count)))

        p.sample(generate_count)
        print('\r\n{} processed'.format(d))


if generate_random_set:

    for d in range(1, 19):

        file_path_list, train_file_path_list = get_random_dataset(os.path.join(visage_instagram_base_path, str(d)), 500)

        # copy files to test path

        for f in file_path_list:
            file_name = os.path.basename(f)

            # output to single file

            if not os.path.exists(output_path):
                os.mkdir(output_path)

            shutil.copy2(f, os.path.join(output_path, file_name))

        for f in train_file_path_list:
            file_name = os.path.basename(f)

            output_directory = os.path.join(train_output_path, str(d))

            if not os.path.exists(output_directory):
                os.mkdir(output_directory)

            shutil.copy2(f, os.path.join(output_directory, file_name))

    print('dataset created')

if data_augmentation:

    augment_path = os.path.expanduser('~/Documents/images/dataset/augmented_visage_instagram_train/')
    output_path = os.path.expanduser('~/Documents/images/facenet/visage_instagram_train/')

    print('making a backing in {} ...'.format(augment_path))
    shutil.copytree(output_path, augment_path)

    print('augmenting dataset ....')
    augment_dataset(augment_path, 5000)

    print('done')