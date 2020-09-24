# iterate visage_base folder and create a new folder with all the files inside and rename the visage dataset
import os
import random
import shutil
import uuid

visage_input_folder = os.path.expanduser('~/Documents/images/dataset/augmented_visage_instagram_train/')
output_folder = os.path.expanduser('~/Documents/images/facenet/augmented_visage_instagram_train/')


if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# rename files so they match instagram

process_visage_instagram = False
random_dataset = True


def get_random_dataset(my_path, number):

    list_directory = [x for x in os.listdir(my_path) if not x.startswith('.')]

    # change here if you already have the folder with a specific number or if you want it to be limited to.

    arr = [x for x in range(0, len(list_directory))]

    random.shuffle(arr)

    random_list = []

    for i in arr[:number]:
        file_path = os.path.join(my_path, list_directory[arr[i]])

        random_list.append(file_path)

    return random_list


if process_visage_instagram:

    for d in range(1, 19):
        directory_visage = os.listdir(os.path.join(visage_input_folder, str(d)))

        for f in directory_visage:
            if f.startswith('.'):
                continue

            file_name, file_extension = os.path.splitext(f)

            unique_filename = str(uuid.uuid4().hex)

            new_name = '{}_{}_x{}'.format(unique_filename, d, file_extension)

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            new_path = os.path.join(output_folder, new_name)

            shutil.copy2(os.path.join(visage_input_folder, str(d), f), new_path)

            print(f, 'processed')


if random_dataset:

    file_directory = os.path.expanduser('~/Documents/images/facenet/augmented_visage_instagram_train/')

    file_list = get_random_dataset(file_directory, 100)

    for f in file_list:
        shutil.copy2(f, 'augmented_sample/{}'.format(os.path.basename(f)))
