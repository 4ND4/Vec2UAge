# 1) Calculate the cosine distance between 2 similar images

    # face detect and crop      DONE

# 2) Calculate the cosine distance between 2 different images   DONE

# 3) Calculate the cosine distance between an original image and rotated image. Do this for each augmentation

#    do this for 5 different settings per augmentation


import os
import shutil

import sklearn
from scipy import spatial

from facenet import get_vectors, get_vectors_np
from preprocessing.crop_faces import crop_images
import numpy as np

image1 = 'augmentation/images/082A21.JPG'
image2 = 'augmentation/images/082A21.JPG'

image_array = [image1, image2]

repeated_path = os.path.expanduser('~/Documents/images/facenet/repeated_augmented_test/')
temp_vector_image_path = 'temp_vector_image'
temp_vector_output_directory = 'temp_vector_output'
temp_vector_crop = 'temp_vector_crop'

crop = True
vectors = True

if os.path.exists(temp_vector_image_path):
    shutil.rmtree(temp_vector_image_path)

if os.path.exists(temp_vector_output_directory):
    shutil.rmtree(temp_vector_output_directory)


if os.path.exists(temp_vector_crop):
    shutil.rmtree(temp_vector_crop)

os.mkdir(temp_vector_image_path)
os.mkdir(temp_vector_output_directory)
os.mkdir(temp_vector_crop)


if not os.path.exists(repeated_path):
    os.mkdir(repeated_path)

if crop:
    # detect with dlib and crop to 224

    # copy images to temp folder

    for i in range(0, len(image_array)):
        shutil.copy2(image_array[i], '{}/{}.jpg'.format(temp_vector_image_path, i))

    crop_images(image_directory=temp_vector_image_path, output_directory=temp_vector_crop, image_size=224)

# get face vector from image

if vectors:
    vectors = get_vectors_np(input_path=temp_vector_crop, image_size=160)

#print(vectors)

# read vector file and do predictions

# do the cosine similarity

#result = 1 - spatial.distance.cosine(vectors[0], vectors[1])

#result = sklearn.metrics.pairwise.cosine_similarity(vectors[0], vectors[1])

result = sklearn.metrics.pairwise.cosine_similarity(vectors[0], vectors[1])

dist = np.linalg.norm(vectors[0]-vectors[1])

print(result)
# print(dist)

print('done')

