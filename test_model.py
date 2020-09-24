# creating facial vectors here

# model path
import os
from facenet import get_vectors
from preprocessing.crop_faces import crop_images

model_path = 'checkpoints/SAN-374/_ckpt_epoch_23.ckpt'
# input_path = 'cropped'
input_path = os.path.expanduser('~/Documents/images/facenet/visage_instagram_validation_test_single/')
output_path = os.path.expanduser('output/out_test_visage_instagram_augmented.json')
repeated_path = os.path.expanduser('~/Documents/images/facenet/repeated_augmented_test/')
crop = False
vectors = True

if not os.path.exists(repeated_path):
    os.mkdir(repeated_path)

if crop:
    # detect with dlib and crop to 224

    crop_images(image_directory='images', output_directory='cropped', image_size=224)

# get face vector from image

if vectors:
    get_vectors(input_path=input_path, output_path=output_path, image_size=160, repeated_path=repeated_path)

# read vector file and do predictions


