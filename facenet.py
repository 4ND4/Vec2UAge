import json
import os
import re
import shutil

import imageio as imageio
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.platform import gfile


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        print(image_paths[i])

        if os.path.exists(image_paths[i]):

            img = imageio.imread(image_paths[i])
            if img.ndim == 2:
                img = to_rgb(img)
            if do_prewhiten:
                img = prewhiten(img)
            img = crop(img, do_random_crop, image_size)
            img = flip(img, do_random_flip)
            images[i, :, :, :] = img
        else:
            print('image might have been repeated and moved')
    return images


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    tf.disable_v2_behavior()
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_image_paths(inpath):
    paths = []

    for (root, dirs, files) in os.walk(inpath):
        for f in (f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
            paths.append(os.path.join(root, f))

    return (paths)


def faces_to_vectors(inpath, modelpath, outpath, imgsize, repeated_path, batchsize=100):
    '''
    Given a folder and a model, loads images and performs forward pass to get a vector for each face
    results go to a JSON, with filenames mapped to their facevectors
    :param imgsize: size of image
    :param repeated_path: path to move duplicate face vector source images
    :param batchsize: size of batch
    :param inpath: Where are your images? Must be cropped to faces (use MTCNN!)
    :param modelpath: Where is the tensorflow model we'll use to create the embedding?
    :param outpath: Full path to output file (better give it a JSON extension)
    :return: Number of faces converted to vectors
    '''

    results = dict()

    tf.disable_v2_behavior()
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:

            load_model(modelpath)
            mdl = None

            image_paths = get_image_paths(inpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Let's do them in batches, don't want to run out of memory

            full_array = np.empty((0, 512), float)
            full_paths = []

            for i in range(0, len(image_paths), batchsize):
                images = load_data(image_paths=image_paths[i:i + batchsize], do_random_crop=False, do_random_flip=False,
                                   image_size=imgsize, do_prewhiten=True)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}

                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                full_array = np.vstack((full_array, emb_array))

                for j in range(0, len(emb_array)):
                    relpath = os.path.relpath(image_paths[i + j], inpath)

                    full_paths.append(relpath)

                    results[relpath] = emb_array[j].tolist()

            unq, count = np.unique(full_array, axis=0, return_counts=True)

            repeated_groups = unq[count > 1]

            for repeated_group in repeated_groups:
                repeated_idx = np.argwhere(np.all(full_array == repeated_group, axis=1))
                repeated_ = repeated_idx.ravel()

                repeated_files = [full_paths[x] for x in repeated_]

                # move files to duplicate folder outside of path (don't move first instance)

                print('removing repeated from:', repeated_files[0])

                [shutil.move(os.path.join(inpath, r), repeated_path) for r in repeated_files[1:]]
                # remove from dictionary

                for r in repeated_files[1:]:
                    del results[r]

    # All done, save for later!
    json.dump(results, open(outpath, "w"))

    return len(results.keys())


def get_vectors(input_path, output_path, image_size, repeated_path):
    mdlpath = 'models/facenet/20180402-114759.pb'

    num_images_processed = faces_to_vectors(
        inpath=input_path, modelpath=mdlpath, outpath=output_path, imgsize=image_size, repeated_path=repeated_path
    )

    if num_images_processed > 0:
        print("Converted " + str(num_images_processed) + " images to face vectors.")
    else:
        print("No images were processed")
