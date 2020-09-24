import os
from imutils import face_utils
from mtcnn import MTCNN
import cv2
import dlib

class_detector = 'DLIB'


def dlib_detector(img):
    if img is not None:
        detector = dlib.get_frontal_face_detector()
        rects = detector(img, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            return face_utils.rect_to_bb(rect)


def dlib_cnn_detector(img):
    WEIGHTS = 'preprocessing/mmod_human_face_detector.dat'

    cnn_face_detector = dlib.cnn_face_detection_model_v1(WEIGHTS)

    dets = cnn_face_detector(img, 1)

    # loop over the face detections

    for i, d in enumerate(dets):
        x = d.rect.left()
        y = d.rect.top()
        w = d.rect.right() - x
        h = d.rect.bottom() - y

        return x, y, w, h


def mtcnn_detector(img):
    detector = MTCNN()

    result_list = detector.detect_faces(img)

    if result_list is not None and len(result_list) != 0:
        # get coordinates

        return result_list[0].get('box')


def crop_images(image_directory, output_directory, image_size):

    image_list = os.listdir(image_directory)

    dimensions = None

    for f in image_list:

        if f.startswith('.'):
            continue

        input_path = os.path.join(image_directory, f)

        image = cv2.imread(input_path)

        if class_detector == 'MTCNN':
            dimensions = mtcnn_detector(img=image)
        elif class_detector == 'DLIB':
            dimensions = dlib_detector(img=image)
        elif class_detector == 'DLIB_CNN':
            dimensions = dlib_cnn_detector(img=image)
        elif class_detector == 'STASM':
            pass
        else:
            print('detector unavailable')
            exit(0)

        # Needed if you use OpenCV, By default, it use BGR instead RGB

        if dimensions is None:
            print('face not detected attempting CNN', f)

            dimensions = dlib_cnn_detector(img=image)

        if dimensions is not None:
            x1, y1, width, height = dimensions

            x2, y2 = x1 + width, y1 + height

            x1 = 0 if x1 < 0 else x1
            x2 = 0 if x2 < 0 else x2
            y1 = 0 if y1 < 0 else y1
            y2 = 0 if y2 < 0 else y2

            cropped_image = image[y1:y2, x1:x2]

            resize_crop = cv2.resize(cropped_image, (image_size, image_size), interpolation=cv2.INTER_AREA)

            cv2.imwrite(os.path.join(output_directory, f), resize_crop)

            print(f, 'cropped')
        else:
            print('face was still not detected', f)
            # copy file to folder
