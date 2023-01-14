import argparse
import cv2
import os
import math
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image

def align_pupils(image, face_cascade, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            left_eye = eyes[0] if eyes[0][0] < eyes[1][0] else eyes[1]
            right_eye = eyes[1] if eyes[0][0] < eyes[1][0] else eyes[0]
            left_eye_center = (x + left_eye[0] + int(left_eye[2]/2), y + left_eye[1] + int(left_eye[3]/2))
            right_eye_center = (x + right_eye[0] + int(right_eye[2]/2), y + right_eye[1] + int(right_eye[3]/2))
            angle = np.arctan((right_eye_center[1] - left_eye_center[1]) / (right_eye_center[0] - left_eye_center[0])) * 180 / np.pi
            rot_mat = cv2.getRotationMatrix2D((left_eye_center[0], left_eye_center[1]), angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
    return image

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Path to the folder containing the face images')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the folder where the aligned images will be saved')
    parser.add_argument('-c', '--opencv_folder', type=str, required=True, help='Path to the folder where opencv is installed.')
    parser.add_argument('-v', '--video_output', type=str, required=True, help='Path to the output video file')
    parser.add_argument('-fps', '--frames_per_second', type=float, default=0.2, help='Number of seconds per picture in the output video')

    # Parse the arguments
    args = parser.parse_args()

    # Load the face and eye detection cascades
    cascades_root = args.opencv_folder + '/data/haarcascades/'
    face_cascade = cv2.CascadeClassifier(cascades_root + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cascades_root + 'haarcascade_eye.xml')
    images = []
    max_width = 0
    max_height = 0

    # Iterate through all the images in the folder
    for image_name in os.listdir(args.input_folder):
        # Load the image
        image = cv2.imread(os.path.join(args.input_folder, image_name))
        align_pupils(image, face_cascade, eye_cascade)
        max_height = max(max_height,image.shape[0])
        max_width = max(max_width,image.shape[1])
        #resizing image to max size
        image = cv2.resize(image, (max_width, max_height))
        #Convert numpy array to PIL image
        image = Image.fromarray(image)
        images.append(image)
    # Creating the video file
    image_clip = ImageSequenceClip(images, fps=1/args.frames_per_second)
    image_clip.write_videofile(args.video_output)