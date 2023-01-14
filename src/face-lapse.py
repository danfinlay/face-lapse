import argparse
import cv2
import os
import math
import numpy as np
from moviepy.editor import ImageSequenceClip

def align_pupils(image, face_detector, eye_cascade):
    (h, w) = image.shape[:2]
    # Pass the image through the face detector
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Iterate through the detections and align the pupils
    for i in range(0, detections.shape[2]):
        # Get the confidence of the detection
        confidence = detections[0, 0, i, 2]

        # If the detection is above a certain confidence threshold
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Get the face ROI
            face_roi = image[startY:endY, startX:endX]

            # Align the pupils
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(gray)

            if len(eyes) == 2:
                # Get the coordinates of the eyes
                left_eye = eyes[0]
                right_eye = eyes[1]
                if left_eye[0] > right_eye[0]:
                    left_eye, right_eye = right_eye, left_eye

                # Get the center of the eyes
                left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
                right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

                # Get the angle between the eyes
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = math.atan2(dy, dx) * 180.0 / math.pi

                # Get the rotation matrix
                rot_mat = cv2.getRotationMatrix2D(left_eye_center, angle, 1.0)

                # Rotate the image
                result = cv2.warpAffine(face_roi, rot_mat, (face_roi.shape[1], face_roi.shape[0]), flags=cv2.INTER_LINEAR)
                face_roi = result

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
        cv2.imwrite(os.path.join(args.output_folder, image_name), image)
        images.append(image)
    # Creating the video file
    image_clip = ImageSequenceClip(images, fps=1/args.frames_per_second)
    image_clip.write_videofile(args.video_output)