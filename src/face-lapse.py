import argparse
import cv2
import os
import math
import numpy as np
import imageio.v3 as iio
from PIL import Image

target_width=3072
target_height=2304
left_eye_pos=(1211, 818)
right_eye_pos=(1825, 818)
target_eye_distance=1825-1211

def save_movie(image_list, filename, fps):
    iio.imwrite(filename, image_list, fps=fps)

def rotate_and_scale(images, face_cascade, eye_cascade):
    lined_up_images = []
    for image in images:
        # Split the image into its 3 channels
        r,g,b = cv2.split(image)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Check if any faces were detected
        if len(faces) > 0:
            # Get the coordinates of the first face
            x, y, w, h = faces[0]
            
            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            
            # Check if any eyes were detected
            if len(eyes) > 1:
                # Get the coordinates of the first eye
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]

                # Get the center of each eye
                center_x1 = ex1 + (ew1 / 2)
                center_y1 = ey1 + (eh1 / 2)
                center_x2 = ex2 + (ew2 / 2)
                center_y2 = ey2 + (eh2 / 2)
                
                # Calculate the angle between the eyes
                angle = math.atan2(ey2 - ey1, ex2 - ex1)

                # Calculate the distance between the eyes
                distance = math.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)
                scale = target_eye_distance / distance
                
                # Rotate each channel by the calculated angle
                rows, cols = gray.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, scale)
                b = cv2.warpAffine(b, M, (cols, rows))
                g = cv2.warpAffine(g, M, (cols, rows))
                r = cv2.warpAffine(r, M, (cols, rows))
            else:
                # If only one eye is detected, don't rotate the image
                pass
        else:
            # If no face is detected, don't rotate the image
            pass
        
        # Merge the channels back together
        lined_up_image = cv2.merge((r, g, b))
        lined_up_images.append(lined_up_image)
    
    return lined_up_images

def line_up(images, face_cascade, eye_cascade):
    lined_up_images = []
    for image in images:
        # Split the image into its 3 channels
        b, g, r = cv2.split(image)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Check if any faces were detected
        if len(faces) > 0:
            # Get the coordinates of the first face
            x, y, w, h = faces[0]
            
            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            
            # Check if any eyes were detected
            if len(eyes) > 1:
                # Get the coordinates of the first eye
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]

                # Get the center of each eye
                center_x1 = ex1 + (ew1 / 2)
                center_y1 = ey1 + (eh1 / 2)
                center_x2 = ex2 + (ew2 / 2)
                center_y2 = ey2 + (eh2 / 2)
                
                # Calculate the angle between the eyes
                angle = math.atan2(ey2 - ey1, ex2 - ex1)

                # Get the current distance between the eyes
                current_eye_distance = math.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)

                # Calculate the scale factor to scale the image to the target eye distance
                scale = target_eye_distance / current_eye_distance
                
                # Scale and rotate each channel
                rows, cols = gray.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, scale)
                b = cv2.warpAffine(b, M, (cols, rows))
                g = cv2.warpAffine(g, M, (cols, rows))
                r = cv2.warpAffine(r, M, (cols, rows))
            else:
                # If only one eye is detected, don't rotate the image
                pass
        else:
            # If no face is detected, don't rotate the image
            pass
        
        # Merge the channels back together
        lined_up_image = cv2.merge((r, g, b))
        lined_up_images.append(lined_up_image)

    return lined_up_images

def crop_images(images):
    # Find the smallest image
    smallest_image = min(images, key=lambda x: x.shape[0]*x.shape[1])
    height, width = smallest_image.shape[:2]
    
    # Crop all images to the size of the smallest image
    cropped_images = []
    for image in images:
        h, w = image.shape[:2]
        if h>height or w>width:
            center = (0, 0)
            cropped = cropped_images.append(cv2.getRectSubPix(image, (width, height), center))
            cropped_images.append(cropped)

        else:
            cropped_images.append(image)
    return cropped_images

def print_dimensions(images):
    for image in images:
        height, width = image.shape[:2]

def save_images(folder_path, images, file_extension = 'jpg'):
    for i, image in enumerate(images):
        file_name = os.path.join(folder_path, str(i) + '.' + file_extension)
        cv2.imwrite(file_name, image)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Path to the folder containing the face images')
    parser.add_argument('-c', '--opencv_folder', type=str, required=True, help='Path to the folder where opencv is installed.')
    parser.add_argument('-v', '--video_output', type=str, required=True, help='Path to the output video file')
    parser.add_argument('-f', '--frames_per_second', type=float, default=0.2, help='Number of seconds per picture in the output video')
    parser.add_argument('-o', '--output_folder', type=str, required=False, help='Path to save processed files in')

    # Parse the arguments
    args = parser.parse_args()

    # Load the face and eye detection cascades
    cascades_root = args.opencv_folder + '/data/haarcascades/'
    face_cascade = cv2.CascadeClassifier(cascades_root + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cascades_root + 'haarcascade_eye.xml')
    images = []

    # Iterate through all the images in the folder
    for image_name in sorted(os.listdir(args.input_folder)):
        # Load the image
        image = cv2.imread(os.path.join(args.input_folder, image_name))
        images.append(image)

    images = rotate_and_scale(images, face_cascade, eye_cascade)
    images = line_up(images, face_cascade, eye_cascade)
    images = crop_images(images)
    images = [im for im in images if isinstance(im, (np.ndarray, Image.Image))]
    print_dimensions(images)

    if args.output_folder:
        save_images(args.output_folder, images)

    # Creating the video file
    save_movie(images, args.video_output, args.frames_per_second)
