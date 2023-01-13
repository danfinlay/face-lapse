import cv2
import os

def image_folder_to_video(folder_path, output_path):
    # Get the list of image filenames
    filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
    filenames.sort()  # Sort the filenames

    # Get the dimensions of the first image
    image = cv2.imread(filenames[0])
    height, width, _ = image.shape

    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Add each image to the video with 0.2 seconds of duration
    for filename in filenames:
        image = cv2.imread(filename)
        for i in range(15):
            out.write(image)

    # Release the video writer object
    out.release()
