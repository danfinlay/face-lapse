import cv2
import os

# Path to the folder containing the face images
face_folder_path = 'path/to/face/images'

# Load the face detection and alignment model
face_detector = cv2.dnn.readNetFromCaffe('path/to/face_detection_model.prototxt', 'path/to/face_detection_model.caffemodel')

# Iterate through all the images in the folder
for image_name in os.listdir(face_folder_path):
    # Load the image
    image = cv2.imread(os.path.join(face_folder_path, image_name))
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
            eye_cascade = cv2.CascadeClassifier('path/to/eye_cascade.xml')
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

            # Save the aligned image
            cv2.imwrite(os.path.join('path/to/aligned/images', image_name), face_roi)


