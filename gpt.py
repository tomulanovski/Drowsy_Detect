import cv2
import dlib

# Initialize the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Define the threshold for yawn detection
yawn_threshold = 20

# Initialize the yawn counter and status
yawn_count = 0
yawn_status = False

# Loop through the video frames
while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through the faces
    for face in faces:
        # Detect landmarks for the face
        landmarks = predictor(gray, face)

        # Get the mouth landmarks
        mouth_landmarks = landmarks.parts()[48:68]

        # Calculate the mouth aspect ratio
        mouth_ar = (mouth_landmarks[3].y - mouth_landmarks[0].y) / (mouth_landmarks[6].x - mouth_landmarks[0].x)

        # Check if the mouth aspect ratio is less than the threshold
        if mouth_ar < yawn_threshold:
            # If the yawn status is False, set it to True and increment the yawn counter
            if not yawn_status:
                yawn_status = True
                yawn_count += 1

            # Draw a rectangle around the face and mouth
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.rectangle(frame, (mouth_landmarks[0].x, mouth_landmarks[0].y),
                          (mouth_landmarks[6].x, mouth_landmarks[6].y), (0, 0, 255), 2)
        else:
            # If the yawn status is True, set it to False
            if yawn_status:
                yawn_status = False

        # Display the yawn count on the frame
        cv2.putText(frame, "Yawn Count: {}".format(yawn_count), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Yawn Detection", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()