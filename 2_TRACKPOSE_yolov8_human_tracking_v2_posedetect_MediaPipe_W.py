from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load YOLOv8 model for human detection
model_detection = YOLO("yolov8n.pt")  # Replace with your YOLOv8 model

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open a video file or webcam
video_path = "Alan Walker - The Spectre (Remix) Shuffle Dance Music Video ♫ LED Shoes Dance Special [x80nDQ7H1vo].mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)  # Use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection (YOLOv8 model)
    results_detection = model_detection(frame)

    # Process results from object detection (YOLO)
    for result in results_detection:
        for box in result.boxes:
            # Extract bounding box and class information
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls)
            conf = box.conf[0]

            # Filter only for 'person' class (class 0 in COCO dataset)
            if cls == 0:  
                # Draw bounding box around person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Crop person area for pose estimation (if necessary)
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

                # Run pose estimation with MediaPipe
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results_pose = pose.process(rgb_frame)

                # Draw pose landmarks
                if results_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(cropped_frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Put the cropped pose frame back into the main frame
                frame[int(y1):int(y2), int(x1):int(x2)] = cropped_frame

    # Display the frame with bounding boxes and pose keypoints
    cv2.imshow("YOLOv8 Human Detection & Pose Estimation", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
