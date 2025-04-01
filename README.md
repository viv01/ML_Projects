### OBJECT DETECTION USING AI/ML
### 0b_shoe_classification_based_on_feeback.py

## MODEL USED
'''Load the Faster R-CNN model pre-trained on COCO dataset and adjust it for custom fine-tuning.  
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  

## WHAT I AM TRYING TO DO
i use the resnet50 pretained model which can detect objects . and then i am trying to provide it feedback to improve its detection. when a picture shows up, the model auto detects some objects and every object is marked in a box. i provide the feedback by giving back a comma seperated list of box labels, which accurately detected the object

![image](https://github.com/user-attachments/assets/0577e7b9-ba47-4f96-8b16-d2cd60a2975f)




### COMPUTER VISION / MOTION DETECTION USING AI/ML
### 2_TRACKPOSE_yolov8_human_tracking_v2_posedetect_MediaPipe_W.py

## MODEL USED
'''Load YOLOv8 model  
model = YOLO("yolov8n.pt")  

## WHAT I AM TRYING TO DO
I was experimenting with computer vision models to detect motion in a video. i tried to detect motion of a person dancing in a youtube video

![image](https://github.com/user-attachments/assets/151424f8-1b02-4de5-a8e4-512a99dc20a1)


### GEN AI/ML MUSIC GENERATION by TRAINING BASED ON EXISTING MUSIC
### 9a_intial_train_LSTM_model.py
## MODEL USED
'''Load YOLOv8 model  
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True, input_shape=input_shape),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(input_shape[1], activation='linear')
])  

## WHAT I AM TRYING TO DO
** CURRENTLY NOT WORKING. I was experimenting with AI LSTM model where i could train the AI/ML model on 5 music tracks of my choice. And then generate a new track based on the training.  

![image](https://github.com/user-attachments/assets/375a6f43-1b16-4b6c-9b2e-9ac71c1d5d72)
