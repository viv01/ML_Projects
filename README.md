### ## 0b_shoe_classification_based_on_feeback.py (OBJECT DETECTION USING AI/ML)

## MODEL USED
'''Load the Faster R-CNN model pre-trained on COCO dataset and adjust it for custom fine-tuning.  
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  

## WHAT I AM TRYING TO DO
i use the resnet50 pretained model which can detect objects . and then i am trying to provide it feedback to improve its detection. when a picture shows up, the model auto detects some objects and every object is marked in a box. i provide the feedback by giving back a comma seperated list of box labels, which accurately detected the object

![image](https://github.com/user-attachments/assets/0577e7b9-ba47-4f96-8b16-d2cd60a2975f)




### ## 2_TRACKPOSE_yolov8_human_tracking_v2_posedetect_MediaPipe_W.py (MOTION DETECTION USING AI/ML)

## MODEL USED
'''Load YOLOv8 model  
model = YOLO("yolov8n.pt")  

## WHAT I AM TRYING TO DO
I was experimenting with computer vision models to detect motion in a video. i tried to detect motion of a person dancing in a youtube video

![image](https://github.com/user-attachments/assets/151424f8-1b02-4de5-a8e4-512a99dc20a1)
