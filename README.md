### 0b_shoe_classification_based_on_feeback.py

## MODEL USED
"""
Load the Faster R-CNN model pre-trained on COCO dataset and adjust it for custom fine-tuning.
"""
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

## WHAT I AM TRYING TO DO
i use the resnet50 pretained model which can detect objects . and then i am trying to provide it feedback to improve its detection. when a picture shows up, the model auto detects some objects and every object is marked in a box. i provide the feedback by giving back a comma seperated list of box labels, which accurately detected the object
![image](https://github.com/user-attachments/assets/0577e7b9-ba47-4f96-8b16-d2cd60a2975f)
