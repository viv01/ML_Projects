import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt


# def create_image_with_boxes(image, boxes, labels, scores):
#     """
#     Draw bounding boxes on the image and display it.
#     """
#     draw = ImageDraw.Draw(image)
#     for i, box in enumerate(boxes):
#         box = [int(coord) for coord in box]
#         draw.rectangle(box, outline="red", width=3)
#         draw.text((box[0], box[1] - 10), f"Label: {labels[i]}, Score: {scores[i]:.2f}", fill="red")
#     return image


def load_model(num_classes=2):
    """
    Load the Faster R-CNN model pre-trained on COCO dataset and adjust it for custom fine-tuning.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



def detect_objects(model, image, threshold=0):
   
    # Ensure the model is in evaluation mode
    model.eval()


    """
    Detect objects in the image using the model.
    """

    # Convert the image to a tensor
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    # Get model predictions
    outputs = model(image_tensor)[0]
    boxes = outputs["boxes"].detach().numpy()
    labels = outputs["labels"].detach().numpy()
    scores = outputs["scores"].detach().numpy()
    # print(boxes)
    # print(scores)
    # print(boxes.shape)

    filtered_indices = np.where(scores > threshold)[0]
    filtered_boxes = boxes[filtered_indices]
    filtered_labels = labels[filtered_indices]
    filtered_scores = scores[filtered_indices]

    return filtered_boxes, filtered_labels, filtered_scores


def create_image_with_copies(image, boxes, labels, scores, columns=4, output_file="boxes_individual.jpg"):
    """
    Create a single image with multiple copies of the original image, each highlighting one bounding box.

    Parameters:
        image (PIL.Image): The original image.
        boxes (list of tuples): Bounding boxes of detected objects.
        labels (list): Labels of detected objects.
        scores (list): Confidence scores of detected objects.
        columns (int): Number of columns in the grid.
        output_file (str): Output file name for the combined image.
    """
    from PIL import ImageFont
    
    # Image dimensions
    image_width, image_height = image.size

    # Determine the grid layout (rows x columns)
    rows = (len(boxes) + columns - 1) // columns  # Calculate required rows

    # Create a blank canvas for the grid
    grid_width = columns * image_width
    grid_height = rows * image_height
    grid_image = Image.new("RGB", (grid_width, grid_height), "white")

    # Try loading a default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Default font for headings
    except IOError:
        font = ImageFont.load_default()  # Fallback if custom font is not available

    for i, box in enumerate(boxes):
        # Create a copy of the original image
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        # Highlight only the current bounding box
        box = [int(coord) for coord in box]
        draw.rectangle(box, outline="red", width=3)  # Draw the box
        
        # Draw the heading above the box
        heading_text = f"Box {i}"
        heading_x = box[0]
        heading_y = max(0, box[1] - 30)  # Ensure text is within bounds
        draw.text((heading_x, heading_y), heading_text, fill="red", font=font)
        
        # Draw label and score near the box
        draw.text((box[0], box[1] - 10), f"Label: {labels[i]}, Score: {scores[i]:.2f}", fill="red", font=font)

        # Paste the modified image into the grid
        row, col = divmod(i, columns)
        x_offset = col * image_width
        y_offset = row * image_height
        grid_image.paste(image_copy, (x_offset, y_offset))

    # Save the final grid image
    #grid_image.save(output_file)
    #print(f"Saved combined image as: {output_file}")
    return grid_image


class FeedbackDataset(Dataset):
    """
    Dataset class for fine-tuning based on user feedback.
    Stores images, bounding boxes, and labels.
    """
    def __init__(self):
        self.images = []
        self.targets = []

    def add_data(self, image, boxes, labels):
        self.images.append(F.to_tensor(image))
        self.targets.append({"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels)})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # return self.images[idx], self.targets[idx]
        image = self.images[idx]
        boxes = self.targets[idx]["boxes"]
        labels = self.targets[idx]["labels"]
        return image, boxes, labels

def get_feedback_from_user(image, boxes, labels, scores):
    image.show()

    print("Detected boxes (indices start from 0):")
    for i, box in enumerate(boxes):
        print(f"{i}: {box}, Label: {labels[i]}, Score: {scores[i]:.2f}")
    feedback = input("Enter indices of correct shoe boxes (comma-separated), or press Enter to skip: ")
    print(feedback)
    return feedback


# def fine_tune_model(model, feedback_dataset, num_epochs=5, learning_rate=0.001):
#     """
#     Fine-tune the model using the feedback dataset.
#     """
#     data_loader = DataLoader(feedback_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()

#     for epoch in range(num_epochs):
#         for images, targets in data_loader:
#             images = list(img for img in images)
#             targets = [{k: v for k, v in t.items()} for t in targets]

#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item():.4f}")

def fine_tune_model(model, feedback_data):
    """
    Fine-tune the model using user-provided feedback.
    :param model: The Faster R-CNN model.
    :param feedback_data: A list of feedback tuples (image, correct_boxes, labels).
    """
    model.train()  # Set the model to training mode

    data_loader = DataLoader(feedback_data, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 5

    for epoch in range(num_epochs):
        for images, boxes, labels in data_loader:
            images = list(image for image in images)
            targets = [{"boxes": box, "labels": label} for box, label in zip(boxes, labels)]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

    model.eval()  # Switch back to evaluation mode



def main():

    # Input and output folders
    input_folder = "./shoes/training/"
    os.makedirs("./shoes/feedback_output/", exist_ok=True)

    #############################################################################

    # Load the pre-trained model
    model = load_model(num_classes=2)
    feedback_dataset = FeedbackDataset()

    #############################################################################

    # Process images in the folder
    for image_name in os.listdir(input_folder):
        
        #--------------------------------------------------------

        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(input_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        print(image_path)

        #--------------------------------------------------------

        # Detect objects
        boxes, labels, scores = detect_objects(model, image)

        print("^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(boxes)
        print(labels)
        print(scores)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^")

        #--------------------------------------------------------

        # # Display the image with bounding boxes
        # output_image = create_image_with_boxes(image.copy(), boxes, labels, scores)
        # output_image.show()
        
        # Display the image with bounding boxes
        output_image = create_image_with_copies(image.copy(), boxes, labels, scores, columns=4)

        # Save the output image
        output_image.save(f"./shoes/feedback_output/{image_name}")

        #output_image.show()

        #--------------------------------------------------------

        print(f"Image: {image_name}")

        # Get feedback from the user
        feedback = get_feedback_from_user(output_image, boxes, labels, scores)

        #--------------------------------------------------------

        # Update the feedback dataset
        if feedback.strip():
            correct_indices = [int(idx) for idx in feedback.split(",")]
            correct_boxes = [boxes[i] for i in correct_indices]
            correct_labels = [1] * len(correct_boxes)  # Label for "shoe"
            # print("*****")
            # print(correct_boxes)
            # print(type(correct_boxes))
            # print(correct_labels)
            # print(type(correct_labels))
            # print("*****")
            feedback_dataset.add_data(image, correct_boxes, correct_labels)

        print("***********")
        print(correct_indices)
        print(correct_boxes)
        print(correct_labels)
        print("***********")

        #--------------------------------------------------------

    print(type(feedback_dataset))  
    print(feedback_dataset) 

    #############################################################################

    # Fine-tune the model with the feedback dataset
    if len(feedback_dataset) > 0:
        print("Fine-tuning the model based on feedback...")
        fine_tune_model(model, feedback_dataset)

    #############################################################################
    

    # Test on new images
    print("Testing the fine-tuned model on new images...")

    test_folder = "./shoes/test_one/"

    #############################################################################

    # Process images in the folder
    for image_name in os.listdir(test_folder):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(test_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        boxes, labels, scores = detect_objects(model, image)

        # Display the image with bounding boxes after fine-tuning
        output_image = create_image_with_copies(image.copy(), boxes, labels, scores, columns=4)

        # Save the output image
        output_image.save(f"./shoes/feedback_output/test_{image_name}")
        output_image.show()



if __name__ == "__main__":
    main()
