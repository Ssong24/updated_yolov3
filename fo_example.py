import torch
import torchvision

# Run the model on GPU if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
_ = model.eval()

print("Model ready")

import fiftyone.zoo as foz
dataset_dir = foz.download_zoo_dataset("coco-2017", split="validation")
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# Print some information about the dataset
print(dataset)

# Print a sample ground truth detection
sample = dataset.first()
print(sample.ground_truth.detections[0])

# Choose a random subset of 100 samples to add predictions to
predictions_view = dataset.take(100, seed=51)

import json
import os
from PIL import Image

from torchvision.transforms import functional as func

import fiftyone as fo

# Get class list
classes = dataset.info["classes"]

# Add predictions to samples
with fo.ProgressBar() as pb:
    for sample in pb(predictions_view):
        # Load image
        image = Image.open(sample.filepath)
        image = func.to_tensor(image).to(device)
        c, h, w = image.shape

        # Perform inference
        preds = model([image])[0]
        labels = preds["labels"].cpu().detach().numpy()
        scores = preds["scores"].cpu().detach().numpy()
        boxes = preds["boxes"].cpu().detach().numpy()

        # Convert detections to FiftyOne format
        detections = []
        for label, score, box in zip(labels, scores, boxes):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

            detections.append(fo.Detection(
                label=classes[label],
                bounding_box=rel_box,
                confidence=score
            ))

        # Save predictions to dataset
        sample["faster_rcnn"] = fo.Detections(detections=detections)
        sample.save()

print("Finished adding predictions")


from fiftyone import ViewField as F

# Only keep detections with confidence >= 0.75
high_conf_view = predictions_view.filter_detections("faster_rcnn", F("confidence") > 0.75)

# Print some information about the view
print(high_conf_view)

# Print a sample prediction from the view
sample = high_conf_view.first()
print(sample.faster_rcnn.detections[0])

# Create a new `faster_rcnn_75` field on `dataset` that contains the detections
# from the `faster_rcnn` field of the samples in `high_conf_view`
new_field = "faster_rcnn_75"
dataset.clone_field("faster_rcnn", new_field, samples=high_conf_view)

# Verify that the new field was created
print(dataset)


import fiftyone.utils.eval as foue

foue.evaluate_detections(predictions_view, "faster_rcnn_75", gt_field="ground_truth")
print(predictions_view)

# Open the dataset in the App
session = fo.launch_app(dataset=dataset)

# Load view containing the subset of samples for which we added predictions
session.view = predictions_view

# The currently selected images in the App
selected_samples = session.selected

# Create a new view that contains only the selected samples
# And open this view in the App!
session.view = dataset.select(selected_samples)

# Resets the session; the entire dataset will now be shown
session.view = None

# Show samples with most true positives first
session.view = predictions_view.sort_by("tp_iou_0_75", reverse=True)

# Show samples with most false positives first
session.view = predictions_view.sort_by("fp_iou_0_75", reverse=True)

# Bounding box format is [top-left-x, top-left-y, width, height]
bbox_area = F("bounding_box")[2] * F("bounding_box")[3]

# Create a view that contains only predictions whose area is < 0.005
small_boxes_view = predictions_view.filter_detections("faster_rcnn_75", bbox_area < 0.005)

session.view = small_boxes_view

# Create a view that contains only samples for which at least one detection has
# its iscrowd attribute set to 1
crowded_images_view = predictions_view.match(
    F("ground_truth.detections").filter(F("attributes.iscrowd.value") == 1).length() > 0
)

session.view = crowded_images_view

sorted_crowded_images_view = crowded_images_view.sort_by(
    "fp_iou_0_75", reverse=True
)

session.view = sorted_crowded_images_view

session.view = predictions_view.sort_by("fp_iou_0_75", reverse=True)
