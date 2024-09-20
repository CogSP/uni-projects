# YOLOv8 Implementation From Scratch for Car Detection

In this project, we have re-implemented the YOLOv8 architecture by [Ultralytics](https://docs.ultralytics.com/models/yolov8/), using it for car detection.

<img src="images/car-detection.jpeg" alt="car_detection" width="393" height="219">

## Project description
We reimplemented the YOLOv8 architecture completely from scratch and used it for car object detection. Object detection is a computer vision task that involves identifying and locating objects within an image or video. It goes beyond simply recognizing what objects are present (object classification) by also determining where these objects are situated.

<img src="images/yolo-v8-introduction.png" alt="yolo-v8-intro" width="500" height="362">

YOLO models are well-regarded for their efficiency, making them popular choices for applications ranging from autonomous vehicles to video surveillance and more. YOLOv8 is the eighth iteration of the YOLO (You Only Look Once) series, which is a popular family of real-time object detection models. YOLOv8 builds on the advancements made in previous versions, offering improved performance in terms of speed and accuracy.

## Results
As follow you can see the model's results:

<p align="left">
  <img src="images/Result1.png" alt="Result 1" width="400"/>
  <img src="images/Result2.png" alt="Result 2" width="400"/>
</p>
<p align="left">
  <img src="images/Result6.png" alt="Result 3" width="400"/>
  <img src="images/Result4.png" alt="Result 4" width="400"/>
</p>

The model's results are shown above, demonstrating various scenarios, from single to multiple detections within an image. The predictions closely align with the ground truth.

## Dataset

We used the Kaggle [Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) dataset. The dataset, containing images of cars in all views, is originally divided in training and test set, and contains a csv with the label $(id, x_{max}, y_{max}, x_{min}, y_{min})$ for the training.

<img src="images/dataset1.png" alt="dataset" width="540" height="540">

#### Annotation of the Test Set
Since the test set has no ground truth, we used a [Software](https://annotate.photo/) to annotate it, getting the ground truth values in JSON format. We then wrote a simple parser in order to obtain the encoded ground truth, that we can compare with the model predictions.  

## Metrics for Evaluation

We quantitavely test the model using IoU for each bounding box come metric.
The average precision is not so relevant in our case, since we have only one class to predict and generally 1 or few bounding boxes. 
For more detailed information on the metrics and architecture used, please refer to the [README in the notebook folder](/notebooks/README.md) of this repository.

## Run the code 
To start the training phase, you can download and run the [train-yolo.ipynb](https://github.com/CogSP/Yolov8-Car-Detection/blob/main/notebooks/train-yolo.ipynb) notebook.

## Testing 
To start the testing phase, you can download and run the [test-yolo.ipynb](https://github.com/CogSP/Yolov8-Car-Detection/blob/main/notebooks/test-yolo.ipynb) notebook. 

 #### Note for train and test phase: 

 If you are working with Kaggle, the dataset used is already available in the file and the site automatically download and add it.


## Acknowledgments

- [Real-Time Flying Object Detection with YOLOv8](https://arxiv.org/pdf/2305.09972)
- [Ultralytics Github](https://github.com/ultralytics)
