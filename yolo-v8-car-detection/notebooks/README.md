For evaluate our model we have used the IoU metric. 

### Intersection over Union

IoU is a quantitative metric used to evaluate the alignment between ground-truth and predicted boxes. It is calculated by dividing the area of overlap between the two boxes by the area of their union. A higher IoU indicates a more accurate prediction.

$$
IoU = \frac{Intersection Area}{Union Area}
$$

<img src="../images/iou.png" alt="IoU" width="447" height="213">

For completeness we report also the definition of the Average precision:

### Average Precision

$$
AP = \sum_{k = 0}^{n-1} [R(k) - R(k+1)] P(k)
$$

With $n$ number of thresholds and $R(n) = 0$, $P(n) = 1$. 

To compute the AP, we need to:
1. Generate the prediction scores using the model.
2. Convert the prediction scores to class labels.
3. Calculate the precision and recall metrics.
4. Create the precision-recall curve.
5. Measure the average precision.

Having only 1 class, the AP is equal to the mAP (mean Average Precision).

## Input Encoding

## Loss Function
We have tried two different losses: the Yolov8 loss, and a YOLOv1-like loss, opting eventually for using the latter.

### Yolo-v1 Loss Calculation

We have modified the classic Yolo-v1 loss, adding a cls loss calculation term also for no-object cell. Given position $(x,y)$ for each cell of the grid, we have that the loss is: 

$$
\sum_{x,y} (1_i^{obj} \ L_{obj} + \lambda_{no-obj}\  1_i^{no-obj} \ L_{no-obj})
$$


Where the loss function used when a cell contains an object is:

$$
L_{obj} = \lambda_{coord}L_{obj-box} + L_{obj-conf} + L_{obj-class} 
$$

With: 

$$
L_{obj-box} = (\hat{\Delta{x}} - \Delta{x})^2 + (\hat{\Delta{y}} - \Delta{y})^2 + (\sqrt{\hat{\Delta{w}}} - \sqrt{\Delta{w}})^2 + (\sqrt{\hat{\Delta{h}}} - \sqrt{\Delta{h}})^2
$$

$$
L_{obj-conf} = (\hat{{C}} - {C})^2 
$$

$$
L_{obj-cls} = (\hat{p} - {p})^2 
$$

with $C = 1$ and $p = 1$

While the loss function used when a cell does not contain an object is:

$$
L_{no-obj} = L_{no-obj-conf} + L_{no-obj-class} + L_{obj-class} 
$$

With analogous expression for the class and confidence losses, with the only difference of $C = 0$ and $p = 0$

#### Yolo-v1 Loss Hyperparameters

* `LAMBDA_COORD`: when > 1 it puts more importance on box parameters than cls and confidence
* `LAMBDA_NO_OBJ`: when < 1 it puts more importance on grid cells that contain object than cells which don't

### Yolo-v8 Loss Calculation
Just for reference, we put the description of the YOLOv8 loss function, although we decided eventually to not use it:

$$
\displaystyle\sum_{x,y} \frac{\lambda_{box}}{N_{pos}}L_{box} + \frac{\lambda_{cls}}{N_{pos}}L_{cls} + \frac{\lambda_{dfl}}{N_{pos}}L_{dfl} + \phi||\theta||_2^2
$$

Where:

$$
L_{box} = 1_{c}[1 - q_{x,y} + \frac{||b_{x,y} - \hat{b_{x,y}}||}{\rho^2} + \alpha_{x,y}v_{x,y}]
$$



$$
L_{cls} = \sum_{c \in classes}y_c\log(\hat{y_c}) + (1- y_c)\log(1 - \hat{y_c})
$$

$$
L_{dfl} = 1_{c}[- (q_{(x,y) + 1} - q_{x,y})\log(\hat{q_{x,y}}) + (q_{x,y} - q_{(x,y) - 1})\log(\hat{q_{(x,y) + 1}})]
$$



With:

$$
q_{x,y} = IoU_{x,y} = \frac{\hat{\beta_{x,y}} \cap \beta_{x,y}}{\hat{\beta_{x,y}} \cup \beta_{x,y}}
$$

$$
v_{x,y} = \frac{4}{\pi^2}(\arctan(\frac{w_{x,y}}{h_{x,y}}) - \arctan(\frac{\hat{w_{x,y}}}{\hat{h_{x,y}}}))^2
$$


$$
\alpha_{x,y} = \frac{v_{x,y}}{1 - q_{x,y}}
$$

$$
\hat{q}_{x,y} = softmax()
$$

- $N_{pos}$: number of cells containing an object
- $1_{c*_{x,y}}$: indicator function, 1 when the cell $(x,y)$ contains an object, 0 otherwise.
- $\beta_{x,y}$: the ground truth bbox
- $\hat{\beta}_{x,y}$: the predicted bbox
- $b_{x,y}$: central point of the ground truth bbox
- $b_{x,y}$: central point of the predicted bbox
- $y_c$: ground truth label for class c
- $q_{(x,y)+1}$ and $q_{(x,y)-1}$: nearest predicted boxes IoUs
- $w_{x,y}$ and $h_{x,y}$: width and height of the bbox
- $\rho$: diagonal lenght of the smallest enclosing box coverting the predicted and ground truth bbox.


## Architecture
### Building Blocks 

#### Convolutional Block
<img src="../images/convblock.png" alt="ConvBlock" width="400">

It is the most basic block in the architecture, comprising a Conv2d layer, a BatchNorm2d layer, and the SiLU activation function.

- Conv2d Layer: Convolution involves sliding a small matrix (known as a kernel or filter) over the input data, performing element-wise multiplication, and summing the results to generate a feature map. The "2D" in Conv2D refers to the application of convolution across two spatial dimensions, typically height and width. A conv layer has $k$ learnable kernels, a stride $s$, padding $p$ and $c$ input channels.

- BacthNorm2d Layer: Batch Normalization is used to enhance training stability and speed up convergence. It normalizes 2D inputs, ensuring that the values within the network are well-scaled, preventing issues during training.

- SiLU Activation Function: The SiLU (Sigmoid Linear Unit) is defined as:

  $$SiLU(x) = x * \sigma(x) = \frac{x}{1 + e^{-x}}$$

  Where $\sigma(x)$ is the sigmoid function. The key characteristic of SiLU is that it allows for smooth gradients, which can be beneficial during the training of neural networks. Smooth gradients can help avoid issues like vanishing gradients, which can impede the learning process in deep neural networks.



#### Bottleneck Block
<img src="../images/bottleneck.png" alt="ConvBlock" width="400">

The bottleneck block consists of two Conv Blocks and an optional shortcut connection. When the shortcut is enabled, it provides a direct path that bypasses the Conv Blocks, also known as residual connection, that allows the gradient to flow more easily through the network during training, addressing the vanishing gradient problem and allowing the model to choose to use the identity mapping provided by the shortcut, making it easier to learn the identity function when needed. 


#### C2f Block:
<img src="../images/c2f.png" alt="ConvBlock" width="400">

The C2f block begins with a convolutional block, after which the resulting feature map is split. One portion of the feature map is directed to a Bottleneck block, while the other bypasses it and goes straight to the Concat block. The number of Bottleneck blocks used within the C2f block is determined by the model's depth_multiple parameter. Finally, the output from the Bottleneck block is concatenated with the bypassed feature map, and this combined output is fed into a concluding convolutional block.


#### SPPF Block
<img src="../images/sppf.png" alt="ConvBlock" width="400">

The SPPF (Spatial Pyramid Pooling Fast) Block consists of an initial convolutional block followed by three MaxPool2d layers. The feature maps produced by these MaxPool2d layers plus the output of the initial conv block are then concatenated and passed through a final convolutional block. The core idea of Spatial Pyramid Pooling (SPP) is to divide the input image into a grid, pooling features from each grid cell independently. This allows the network to effectively handle images of varying sizes by capturing multi-scale information, which is especially useful for tasks like object recognition, where objects may appear at different scales within an image.

While traditional SPP can be computationally intensive due to the multiple pooling levels with different kernel sizes, SPPF (SPP-Fast) optimizes this process by using a simpler pooling scheme. SPPF may utilize a single fixed-size pooling kernel, reducing the computational burden while maintaining a trade-off between accuracy and speed.

The MaxPool2d layers in this block are responsible for downsampling the spatial dimensions of the input, which reduces computational complexity and helps extract dominant features. Max pooling retains only the maximum value within each pooling region, effectively discarding less important information. In MaxPool2d, pooling is applied across both the height and width dimensions of the input tensor.

The main function of the SPPF block is to generate a fixed feature representation of objects in various sizes within an image, without needing to resize the image or losing spatial information.

#### Detect Block
<img src="../images/detect.png" alt="ConvBlock" width="400">

The Detect Block in YOLOv8 handles object detection using an anchor-free approach, predicting object centers directly rather than relying on anchor boxes. This streamlines the process by reducing the number of box predictions and speeding up post-processing

**Important Note**: In our setup, we're using a single branch to predict bounding box coordinates, confidence and class simultaneously, though a two-branch approach is also possible.

### The Three parts of Yolo-v8

Now that we've explained the individual blocks, let's introduce the overall architecture of YOLOv8.

<img src="../images/yolo-v8-architecture.png" alt="ConvBlock" width="1001" height="913">

The architecture consists of three main sections: Backbone, Neck, and Head.

- Backbone: this is the deep learning architecture that acts as the feature extractor for the input image.

- Neck: this section combines the features from various layers of the Backbone. It is responsible for upsampling the feature map and merging features from different layers. The Concat block in this section sums the output channels of concatenated blocks without changing their resolution.

- Head: this section predicts the classes, bounding boxes coordinates and confidence of objects, producing the final output of the object detection model.


Each of the building blocks has an identifier within the architecture. Note that diagram shows an input image size of 640 x 640, but, due to the original dimension of the dataset images, we have opted for 128 x 128, which changes the dimensions of intermediate blocks and the output.

**Important Note**: we are using a single detect block, but using three detect blocks, as shown in the diagram, could allow for specialized detections based on object size:
- The first Detect block, connected to Block 15, specializes in detecting small objects, utilizing a smaller grid size.
- The second Detect block, connected to Block 18, focuses on detecting medium-sized objects.
- The third Detect block, connected to Block 21, is designed to detect large objects, utilizing a larger grid size.
