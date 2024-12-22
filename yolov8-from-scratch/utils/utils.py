import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
import os 
import cv2 

DELTA_X = 0
DELTA_Y = 1
WIDTH = 2
HEIGHT = 3
CONFIDENCE = 4
CLASS = 5

def scale_to_range(bboxes, old_x=676, old_y=380, new_x = 128, new_y=128):
    x_scale = new_x / old_x
    y_scale = new_y / old_y
    
    for box in bboxes:
        box[0] = int(np.round(box[0]*x_scale))
        box[1] = int(np.round(box[1]*y_scale))
        box[2] = int(np.round(box[2]*x_scale))
        box[3] = int(np.round(box[3]*y_scale))
    return bboxes

def show_image_with_boxes(self, idx):
        image, boxes_list = self[idx]
        image_draw = ImageDraw.Draw(image)
        
        # Draw the bounding boxes
        for box in boxes_list:
            xmin, ymin, xmax, ymax = box
            image_draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        
        # Display the image
        plt.imshow(image)
        plt.axis("off")
        plt.show()

def compute_IoU(beta_output, beta_target): 
    x_center_output = beta_output[0]
    y_center_output = beta_output[1]
    width_output = beta_output[2]
    heigth_output = beta_output[3]

    x_center_target = beta_target[0]
    y_center_target = beta_target[1]
    width_target = beta_target[2]
    heigth_target = beta_target[3]
     
    # first convert from the YOLO format to (x_min, y_min, x_max, y_max)
    xmin_output = (2*x_center_output - width_output) / 2
    ymin_output = (2*y_center_output - heigth_output) / 2
    xmax_output = (2*x_center_output + width_output) / 2 
    ymax_output = (2*y_center_output + heigth_output) / 2

    boxA = [xmin_output, ymin_output, xmax_output, ymax_output]

    xmin_target = (2*x_center_target - width_target) / 2
    ymin_target = (2*y_center_target - heigth_target) / 2
    xmax_target = (2*x_center_target + width_target) / 2 
    ymax_target = (2*y_center_target + heigth_target) / 2

    boxB = [xmin_target, ymin_target, xmax_target, ymax_target]
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def from_grid_coordinate_to_bbox(output, size="large"):
        if size == "small":
            number_of_cells = 16
            n_pixel_per_grid = 8
        
        if size == "medium":
            number_of_cells = 8
            n_pixel_per_grid = 16
        
        if size == "large":
            number_of_cells = 4
            n_pixel_per_grid = 32
    
        bboxes = []
        for i in range(number_of_cells):
            for j in range(number_of_cells):
                
                if output[0][CLASS][i][j] >= 0.5:
                    delta_x = float(output[0][DELTA_X][i][j])
                    delta_y = float(output[0][DELTA_Y][i][j])
                    delta_w = float(output[0][WIDTH][i][j])
                    delta_h = float(output[0][HEIGHT][i][j])

                    x_a = n_pixel_per_grid * j
                    y_a = n_pixel_per_grid * i

                    x = delta_x * n_pixel_per_grid + x_a
                    y = delta_y * n_pixel_per_grid + y_a
                    w = delta_w * 128
                    h = delta_h * 128
                    confidence = float(output[0][CONFIDENCE][i][j])                 

                    bbox = [x, y, w, h, confidence]
                    #print(f"bbox = {bbox}, class = {output[0][CLASS][i][j]}")

                    bboxes.append(bbox)
        return bboxes                 


def show_image_and_bbox(image, encoding_of_boxes):
    image = image[0]
    
    if image.shape == torch.Size([3, 128, 128]):
        image_np = image.permute(1, 2, 0).numpy()
    
    # Plot the image with bounding boxes
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    
    bboxes = from_grid_coordinate_to_bbox(encoding_of_boxes)

    for box in bboxes:
        
        #print(f"box = {box}")

        x_center = box[0]
        y_center = box[1]
        width = box[2]
        height = box[3]
        
        xmin = (2*x_center - width) / 2
        ymin = (2*y_center - height) / 2
        
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        
         # Plot the center point
        ax.plot(x_center, y_center, 'ro')  # 'bo' is for blue circle marker
    
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.axis('off')  # Turn off axis
    plt.show()
    
    return 

def create_video_from_images(image_folder, output_video_path, frame_rate=30):
    # Obtain a list of all files in the folder 
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg"))]
    
    # If the list is empty, exit 
    if not images:
        print("La cartella delle immagini è vuota.")
        return
    
    # Read the first image for obtain the video's dimension
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # Add all images on video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    #Release the object 
    video.release()
    print(f"Video creato con successo: {output_video_path}")


def create_images_from_video(video_path, output_dir, ):
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If no frame is read (end of the video), break the loop
        if not ret:
            break

        # Construct the file name for the image
        image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")

        # Save the frame as a JPEG file
        cv2.imwrite(image_path, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'Extracted {frame_count} frames and saved to {output_dir}')


#Return a list of path images from folder_path
def process_images_from_folder(folder_path):
    file_list = os.listdir(folder_path)
    
    image_list = [f for f in file_list if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_list.sort()
    list_complete = []
    
    for l in image_list: 
        complete_path = os.path.join(folder_path,l)
        list_complete.append(complete_path)
    
    return list_complete