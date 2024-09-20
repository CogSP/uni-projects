import numpy as np 
from torch.utils.data import Dataset
import pandas as pd 
import torch 
import os
from PIL import Image
from utils import scale_to_range

# WHAT TO DO WITH THE INPUT:
# STARTING FROM IMAGE_ID AND 4 PARAMETERS FOR BBOX
# DIVIDE THE IMAGE IN GRID SIZE (128 -> 16, 8 AND 4)
# CALCULATE THE CENTER OF THE OBJECTS IN THE IMAGE BY CALCULATING THE CENTER OF THE BOUNDING BOX -> WE HAVE IT
# CALCULATE THE CELL IN WHICH THE CENTER LIES: THAT IS THE CELL RESPONSIBLE OF CALCULATING THE BBOX
# TRANSFORM THE XYWH QUANTITIES IN DELTA_X DELTA_Y DELTA_W DELTA_H 
# ADD CONFIDENCE 100% AND CLASS PROBABILITY 1 (CAR) FOR THAT GRID
# PUT ALL ZEROS IN ALL THE OTHER CELLS IN WHICH WE DON'T HAVE OBJECTS
class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = self.annotations['image'].unique()
        
        
    def transform_in_grid_coordinates(self, x_center, y_center, width, height, size="large"):
        if size == "small":
            number_of_cells = 16
            n_pixel_per_grid = 8
        
        if size == "medium":
            number_of_cells = 8
            n_pixel_per_grid = 16
        
        if size == "large":
            number_of_cells = 4
            n_pixel_per_grid = 32
        
        i = 0
        for cell in range(number_of_cells * number_of_cells):
            if cell % number_of_cells  == 0:
                i += 1
            
            x_a = n_pixel_per_grid * (cell % number_of_cells)
            y_a = n_pixel_per_grid * (i-1) 
            
            if x_center >= x_a and x_center <= (x_a + n_pixel_per_grid):
                if y_center >= y_a and y_center <= (y_a + n_pixel_per_grid):
                    delta_x = (x_center - x_a) / n_pixel_per_grid
                    delta_y = (y_center - y_a) / n_pixel_per_grid
                    delta_width = width / 128
                    delta_height = height / 128 # 128 is the heigth and width of the image
                    confidence = 1 # 100% confidence that is a car
                    cl = 1 # 1 = car, 0 = nothing
                    column = cell
                    row = i-1
                    return delta_x, delta_y, delta_width, delta_height, confidence, cl, column%number_of_cells, row
                else:
                    continue
            else:
                  continue
            print("[ERROR]: there is a box but we have not found the cell in which it lies")
            

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]    
        
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
                    
        boxes = self.annotations[self.annotations['image'] == image_id][['xmin', 'ymin', 'xmax', 'ymax']].values
        #print(f"ID: {image_id}, shape: {boxes.shape}\n")
        boxes = boxes.astype(float)
        
        boxes = scale_to_range(boxes)
        
        batch_size = 1 
        number_of_cells = 4
        label_encoding = torch.zeros(6, number_of_cells, number_of_cells)

        N_POS = 0
        for i in boxes:
            N_POS +=1
            
            xmin, ymin, xmax, ymax = i     
            x_center = int((xmin + xmax) / 2.0)
            y_center = int((ymin + ymax) / 2.0)
            width = int(xmax - xmin)
            height = int(ymax - ymin)
            delta_x, delta_y, delta_w, delta_h, confidence, cl, column, row = self.transform_in_grid_coordinates(x_center, y_center, width, height, size="large")
            
            box = [delta_x, delta_y, delta_w, delta_h, confidence, cl]
            
            label_encoding[0][row][column] = box[0]
            label_encoding[1][row][column] = box[1]
            label_encoding[2][row][column] = box[2]
            label_encoding[3][row][column] = box[3]
            label_encoding[4][row][column] = box[4]
            label_encoding[5][row][column] = box[5]
            
        return image, label_encoding, N_POS