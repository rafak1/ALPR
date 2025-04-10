import os
import cv2
import numpy as np
import glob

def load_data(data_dir, patch_size=(64, 32)):
    images = []
    labels = []
    annotations = load_annotations(data_dir)
    
    for image_path in glob.glob(os.path.join(data_dir, "train", "*.jpg")):
        base_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        patches, patch_labels = get_image_patches(image, img_height, img_width, patch_size, annotations[base_name])
        
        images.extend(patches)
        labels.extend(patch_labels)
    
    return np.array(images), np.array(labels)

def load_annotations(data_dir):
    annotations = {}
    
    for image_path in glob.glob(os.path.join(data_dir, "train", "*.jpg")):
        base_name = os.path.basename(image_path)
        txt_path = image_path.replace(".jpg", ".txt")
        
        if not os.path.exists(txt_path):
            continue
        
        with open(txt_path, 'r') as f:
            boxes = f.readlines()
            annotations[base_name] = boxes
    
    return annotations

def get_image_patches(image, img_height, img_width, patch_size, boxes):
    patches = []
    patch_labels = []
    
    patch_width, patch_height = patch_size
    for y in range(0, img_height - patch_height + 1, patch_height):
        for x in range(0, img_width - patch_width + 1, patch_width):
            patch = image[y:y+patch_height, x:x+patch_width]
            label = 0
            if has_plate_in_patch(x, y, x + patch_width, y + patch_height, boxes, img_width, img_height):
                label = 1
            patches.append(patch)
            patch_labels.append(label)
    
    return patches, patch_labels

def has_plate_in_patch(x1, y1, x2, y2, boxes, img_width, img_height):
    for box in boxes:
        parts = box.strip().split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        x_center_pixel = int(x_center * img_width)
        y_center_pixel = int(y_center * img_height)
        width_pixel = int(width * img_width)
        height_pixel = int(height * img_height)
        
        box_x1 = x_center_pixel - width_pixel // 2
        box_y1 = y_center_pixel - height_pixel // 2
        box_x2 = x_center_pixel + width_pixel // 2
        box_y2 = y_center_pixel + height_pixel // 2
        
        if not (x2 < box_x1 or x1 > box_x2 or y2 < box_y1 or y1 > box_y2):
            return True
    
    return False
