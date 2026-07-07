import cv2
import numpy as np
import os

def load_mask(mask_path):

    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    return mask

def get_binary_mask(mask,class_id):
    return (mask==class_id).astype(np.uint8)

def find_contours(binary_mask):

    contours,hierarchy=cv2.findContours(binary_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    simplified_contours=[]

    for contour in contours:

        epsilon=0.001*cv2.arcLength(contour,True)

        simplified=cv2.approxPolyDP(contour,epsilon,True)

        simplified_contours.append(simplified)

    return simplified_contours

def contour_to_polygon(contour,mask):

    polygon=[]
    height,width=mask.shape


    for point in contour:
        x,y=point[0]

        x_norm=x/width
        y_norm=y/height
        polygon.extend([float(x_norm),float(y_norm)])

    return polygon

def create_yolo_label(class_id, polygon):

    line=(str(class_id)+ " "+ " ".join(map(str, polygon)))

    return line

def save_labels(output_path,labels):

    with open(output_path,"w") as f:

        for label in labels:

            f.write(label + "\n")

def process_split(mask_dir,output_dir):
    
    os.makedirs(output_dir,exist_ok=True)

    for filename in os.listdir(mask_dir):

        if not filename.endswith(".png"):
            continue

        mask_path=os.path.join(mask_dir,filename)

        mask=load_mask(mask_path)

        labels=[]

        for class_id in np.unique(mask):

            if class_id==0:
                continue

            yolo_class = class_id - 1

            binary_mask=get_binary_mask(mask,class_id)
            contours=find_contours(binary_mask)

            for contour in contours:

                polygon=contour_to_polygon(contour,mask)

                if len(polygon)<8:
                    continue

                label=create_yolo_label(yolo_class,polygon)

                labels.append(label)
        
        txt_filename=filename.replace(".png",".txt")

        output_path=os.path.join(output_dir,txt_filename)

        save_labels(output_path,labels)

        print(f"Processed {filename}")

if __name__=="__main__":
    
    DATASET_ROOT = "FoodSeg103"

    splits = ["train", "test"]

    for split in splits:

        mask_dir = os.path.join(DATASET_ROOT,"Images","ann_dir",split)

        output_dir = os.path.join(DATASET_ROOT,"labels",split)

        print(f"\nProcessing {split} set...")

        process_split(mask_dir, output_dir)

    print("\nDataset conversion complete")