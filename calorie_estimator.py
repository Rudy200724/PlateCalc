from ultralytics import YOLO
import cv2
import numpy as np
from nutrition_db import COUNTABLE_FOODS
import matplotlib.pyplot as plt
from usda_api import get_calorie_per_100g

CONF_THRESHOLD=0.5
AREA_SCALE_FACTOR=0.0015

model=YOLO("model/best.pt")

def show_results(result):
    annotated=result.plot()
    annotated=cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)

    plt.imshow(annotated)
    plt.axis("off")
    plt.show()

def estimate_weight(name,area,count=None):

        if name in COUNTABLE_FOODS:
            if count is not None:
                return count*COUNTABLE_FOODS[name]

        return area*AREA_SCALE_FACTOR

def analyze_image(image_path):
    
    DEBUG=False

    results= model(image_path)
    result=results[0]

    height,width=result.orig_shape

    image_area=height*width

    if DEBUG:
        print()
        print(f"image area: {image_area}")
        print()

    if result.masks is None:
        print("No food detected")
        return [], result

    food_data=[]

    for i,mask in enumerate(result.masks.data):

        conf=float(result.boxes.conf[i])

        if conf<CONF_THRESHOLD:
            continue

        area=mask.cpu().sum().item()

        cls_id=int(result.boxes.cls[i])

        name=model.names[cls_id]

        percentage=(area/image_area)*100

        count=None

        if name in COUNTABLE_FOODS:

            mask_np=mask.cpu().numpy().astype(np.uint8)

            num_labels,labels=cv2.connectedComponents(mask_np)

            count=num_labels-1

            if DEBUG:
                print(f"{name}: {count} objects")

        weight=estimate_weight(name,area,count)

        food_data.append(
            {
            "class_id":cls_id,
            "name":name,
            "area":area,
            "percentage":percentage,
            "weight":weight,
            "count":count,
            "confidence":conf
            })
    
    total_calories=0

    for food in food_data:

        calories_per_100g=get_calorie_per_100g(food["name"])

        if calories_per_100g is None:
             print(f"No USDA match for {food['name']}")
             continue
        
        calories=(food["weight"]*calories_per_100g/100)

        food["calories"]=calories

        total_calories+=calories
    
    for food in food_data:

        if DEBUG:
            print(
                f"{food['name']:<20}"
                f" weight={food['weight']:>7.1f}g"
                f" calories={food.get('calories',0):>7.1f}"
                f" count={food['count']}"
                f" conf={food['confidence']:.2f}"
            )
    
    if DEBUG:
        print()
        print("-" * 50)
        print(f"TOTAL CALORIES: {total_calories:.1f} kcal")
    
    return food_data,total_calories,result

if __name__ == "__main__":

    foods,total_calories,result = analyze_image(r"FoodSeg103\Images\img_dir\test\00006128.jpg")