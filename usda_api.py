import os
import requests
from dotenv import load_dotenv
from nutrition_db import USDA_MAPPING

DEBUG=True

load_dotenv()

USDA_API=os.getenv("USDA_API")

CALORIE_CACHE={}

def search_food(food_name):

    url="https://api.nal.usda.gov/fdc/v1/foods/search"

    params={"api_key":USDA_API}

    body={"query":food_name,"pageSize":10}

    response=requests.post(url,params=params,json=body)

    response.raise_for_status()

    return response.json()

def get_best_match(foods):

    for food in foods:

        description=food["description"].lower()

        if "raw" in description:
            return food

    preferred = [
        "Foundation",
        "SR Legacy",
        "Survey (FNDDS)"
    ]

    for datatype in preferred:
        for food in foods:

            if food["dataType"]==datatype:
                return food
    
    return foods[0]

def get_calorie_per_100g(food_name):

    if food_name in CALORIE_CACHE:
        return CALORIE_CACHE[food_name]

    search_term=USDA_MAPPING.get(food_name,food_name)

    data=search_food(search_term)

    if not data["foods"]:
        return None
    
    food = get_best_match(data["foods"])

    if DEBUG:
        print()
        print("Search term:", search_term)
        print("Matched food:", food["description"])
        print("Type:", food["dataType"])
        print()

    for nutrient in food["foodNutrients"]:

        if "Energy" in nutrient["nutrientName"]:

            calories=nutrient["value"]

            CALORIE_CACHE[food_name]=calories

            return calories
        
    return None

if __name__ == "__main__":

    print(get_calorie_per_100g("pork"))
    print(get_calorie_per_100g("carrot"))