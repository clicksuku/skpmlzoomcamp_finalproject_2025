import requests
import pandas as pd
import numpy as np
import json

url_property_price = "http://127.0.0.1:8000/predict_property_price"


# List of features used in the model
interested_features = [
    'host_response_rate', 'host_acceptance_rate', 
    'host_is_superhost', 'host_identity_verified',
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'number_of_reviews', 'review_scores_rating',  
    'minimum_nights','instant_bookable', 
    'entire_home_apt', 'private_room', 'shared_room', 'hotel_room',          
    'leopoldstadt', 'others', 'margareten', 'brigittenau', 
    'landstrae', 'ottakring', 'rudolfsheim_fnfhaus', 
    'neubau', 'alsergrund', 'meidling', 'favoriten'
]

properties = [
    {"minimum_nights":31,"host_response_rate":96.0,"host_acceptance_rate":88.0,"host_is_superhost":True,"host_identity_verified":True,"accommodates":2,"bathrooms":1.0,"bedrooms":0.0,"beds":1.0,"number_of_reviews":74,"review_scores_rating":4.86,"instant_bookable":True,"entire_home_apt":True,"private_room":False,"shared_room":False,"hotel_room":False,"leopoldstadt":False,"others":False,"margareten":True,"brigittenau":False,"landstrae":False,"ottakring":False,"rudolfsheim_fnfhaus":False,"neubau":False,"alsergrund":False,"meidling":False,"favoriten":False},
    {"minimum_nights":1,"host_response_rate":100.0,"host_acceptance_rate":98.0,"host_is_superhost":False,"host_identity_verified":True,"accommodates":3,"bathrooms":1.0,"bedrooms":1.0,"beds":2.0,"number_of_reviews":345,"review_scores_rating":4.7,"instant_bookable":False,"entire_home_apt":True,"private_room":False,"shared_room":False,"hotel_room":False,"leopoldstadt":False,"others":False,"margareten":False,"brigittenau":False,"landstrae":False,"ottakring":False,"rudolfsheim_fnfhaus":False,"neubau":False,"alsergrund":False,"meidling":True,"favoriten":False},
    {"minimum_nights":2,"host_response_rate":100.0,"host_acceptance_rate":100.0,"host_is_superhost":False,"host_identity_verified":True,"accommodates":5,"bathrooms":1.0,"bedrooms":2.0,"beds":4.0,"number_of_reviews":7,"review_scores_rating":5.0,"instant_bookable":True,"entire_home_apt":True,"private_room":False,"shared_room":False,"hotel_room":False,"leopoldstadt":False,"others":True,"margareten":False,"brigittenau":False,"landstrae":False,"ottakring":False,"rudolfsheim_fnfhaus":False,"neubau":False,"alsergrund":False,"meidling":False,"favoriten":False},
    {"minimum_nights":1,"host_response_rate":100.0,"host_acceptance_rate":99.0,"host_is_superhost":False,"host_identity_verified":True,"accommodates":2,"bathrooms":2.0,"bedrooms":1.0,"beds":1.0,"number_of_reviews":2,"review_scores_rating":4.0,"instant_bookable":True,"entire_home_apt":False,"private_room":True,"shared_room":False,"hotel_room":False,"leopoldstadt":False,"others":False,"margareten":False,"brigittenau":False,"landstrae":False,"ottakring":False,"rudolfsheim_fnfhaus":False,"neubau":False,"alsergrund":False,"meidling":True,"favoriten":False},
    {"minimum_nights":2,"host_response_rate":100.0,"host_acceptance_rate":92.0,"host_is_superhost":True,"host_identity_verified":True,"accommodates":2,"bathrooms":1.5,"bedrooms":1.0,"beds":2.0,"number_of_reviews":91,"review_scores_rating":4.7,"instant_bookable":False,"entire_home_apt":True,"private_room":False,"shared_room":False,"hotel_room":False,"leopoldstadt":False,"others":True,"margareten":False,"brigittenau":False,"landstrae":False,"ottakring":False,"rudolfsheim_fnfhaus":False,"neubau":False,"alsergrund":False,"meidling":False,"favoriten":False}
]

df_new_data = pd.DataFrame(properties)

for index, row in df_new_data.iterrows():
    print(properties[index])
    json_string = json.dumps(properties[index])
    print(json_string)
    input()
    predicted_property_price = requests.post(url_property_price, json=properties[index])
    property_price=np.expm1(predicted_property_price.json()['property_price'])
    print(f"Property Price: ${property_price}")