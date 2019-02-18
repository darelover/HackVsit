from django.shortcuts import render
from django.http import HttpResponse

import datetime
import pickle
import json
from rest_framework.decorators import api_view
from AiForFarmers.settings import BASE_DIR
import numpy as np
import pandas as pd
from keras import backend as K

import image_converter

@api_view(['GET'])
def __index__function(request):
    start_time = datetime.datetime.now()
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_time_ms = (elapsed_time.days * 86400000) + (elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000)
    return_data = {
        "error" : "0",
        "message" : "Successful",
        "restime" : elapsed_time_ms
    }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')

@api_view(['POST','GET'])
def predict_plant_disease(request):
    try:
        if request.method == "POST" :
            if request.body:
                print("HIT-Disease")
                request_data = request.data["plant_image"]
                header, image_data = request_data.split(';base64,')
                image_array, err_msg = image_converter.convert_image(image_data)
                if err_msg == None :
                    image_array = np.array(image_array,dtype=np.float16)/225.0
                    image_array = image_array.reshape((1,256,256,3))
                    model_file = f"{BASE_DIR}/ml/cnn_model.pkl"
                    saved_classifier_model = pickle.load(open(model_file,'rb'))
                    prediction = saved_classifier_model.predict(image_array) 
                    K.clear_session()
                    label_binarizer = pickle.load(open(f"{BASE_DIR}/ml/label_transform.pkl",'rb'))
                    ans = label_binarizer.inverse_transform(prediction)[0]

                    return_data = {
                        "error" : "0",
                        "data" : f"{ans}"
                    }
                else :
                    return_data = {
                        "error" : "4",
                        "message" : f"Error : {err_msg}"
                    }
            else :
                return_data = {
                    "error" : "1",
                    "message" : "Request Body is empty",
                }
        elif request.method == "GET":
            return_data = {
                "error" : "0",
                "message" : "Plant Disease Recognition Api. Request a POST request"
            }

    except Exception as e:
        return_data = {
            "error" : "3",
            "message" : f"Error : {str(e)}",
        }
    print(return_data)
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')


@api_view(['POST','GET'])
def predict_crop(request):
    try:
        if request.method == "POST" :
            if request.body:
                print("Hit-Crop")
                soil_type = request.data["soil"]
                mon = request.data["month"]
                loc = request.data["loc"]
                soildf = pd.read_csv(f"{BASE_DIR}/ml/soil_codes.csv")
                mondf = pd.read_csv(f"{BASE_DIR}/ml/month_codes.csv")
                locdf = pd.read_csv(f"{BASE_DIR}/ml/states_codes.csv")
                # print("Hello",loc ," ", type(loc))

                soil_type = soildf.loc[soildf['Soil']==str(soil_type)]['Code']
                soil_type = soil_type.item()
                mon = (mondf.loc[mondf['Month']==str(mon)]['Code']).item()
                loc = (locdf.loc[locdf['State']==loc]['Code']).item()

                # print("mon: ")

                model_file = f"{BASE_DIR}/ml/crops_model.pkl"
                saved_classifier_model = pickle.load(open(model_file,'rb'))
                ar = np.array([soil_type,mon,loc]).reshape((-1,1))
                df = pd.DataFrame()
                df["I"] = np.array([soil_type])
                df["R"] = np.array([mon])
                df["Y"] = np.array([loc])
                
                
                # print( df.iloc[[0]] )
                prediction = saved_classifier_model.predict(df.iloc[[0]])
                prediction = list(prediction[0]) 
                K.clear_session()
                # print("Prediction: ",type(prediction))

                cropdf = pd.read_csv(f"{BASE_DIR}/ml/crop_codes.csv")
                ans=""
                for index , i in enumerate(prediction):
                    # print("i: ",type(i), i)
                    # print("ans: ",ans)
                    if i > 0.2:
                        ans+= " "+ cropdf.iloc[[index]]['CropType'].item()
                # print("Hello")
                #crop = cropdf.loc[cropdf['Code']==prediction]
                # print("Hello")

                return_data = {
                    "error" : "0",
                    "data" : ans
                }

            else :
                return_data = {
                    "error" : "1",
                    "message" : "Request Body is empty",
                }
        elif request.method == "GET":
            return_data = {
                "error" : "0",
                "message" : "Crop Prediction Api. Request a POST request"
            }

    except Exception as e:
        return_data = {
            "error" : "3",
            "message" : f"Error : {str(e)}",
        }
    print(return_data)
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')