# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:08:50 2021

@author: ASUS
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('warehouse_rf_base.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        Customer_care_calls = int(request.form['Customer_care_calls'])
        Cost_of_the_Product = int(request.form['Cost_of_the_Product'])
        Prior_purchases = int(request.form['Prior_purchases'])
        Discount_offered = int(request.form['Discount_offered'])
        Weight_in_gms = int(request.form['Weight_in_gms'])
        Gender=request.form['Gender']
        if(Gender=='Male'):
            Gender=0
        else:
            Gender=1
        Customer_rating=request.form['Customer_rating']
        if(Customer_rating==1):
            Customer_rating=int(1)
        elif(Customer_rating==2):
            Customer_rating=int(2)
        elif(Customer_rating==3):
            Customer_rating=int(3)
        elif(Customer_rating==4):
            Customer_rating=int(4)
        else:
            Customer_rating=int(5)	
        Warehouse_block=request.form['Warehouse_block']
        if(Warehouse_block=='A'):
            WB_A=1
            WB_B=0
            WB_C=0
            WB_D=0
            WB_F=0
        elif(Warehouse_block=='B'):
            WB_A=0
            WB_B=1
            WB_C=0
            WB_D=0
            WB_F=0
        elif(Warehouse_block=='C'):
            WB_A=0
            WB_B=0
            WB_C=1
            WB_D=0
            WB_F=0
        elif(Warehouse_block=='D'):
            WB_A=0
            WB_B=0
            WB_C=0
            WB_D=1
            WB_F=0
        else:
            WB_A=0
            WB_B=0
            WB_C=0
            WB_D=0
            WB_F=1
        Mode_of_Shipment=request.form['Mode_of_Shipment']
        if(Mode_of_Shipment=='Flight'):
            MS_Ship=0
            MS_Road=0
            MS_Flight=1
        elif(Mode_of_Shipment=='Road'):
            MS_Ship=0
            MS_Road=1
            MS_Flight=0
        else:
            MS_Ship=1
            MS_Road=0
            MS_Flight=0
        Product_Importance=request.form['Product_Importance']
        if(Product_Importance=='High'):
            PI_high=1
            PI_medium=0
            PI_low=0
        elif(Product_Importance=='Medium'):
            PI_high=0
            PI_medium=1
            PI_low=0
        else:
            PI_high=0
            PI_medium=0
            PI_low=1
        prediction=model.predict([[Customer_care_calls, Customer_rating, Cost_of_the_Product, Prior_purchases, Gender, Discount_offered, Weight_in_gms, WB_A, WB_B, WB_C, WB_D, WB_F, MS_Flight, MS_Road, MS_Ship, PI_high, PI_low, PI_medium]])
        if prediction<0:
            return render_template('index.html',prediction_texts="Sorry, Could you fill in the details without any errors!")
        elif(prediction==0):
            return render_template('index.html',prediction_text="Congrats! Item will reach on time")
        elif(prediction==1):
            return render_template('index.html',prediction_text="Sorry, Item will not reach on time")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)