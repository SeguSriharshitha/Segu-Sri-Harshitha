from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open(r"C:\Users\Administrator\Desktop\flight price\model1.pkl", 'rb'))

app = Flask(__name__)
import json
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "4oDBgnbS-Aikf8dqUWmY8Mo3lPYc3EASfN9TyomfZKv3"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


@app.route("/")
def about():
    return render_template('home.html')


@app.route("/predict")
def home1():
    return render_template('predict.html')


@app.route("/pred", methods=['POST'])
def predict():
    x = [[int(x) for x in request.form.values()]]
    print(x)
    x = np.array(x)
    print(x.shape)
    print(x)
    pred = model.predict(x)
    print(pred[0])
    return render_template('submit.html', prediction_text=pred[0])
    payload_scoring = {"input_data": [{"fields": ['Airline','Source','Destination','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date','Arrival_Time_Hour','Arrival_Time_Mins'], "values": [x]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/e52653f6-3287-4114-9910-b18179c5861d/predictions?version=2022-08-28', json=payload_scoring,
     headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    prediction=response_scoring.json()
    print(prediction)
    pred = prediction["predictions"][0]["values"][0][0]

if __name__ == "__main__":
    app.run(debug=False)
