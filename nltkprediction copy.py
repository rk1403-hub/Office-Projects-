# %%
import pandas as pd
import pickle
import numpy as np
import os.path
import joblib

from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from helpers import make_dataset,invert_multi_hot,make_model, preprocess
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient

class nltpred:
    def __init__(self):
        client = MongoClient("mongodb+srv://prajwal:wSnhpJZel3jMaesA@staging.awytu.mongodb.net/?retryWrites=true&w=majority")


        self.db = client.dashboard


        teams = self.db.accessList.find_one({"name": "Teams"})["teams"]
        teams.remove('GridSec')

        self.filterDict = {
                
                "type": 1,
                "status":1,
                "body":1,
                "reason":1,
                "number": 1,
                "notes":1,
                "sites": 1,
                "requester": 1,
                
                "subject": 1,
                "assign":1,
                "cip_impact": 1,
                
                
                'name': 1,
                'email':1,
                "urgency":1,
                "category":1,
                "severity":1,
                'teams':1,
                "level": 1,
                
        
            }
    
    def prediction(self, teams_model, severity_model, vectorizer, description):
        processed_description = preprocess(description)
        vectorized_data = vectorizer.transform([processed_description])
        severity_prediction = severity_model.predict(vectorized_data)
        teams_prediction = teams_model.predict(vectorized_data)[0]
        
        return "\nTeam: {}\nSeverity: {}".format(teams_prediction, severity_prediction)
    
    def random_forest_classifier(self):
        df = joblib.load("./df.joblib")
        x_axis = joblib.load("./x_axis.joblib")
        x_train = joblib.load("./x_train.joblib")
        x_test = joblib.load("./x_test.joblib")
        y_train = joblib.load("./y_train.joblib")
        y_test = joblib.load("./y_test.joblib")

        rf = RandomForestClassifier(n_estimators=150, random_state=42)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
       

        accuracy = accuracy_score(y_test, y_pred)
        print("Severity Prediction Accuracy: %.2f%%" % (accuracy * 100.0))

        teams_x_train, teams_x_test, teams_y_train, teams_y_test = train_test_split(x_axis, df['teams'], test_size=0.4, random_state=0)
        return rf, teams_x_train, teams_x_test, teams_y_train, teams_y_test
        


    def teams_rf_classifier(self, rf, teams_x_train, teams_x_test, teams_y_train, teams_y_test):
        vectorizer = joblib.load("./vectorizer.joblib")
        rf_teams = RandomForestClassifier(n_estimators=150, random_state=42)
        rf_teams.fit(teams_x_train, teams_y_train)
        
        teams_y_pred = rf_teams.predict(teams_x_test)
        # Evaluate classifier on test set
        accuracy_teams = accuracy_score(teams_y_test, teams_y_pred)
        print("Teams Prediction Accuracy: %.2f%%" % (accuracy_teams * 100.0))

        #test ticket description
        description = "Site wide inverter warranty repairs"
        print(description)
        predicted_severity_team = self.prediction(rf_teams, rf, vectorizer, description)
        print("The team and severity for the above description is,", predicted_severity_team)

        model_name = 'ticket_classifier.pkl'
         
        with open(model_name, 'wb') as f:                   #Save the model
            pickle.dump(rf_teams, f)
        with open('vectorizer.pkl', 'wb') as f:             #Save the vectorizer
            pickle.dump(rf, f)
        if os.path.isfile(model_name):
            print("Model file exists")
        else:
            print("Model file does not exist")

nltp=nltpred()
rf, teams_x_train, teams_x_test, teams_y_train, teams_y_test= nltp.random_forest_classifier()
nltp.teams_rf_classifier(rf, teams_x_train, teams_x_test, teams_y_train, teams_y_test)
