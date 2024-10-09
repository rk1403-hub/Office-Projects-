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

class nlt:
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
        

    def preprocess(self):
        df = pd.DataFrame.from_dict(list(self.db.tickets.find({'teams':{"$exists":True}},self.filterDict)))
        print(df)
        df["teams"] = df["teams"].fillna("").apply(list)
        df = df.fillna("")
        df['text_data'] = df[['reason', 'subject', 'body', 'notes']].agg(lambda x: ' '.join(x.values), axis=1)
        df["teams"] = df["teams"].astype(str)
        df["severity"] = df["severity"].fillna("")
        df["severity"] = df["severity"].astype(str)
        # Preprocess data
        df['processed_body'] = df['text_data'].apply(preprocess)
        return df

    def splitdata(self, df):
    # Split data into training and testing sets
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        return train_data, test_data
    

    # Vectorize text using bag-of-words approach
    def vectorize(self, df, train_data, test_data):
        vectorizer = CountVectorizer(stop_words='english')
        train_features = vectorizer.fit_transform(train_data['processed_body'])
        test_features = vectorizer.transform(test_data['processed_body'])
        x_axis = vectorizer.fit_transform(df['processed_body'])
        x_train, x_test, y_train, y_test = train_test_split(x_axis, df['severity'], test_size=0.4, random_state=0)
        joblib.dump(vectorizer, "./vectorizer.joblib")
        print(" Training completed, vectorizer file generated ")
        return vectorizer, x_axis,x_train, x_test, y_train, y_test
        


nlt1 = nlt()
df = nlt1.preprocess()
train, test = nlt1.splitdata(df)
vectorizer, x_axis,x_train, x_test, y_train, y_test= nlt1.vectorize(df, train, test)

joblib.dump(df, "./df.joblib")
joblib.dump(x_axis,"./x_axis.joblib")
joblib.dump(x_train,"./x_train.joblib")
joblib.dump(x_test,"./x_test.joblib")
joblib.dump(y_train,"./y_train.joblib")
joblib.dump(y_test,"./y_test.joblib")