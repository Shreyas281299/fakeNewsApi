# Check Logs to see if the Model is working or not
import os
from flask.json import jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

class Model:
        def __init__(self):
            self.BASEDIR = '/Users/apple/Desktop/Epicenter/fakeNew/' 
            #Input the Base Directory where datasets are present
            try:
                self.model = joblib.load(self.BASEDIR+'savedModels/fakeModel.model')
                self.tfidf_vectorizer = joblib.load(self.BASEDIR+'/savedModels/tfidf.vec')
                #print('Modeled model')             #DEBUGING     
            except :
                self.model = None
                self.tfidf_vectorizer = None

        def fit(self):
            train = self.BASEDIR + 'train.csv'
            self.df = pd.read_csv(train)
            self.df.loc[(self.df['label'] == 1) , ['label']] = 'FAKE'
            self.df.loc[(self.df['label'] == 0) , ['label']] = 'REAL'
            labels = self.df['label']
            
            x_train,x_test,y_train,y_test=train_test_split(self.df['text'].values.astype('str'), labels, test_size=0.33, random_state=1)

            #print('Splitted')              #DEBUGING
            self.tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
            tfidf_train=self.tfidf_vectorizer.fit_transform(x_train) 
            tfidf_test=self.tfidf_vectorizer.transform(x_test)
            self.model=PassiveAggressiveClassifier(max_iter=50)
            self.model.fit(tfidf_train,y_train)

            #print('Trained')               #DEBUGING
            y_pred=self.model.predict(tfidf_test)
            score=accuracy_score(y_test,y_pred)
            
            #print(f'Accuracy: {score}%')    #DEBUGING
            joblib.dump(self.model,self.BASEDIR+'/savedModels/fakeModel.model')
            joblib.dump(self.tfidf_vectorizer,self.BASEDIR+'/savedModels/tfidf.vec')
            return jsonify({"Accuracy ": score,})

        def predict(self, x_test):
            #print(x_test.head(20))            #DEBUGING
            if not os.path.isfile(self.BASEDIR+'/savedModels/fakeModel.model'):
                return jsonify({"Message" : "Train a Model First"})
            try:
                x_test = pd.DataFrame(x_test,index = [0])
                x_test.head()
                tfidf_test=self.tfidf_vectorizer.transform(x_test['text'].values.astype('str'))
                y_pred = self.model.predict(tfidf_test)
                #print(y_pred)                  #DEBUGING
                return jsonify({"Label" : y_pred[0]})
            except:
                #print("Wrong data")            #DEBUGING
                return jsonify({"Message":'Invalid Data Input'})

if __name__ == "__main__":
    print('Inside agressive')                   # Check Logs to see if model is working or not

