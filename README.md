# fakeNewsApi.  
Fake News Detection API 

### agressive.py
preprocessing - TFIDF Vectorizer
model - Passive Agreesive Model
methods : 

  fit - creates a new model and tfidf vector if already not present. If the user tries to predict before creating a model the function will return a json response asking to fit a model first

  perdict - takes a data in json format which must has 'text' in it. If user tries to predict before creating the model ,json respnse will promp to fit the model first and if there is no 'text' field it will raise a invalidValueError 
          
          
### api.py
endpoint : '/model/fit' used to fit the model. - GET
           'model/predict' used for predicting with the model - POST 

Epicenter News
