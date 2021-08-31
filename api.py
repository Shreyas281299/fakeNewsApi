from flask import Flask, request
from model.agressive import Model


app = Flask(__name__)
@app.route('/')
def hello():
    return 'Fake News Detector'

@app.route('/model/fit')
def modelFit():
    my_model = Model()
    ac = my_model.fit()
    return ac

@app.route('/model/predict',methods = ['POST'])
def modelPredict():
    my_model = Model()
    res = my_model.predict(request.json)
    return res

if __name__ == "__main__":
    app.run(debug=True)