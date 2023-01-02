"""
Application that predicts an Afrikan language given a series of text files
"""

from crypt import methods
from pyexpat import features
import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the flask class
app = Flask(__name__)

#Load the training model (Pickle file)
lg_clf = pickle.load(open('models/lg_clf.pkl', 'rb'))

#send the app to index,html when running the app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = lg_clf.predict(features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Language prediction{}'.format(output))


if __name__ == "__main__":
    app.run()
    