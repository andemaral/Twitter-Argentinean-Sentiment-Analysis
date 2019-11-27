#https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vec = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    texto = [request.form['texto'],]
    
    if model.predict(vec.transform(texto)) == 0:
        output = 'NEGATIVO'
    else:
        output = 'POSITIVO'
 
    return render_template('index.html', prediction_text='El texto es {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)