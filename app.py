
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



app1 = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app1.route('/')
def home():
    return render_template('index.html')

@app1.route('/predict', methods = ['POST'])
def predict():
    #for rendering result on flask
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    print(output)
    return render_template('index.html', prediction_text = 'Employee Salary should be ${}'.format(output))


if __name__ == "__main__":
    app1.run(debug=True)
