import numpy as np
from flask import Flask, request, render_template
import sklearn
import xgboost
from xgboost.sklearn import XGBRegressor
import pickle
import os
app = Flask(__name__)
filename = os.path.join(app.root_path, 'MedicalData_RegressionModel.sav')
model = pickle.load(open(filename, 'rb'))
@app.route('/')
def home():
    return render_template('InputScreen.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    prediction = model.predict([features])
    # output = round(prediction, 4)
    return render_template('InputScreen.html', prediction = 'Predicted value is $ {}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)