from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('heart_prediction_project.pickle', 'rb'))
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    data1 = float(request.form['male'])
    data2 = float(request.form['age'])
    data3 = float(request.form['currentsmoker'])
    data4 = float(request.form['cigaretteperday'])
    data5 = float(request.form['bpmeds'])
    data6 = float(request.form['prevalantstroke'])
    data7 = float(request.form['prevalenthyp'])
    data8 = float(request.form['diabetes'])
    data9 = float(request.form['totchol'])
    data10 = float( request.form['sysbp'])
    data11 = float(request.form['diabp'])
    data12 = float(request.form['bmi'])
    data13 = float(request.form['heartrate'])
    arr = np.array([[data1, data2, data3, data4,data5, data6, data7, data8,data9,data10,data11,data12,data13]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)
    
    
    
    
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)