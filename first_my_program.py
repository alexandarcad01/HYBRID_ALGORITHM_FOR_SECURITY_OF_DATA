from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
face_flag = 0
regressor = None  
# Define regressor globally

@app.route('/heart', methods=['POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        data = request.json['data_set']
        result = input(data)
        return result

def data_analysis():
    global regressor
    dataset = pd.read_csv('diabetes3.csv')
    columnlist = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'Outcome']
    for i in columnlist:
        labelencoder_X = LabelEncoder()
        dataset[i] = labelencoder_X.fit_transform(dataset[i])
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = LogisticRegression()
    regressor.fit(X_train, Y_train)
    classifier = SVC(kernel='linear', C=1, random_state=0, probability=True)
    classifier.fit(X_train, Y_train)
    probs = classifier.predict_proba(X_test)
    onlyprobs = probs[:, 1]
    auc = roc_auc_score(Y_test, onlyprobs)
    print('AUC: %.3f' % auc)
    Y_pred = regressor.predict(X_test)
    print("ACC:", roc_auc_score(Y_test, Y_pred))

def input(s):
    data_analysis()
    patient_condition = 'Not found'
    x = list(map(float, s))
    Y_pred = regressor.predict([x])
    if Y_pred[0] == 1:
        patient_condition = 'Critical'
        print("Patient is critical")
    else:
        patient_condition = 'Normal'
        print("Patient is normal")
    return {"data": patient_condition}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008)