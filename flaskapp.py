from flask import Flask, render_template, request
from naivebayes import NaiveBayesBinary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

app = Flask(__name__)

df=pd.read_csv('IMDBDataset.csv')
X=df['review']
y=df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

NBClf = NaiveBayesBinary()
NBClf.fit(X_train, y_train)

pred = NBClf.predict(X_test)

accuracy = accuracy_score(y_test, pred) * 100
precisionPos = precision_score(y_test, pred, pos_label='positive') * 100
precisionNeg = precision_score(y_test, pred, pos_label='negative') * 100
recallPos = recall_score(y_test, pred, pos_label='positive') * 100
recallNeg = recall_score(y_test, pred, pos_label='negative') * 100


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        pred = NBClf.predict([text])
        result = pred[0]
        return render_template('index.html', text=text, result=result, accuracy=accuracy, precisionPos=precisionPos, precisionNeg=precisionNeg, recallPos=recallPos, recallNeg=recallNeg)

if __name__ == '__main__':
    app.run(debug=True)