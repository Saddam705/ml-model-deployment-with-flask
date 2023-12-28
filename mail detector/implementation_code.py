import pickle
from flask import *
app = Flask(__name__)

cv = pickle.load(open('mail detector/cv.pkl','rb'))
clf = pickle.load(open('mail detector/clf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction==1 else -1
    return render_template('index.html',prediction=prediction,email = email)

if __name__=='__main__':
    app.run(debug=True)