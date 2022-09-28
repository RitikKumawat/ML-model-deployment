import numpy as np
from flask import Flask,render_template,request
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict_price():
    year = int(request.form.get('year'))
    mpg = float(request.form.get('mpg'))
    engine = float(request.form.get('EngineSize'))

    result = model.predict(np.array([year,mpg,engine]).reshape(1,3))
    return render_template('index.html',result=result)
if __name__ == "__main__":
    app.run(debug=True)