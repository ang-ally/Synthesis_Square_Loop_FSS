from flask import *
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

app=Flask(__name__, static_url_path='/static')
model_d=pickle.load(open('model_for_d.pkl', 'rb'))
model_s=pickle.load(open('model_for_s.pkl', 'rb'))
@app.route('/')

def Home():
    return render_template('index.html')
    
@app.route("/add", methods = ['POST'])
def predict():
    if request.method == 'POST':
        h=float(request.form.get("hi", False))
        fr=float(request.form.get("fri", False))
        fl=float(request.form.get("fli", False))
        fh=float(request.form.get("fhi", False))
        g=float(request.form.get("gi", False))
        bw=float(fh-fl)
        fbw=float(bw/fr)
        prediction=model_d.predict([[h,fr,fl,fh,bw,fbw,g]])
        d_pred=prediction[0]
        out=list()
        out.append(d_pred)
        df=pd.read_excel('final fr4 ds.xlsx')
        X=df[['h','fr','fl','fh','bw','fbw','g']]
        Y=df[['s']]
        scale_in=RobustScaler()
        scale_out=RobustScaler()
        x=scale_in.fit_transform(X)
        y=scale_out.fit_transform(Y)
        prediction=scale_in.transform([[h,fr,fl,fh,bw,fbw,g]])
        s_pred=model_s.predict(prediction)
        s_pred=s_pred.reshape(-1,1)
        s_pred=scale_out.inverse_transform(s_pred)
        s_pred=s_pred.reshape(1,-1)
        s_pred=list(s_pred)
        out.append(s_pred)
        return render_template('index.html', results = out)
    else:
        return render_template('index.html')
        

if __name__ == "__main__":
    app.run(debug = True)



