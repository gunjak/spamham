from flask import Flask,request,render_template
from  check import Pipeline 
import dill
import pandas as pd

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['GET','POST'])
def index():
    pred=None
    if request.method=='POST':
        msg=request.form['message']
        df=pd.DataFrame({'message':[msg]})
        with open('transformed_obj_file','rb') as file:
            transformed_obj=dill.load(file)
        transformed_obj=transformed_obj.transform(df)
        model=Pipeline.fetch_model()
        pred=model.predict(transformed_obj)
        return render_template('index.html',predict=pred)
    return render_template('index.html',predict=pred)    
if __name__=='__main__':
    app.run()
    