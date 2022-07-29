import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import dill
from logger import logging
from fraudException import FraudException
import sys


class Pipeline:
    def data_ingestion(self):
        try:
            df=pd.read_csv("https://raw.githubusercontent.com/krishnaik06/SpamClassifier/master/smsspamcollection/SMSSpamCollection",sep='\t')
            df.columns=['label','message']
            df.to_csv('smsspamcollection.csv')
            logging.info('data ingestion has been completed')
        except Exception as e:
            raise FraudException(e,sys) from e
            
         
    def data_transformation(self):
        try:
            df=pd.read_csv('smsspamcollection.csv') 
            corpus=[]
            stemmer=PorterStemmer()
            for i in range(0,len(df)):
                review=re.sub('[^a-z0-9A-Z]'," ",df['message'][i])
                review=review.lower()
                review=review.split()
                review=[stemmer.stem(word)for word in review if not word in stopwords.words('english')]
                corpus.append(" ".join(review))
            preprocess_obj=CountVectorizer()
            X=preprocess_obj.fit_transform(corpus).toarray()
            y=pd.get_dummies(df['label'])
            y=y.iloc[:,1]
            transformed_data=np.c_[X,np.array(y)]
            with open('transformed_data_file', 'wb') as file_obj:
                np.save(file_obj, transformed_data) 
            with open('transformed_obj_file', 'wb') as file_obj:
                dill.dump(preprocess_obj,file_obj)      
            logging.info("data transformation has been completed")    
        except Exception as e:
            raise FraudException(e,sys)  from e    
            
            
    def model_trainer(self):
        try:
            with open ('transformed_data_file','rb')as file_obj:
                train_array=np.load(file_obj)
                X_train,X_test,Y_train,Y_test=train_test_split(train_array[:,:-1],train_array[:,-1],random_state=0,test_size=0.20)
                classifier=RandomForestClassifier().fit(X_train,Y_train)
                pred=classifier.predict(X_test)
                with open('model', "wb") as file_obj:
                    dill.dump(classifier, file_obj)
            logging.info('model training has been completed')        
        except Exception as e:
            raise FraudException(e,sys) from e           
    def fetch_model():
        with open('model', "rb") as file_obj:
            model=dill.load(file_obj)
        return model  
if __name__=='__main__':
    a=Pipeline() 
    a.data_ingestion()
    a.data_transformation()   
    a.model_trainer()
 