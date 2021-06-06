import numpy as np
import pandas as pd
import re   
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

dataset = pd.read_csv(r"Restaurant_Reviews.tsv",delimiter = "\t")

data = []
for i in range(0,1000):
    review = dataset["Review"][i]
    review = re.sub('[^a-zA-Z]',' ',review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)

x = np.array(data)
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=40)


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import  accuracy_score,confusion_matrix,classification_report
accuracy_score(y_pred,y_test)

text = 'Not tasty and the texture was just nasty'
ypred = model.predict(vect.transform([text]))

import joblib
joblib.dump(model,'sentiment')

import joblib
joblib.dump(vect,'vectt')

import streamlit as st
import joblib

model = joblib.load('sentiment')
vect1 = joblib.load('vectt')

st.title('Sentimental Analysis')
ip = st.text_input("Enter the message")
op = model.predict(vect.transform([ip]))
if st.button('Predict'):
  if op[0]==1:
    st.title('Positive Review')
  else:
    st.title('Negative Review')
