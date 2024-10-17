import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import nltk

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import string

from nltk.corpus import stopwords
stopwords.words('english')

def transform(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title('SMS spam Classifier')

input_sms=st.text_area('Enter the Message')

if st.button('Predict'):

    transformed_sms=transform(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]

    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')