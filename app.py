#use pip install streamlit to develop the app
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
#we need to create pickle files for the deployment

#to open the pickle files
tfidf=pickle.load(open('C:/Users/febin/OneDrive/Desktop/Intership/vectorizer.pkl', 'rb'))
model=pickle.load(open('C:/Users/febin/OneDrive/Desktop/Intership/model.pkl', 'rb'))
st.title("SMS SPAM CLASSIFIER")#THE TITLE OF THE APPLICATION
input = st.text_input("Enter your SMS here")

#NOW WE CREATE A BUTTON TO PREDICT THE RESULT
if st.button("Predict"):
    #for preprocessing the data we use the transform function
    def transform(m):
        m=m.lower()
        n = nltk.word_tokenize(m)
        y = []
        for i in n:
            if i.isalnum():
                y.append(i)
                n=y[:]
                y.clear()
        for i in n:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                n = y[:]
                y.clear()

        for i in n:
            y.append(ps.stem(i))
        n = y[:]

        return n

    sms = transform(input)#we preprocess the data
    #now we need to convert it into a vector
    vector_input = tfidf.transform(sms)

    result = model.predict(vector_input)#this will give the result of the input

    if result==1:
        st.header("SPAM")
    else:
        st.header("HAM")


