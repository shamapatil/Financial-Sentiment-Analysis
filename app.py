# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:40:45 2023

@author: Prayag V K
"""

import streamlit as st

import nltk
import re
import pickle
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

pickle_out=open('modell.pkl','rb')
model=pickle.load(pickle_out)

handle=open('tfidf.pkl','rb')
tokenizer=pickle.load(handle)

st.title('Financial Sentiment Analysis!')

text = st.text_area("TYPE YOUR FEEDBACK HERE", height=80)

if st.button("Predict"):
    text=text.lower()
    
    # Removing the punctuation
    text=re.sub(r'[^\w\s]','',text)
    
    pattern = re.compile("¬")
    text=pattern.sub("",text)
    
    # Removing numbers in the text.
    text=re.sub(r'\d+','',text)
    
    # Deleting the new lines.
    text=re.sub('\n','',text)
    
    # Removing non-english alphabets
    text =''.join ([word for word in text if word.isalpha() or word.isspace()])
    
    # Creating the tokens
    tokens=nltk.word_tokenize(text)
    
    # Removing the stopwordmand lemmatizing them
    lemma=WordNetLemmatizer()
    
    pattern = re.compile("¬")
    text=pattern.sub("",text)

    # Making tokens.
    tokens=[lemma.lemmatize(word) for word in tokens if not word in stopwords.words('english')]
    
    # Joing the clean text and returning
    text=' '.join(tokens)
    
    
    pickle_out=open(r'modell.pkl','rb')
    model=pickle.load(pickle_out)
    pickle_out=open(r'tfidf.pkl','rb')
    tokenizer=pickle.load(pickle_out)
    
    transformed_output=tokenizer.transform([text])
    output=model.predict(transformed_output)[0]
    
    if output==1:
        st.write("Positive")
    #elif output==0:
        #st.write("Neutral")
    else:
        st.write("Negative")