import os
import json
import pickle
import nltk
import random
import numpy as np
import warnings
import re
import string
import unidecode 

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize , download , pos_tag
from nltk.stem import WordNetLemmatizer
download(['punkt','averaged_perceptron_tagger','stopwords','wordnet','omw-1.4'])


class TatuIA:
    def __init__(self, intent_file_path, dfa_file_path ):
        self.intent_file = intent_file_path
        self.dfa_file = dfa_file_path
        
    def __train__(self):
        pass 
    
    def get_model(self):
        pass
    
    def eval_model(self):
        pass
    
    def __process_text__(self,message=None):
        """
            Função de concentra todo o Pré-processamento do Texto
        """
        def clean_emotes(message):
          emoj = re.compile("["
                u"\U0001F600-\U0001F64F"  
                u"\U0001F300-\U0001F5FF"  
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF"  
                u"\U00002500-\U00002BEF"  
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  
                u"\u3030"
                "]+",
                              re.UNICODE)
          return re.sub(emoj, '', message)

        def clean_urls(message): 
          return re.sub('http\S+|www\S+|https\S+', '', message)

        def clean_punctuation(message) : 
          return re.sub(r'[^\w\s]','',message)

        def clean_text(message): 
          message = message.lower()
          message = unidecode.unidecode(message) # remove accentuation
          message = clean_emotes(message)  
          message = clean_punctuation(message)
          message = clean_urls(message)
          message = re.sub(r'\s+',' ',message)
          return message


        def get_tokens(message, tokenizer = None,stopwords=None):
            return nltk.word_tokenize(message)

    
    def show_data(self):
        pass 
    
    def __valid_path__(self):
        pass 
    
    def __load_data__(self):
        self.dfa =  
    
    def get_reply(self, message):
        pass 