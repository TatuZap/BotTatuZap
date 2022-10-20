import os
import json
import pickle
from xml.dom.expatbuilder import InternalSubsetExtractor
from matplotlib.pyplot import get
import nltk
import random
import pandas as pd
import numpy as np
import warnings
import re
import string
import unidecode 
from dataclasses import dataclass
from typing import Callable

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize , download , pos_tag
from nltk.stem import WordNetLemmatizer

# deixe a linha abixo sem comentários somente se precisar dessas bibliotecas de nlp
#download(['punkt','averaged_perceptron_tagger','stopwords','wordnet','omw-1.4'\])
stopwords = nltk.corpus.stopwords.words('portuguese')
warnings.filterwarnings('ignore')

@dataclass
class MessageUtils:    
    """
        Classe que concentra todo o processamento necessário para
        textos de forma numérica.
    """
    url_pattern = 'http\S+|www\S+|https\S+'
    punctuacion_pattern = r'[^\w\s]'
    multiple_backspace_pattern = r'\s+'
    emote_pattern = re.compile("["
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
            "]+",re.UNICODE)
    
    
    clean_by_sub  = lambda self, message, pattern, sub_string : re.sub(pattern,sub_string,message)
    
    clean_emotes = lambda self, message, : self.clean_by_sub(message, self.emote_pattern, '')
    clean_urls   = lambda self, message, : self.clean_by_sub(message, self.url_pattern, '')
    clean_punctuation = lambda self, message :  self.clean_by_sub(message, self.punctuacion_pattern, '')
    clean_multiple_backspaces = lambda self, message : self.clean_by_sub(message, self.multiple_backspace_pattern, ' ')
    
    def full_clean_text(self,message): 
        message = self.clean_emotes(message.lower())
        message = self.clean_urls(message)
        message = self.clean_punctuation(message)
        message = self.clean_multiple_backspaces(message)
        message = unidecode.unidecode(message)
        return message

    def get_tokens(self,message, tokenizer = None, stopwords=nltk.corpus.stopwords.words('portuguese')):
        return [ word for word in nltk.word_tokenize(self.full_clean_text(message)) if word not in stopwords]


    def bag_of_words_corpus_rep(self):
        """
            Função que devolve a representação bag of words para
            um determinado corpus de entrada.
        """
        bag_X_and_Y = []
        output_empty = [0] * len(self.classes)
        for document in self.documents:
            bag = []
            pattern_words = document[0]
            for word in self.vocabulary:
                bag.append(1) if word in pattern_words else bag.append(0)

            # output_row atuará como uma chave para a lista, 
            # onde a saida será 0 para cada tag e 1 para a tag atual
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1

            bag_X_and_Y.append([bag, output_row])

        random.shuffle(bag_X_and_Y)
        bag_X_and_Y = np.array(bag_X_and_Y)
        # criamos lista de treino sendo x os patterns e y as intenções
        self.X = np.array(list(bag_X_and_Y[:, 0]))
        self.Y = np.array(list(bag_X_and_Y[:, 1]))

    

    def process_training_data(self, corpus, stopwords=nltk.corpus.stopwords.words('portuguese')):
        self.corpus = corpus
        self.vocabulary = []
        self.documents  = []
        intents    = corpus

        self.classes = [ intent['tag'] for intent in intents['intents']]

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                word = self.get_tokens(pattern)
                self.vocabulary.extend(word)
                self.documents.append((word, intent['tag']))

        self.vocabulary = sorted(list(set(self.vocabulary)))
        self.classes = sorted(list(set(self.classes)))

        pickle.dump(self.vocabulary, open('train_vocabulary.pkl', 'wb'))
        pickle.dump(self.classes, open('train_classes.pkl', 'wb'))
        self.bag_of_words_corpus_rep()


    def bag_for_message(self, message):
        sentence_words = self.get_tokens(message)

        bag = [0]*len(self.vocabulary)
        for setence in sentence_words:
            for i, word in enumerate(self.vocabulary):
                if word == setence:
                    bag[i] = 1
        return(np.array(bag))

    def show_data(self, documents):
        return pd.DataFrame(documents, columns = ['Message_tokens', 'Intent'])
    
    def __valid_path__(self):
        return os.path.exists(self.dfa_file)
    
    def __load_data__(self):
        if self.__valid_path__(self):
            return json.load(self.dfa_file)
   
    def is_ra(message):
        ra = re.findall('\d+', message)

        if ra != []:
            if len(ra) > 1:
                return False
            elif len(ra[0]) != 11 and len(ra[0]) != 9:
                return False
            return ra[0]
        return False

def main():
    database = {
        "intents": [
                {
                    "tag": "welcome",
                    "patterns": ["Oi","Oi, bom dia","Oi, boa tarde", "bom dia", "boa tarde", "boa noite", "oi, boa noite", "olá, boa noite", "oiiiii", "Olá","oiii, como vai?","opa, tudo bem?"],
                    "responses": ["Olá, serei seu assistente virtual, em que posso te ajudar?","Salve, qual foi ?", "Manda pro pai, Lança a braba", "No que posso te ajudar ?"],
                    "context": [""]
                },
                {
                    "tag": "my_classes",
                    "patterns": ["Quais são as minhas matérias ?","Quais são as minhas matérias de hoje ? ","Quais são as minhas disciplinas de hoje ? ", "Que aulas eu tenho Hoje","me fale minhas turmas", "que sala eu devo ir?", "Qual minha Sala ?","quais as minhas turmas ?"],
                    "responses": ["Entendi, você deseja saber suas salas","Você deseja saber suas salas ?", "Ah, você quer saber qual sala ? ", "Suas Aulas ?"],
                    "context": [""]
                },
                {
                    "tag": "anything_else",
                    "patterns": [],
                    "responses": ["Desculpa, não entendi o que você falou, tente novamente!","Não compreendi a sua solicitação, talvez eu possa te ajudar"],
                    "context": [""]
                }
            ]
        }
    # demo da funcionalide da classe utils para mensagem
    text_utils = MessageUtils()
    print("Exemplo de pre-procesamento ",text_utils.get_tokens("200 comentários com o emoji do #TimeDoBigode, bora? 👨🏻👨🏻👨🏻"))
    text_utils.process_training_data(database,None)
    print("Exemplo de Bag of word ",text_utils.bag_for_message("oi, como vai você, quais são as minhas matérias de hoje"))
    print(text_utils.vocabulary)
if __name__ == "__main__":
    main()