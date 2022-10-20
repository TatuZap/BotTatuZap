from email import message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# tensorflow tags
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed

import json
import pickle

import random
import numpy as np
import warnings
from messageutils import MessageUtils # nossa classe de pré-processamento

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

class TatuIA:
    def __init__(self, dfa_file_path , message_utils ):
        self.dfa_file = dfa_file_path
        self.message_utils = message_utils # classe de pré-processamento de textos
        self.model = self.__simple_ann() # neuranet do bot
        self.__train()
        
        
    def __simple_ann(self):
        
        input_shape = (self.message_utils.X.shape[1],)
        output_shape = self.message_utils.Y.shape[1]
        # the deep learning model
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation = "softmax"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[tf.keras.metrics.Precision()])

        return model
    
    def __train(self):
        self.model.fit(self.message_utils.X, self.message_utils.Y, epochs=200, verbose=0)
    
    def get_model(self):
        return self.model
    
    def print_model(self):
        print(self.get_model().summary())
    
    def eval_model(self):
        print("Evaluate on train data")
        results = self.model.evaluate(self.message_utils.X, self.message_utils.Y, batch_size=1)
        print("test loss, test acc:", results)
    
    def show_data(self):
        """
            cria um dataframe dos dados de treino e os retorna.
        """
        pass 
    
    def __valid_path__(self):
        pass 
    
    def __load_data__(self):
        pass
    
    def get_reply(self, user_message):
        pass 

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
    message_utils = MessageUtils()
    message_utils.process_training_data(database,None)

    tatu_zap = TatuIA("", message_utils=message_utils)
  
    tatu_zap.print_model()

    tatu_zap.eval_model()

if __name__ == "__main__":
    main()