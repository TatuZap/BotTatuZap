import os
from pyexpat import model
from unittest import result
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
    def __init__(self, dfa_file_path , message_utils: MessageUtils ):
        self.dfa_file = dfa_file_path
        self.message_utils = message_utils # classe de pré-processamento de textos
        self.model = self.__simple_ann() # neuranet do bot
        self.__load_model()
        self.PROB_SAFE_VALUE = 0.25
    

    def __load_model(self):
        current_filepath = os.getcwd()
        model_folder = "model_bot"
        complete_path = os.path.join(current_filepath, model_folder)
        if os.path.exists(complete_path):
            self.model = tf.keras.models.load_model(complete_path + "/model")
        else:
            self.model = self.__simple_ann()
            self.__train()
            os.mkdir(complete_path)
            self.model.save(complete_path + "/model")

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
        return os.path.exists(self.dfa_file)
    
    def __load_data__(self):
        if self.__valid_path__(self):
            return json.load(self.dfa_file)
    
    def __intent_prediction(self,user_message):
        user_message_bag = self.message_utils.bag_for_message(user_message)
    
        response_prediction = self.model.predict(np.array([user_message_bag]))[0]
        
        print(response_prediction)

        results = [[index, response] for index, response in enumerate(response_prediction) if response > self.PROB_SAFE_VALUE ]    
        #print(results)
        # verifica nas previsões se não há 1 na lista, se não há envia a resposta padrão (anything_else) 
        # ou se não corresponde a margem de erro

        if "1" not in str(user_message_bag) or len(results) == 0 :
            results = [[0, response_prediction[0]]]

        results.sort(key=lambda x: x[1], reverse=True)
        print([{"intent": self.message_utils.classes[r[0]], "probability": str(r[1])} for r in results])
        return [{"intent": self.message_utils.classes[r[0]], "probability": str(r[1])} for r in results]


    def get_reply(self,user_message):
        most_prob_intent = self.__intent_prediction(user_message)[0]['intent'] # a classe mais provável
        list_of_intents = self.message_utils.corpus['intents'] # lista de intenções

        for idx in list_of_intents:
            if idx['tag'] == most_prob_intent:
                result = random.choice(idx['responses'])
                break

        return result 

    def is_ra(message):
        ra = re.findall('\d+', message)

        if ra != []:
            # testa se o user digitou mais de um numero
            if len(ra) > 1:
                return False
            # descarta ra com tamanho diferente de 11 e 9
            elif len(ra[0]) != 11 and len(ra[0]) != 9:
                return False
            return True
        # retorna false caso nao encontre um ra
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
    message_utils = MessageUtils()
    message_utils.process_training_data(database,None)

    tatu_zap = TatuIA("", message_utils=message_utils)
  
    #tatu_zap.print_model()

    #tatu_zap.eval_model()

    while True:
        try:
            print("Manda uma mensagem para o TatuBot !")
            tatu_zap.get_reply(input())
        except EOFError:
            break

if __name__ == "__main__":
    main()