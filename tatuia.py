from email import message
from tensorflow.keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout, SpatialDropout1D, LSTM, Embedding
from keras.models import Sequential
import tensorflow as tf
from messageutils import MessageUtils  # nossa classe de pré-processamento
import warnings
import numpy as np
import pandas as pd
import random
import pickle
import json
import os
from unittest import result
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tensorflow tags
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


# from keras.preprocessing.sequence import pad_sequences


class TatuIA:
    def __init__(self, dfa_file_path, message_utils: MessageUtils, lstm = True):
        self.dfa_file = dfa_file_path
        self.message_utils = message_utils  # classe de pré-processamento de textos
        self.model = self.__simple_ann() if not lstm else self.__simple_lstm() # neuranet do bot
        self.__train()
        self.PROB_SAFE_VALUE = 0.25

    def __load_model(self):
        # current_filepath = os.getcwd()
        # model_folder = "modelbot"
        # complete_path = os.path.join(current_filepath, model_folder)
        # if os.path.exists(complete_path):
        #     print(">>> Carregando o TatuBot do Disco")
        #     self.model = tf.keras.models.load_model(complete_path + "/model")
        #     print(">>> Fim do Carregando o TatuBot do Disco")
        # else:
        print(">>> Build do TatuBot")
        if self.model:
            self.__train()
        self.model = self.__simple_ann()
        self.__train()
        #os.mkdir(complete_path)
        #self.model.save(complete_path + "/model")
        print(">>> Fim do Build, TatuBot dumped")

    def __simple_ann(self):
        self.X = self.message_utils.X
        self.Y = self.message_utils.Y

        input_shape = (self.message_utils.X.shape[1],)
        output_shape = self.message_utils.Y.shape[1]
        # the deep learning model
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation="softmax"))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[tf.keras.metrics.Precision()])
        return model

    def __simple_lstm(self):
        """
            pré-processamento especial para a LSTM FIXME: realocar esse código.
        """
        df = pd.DataFrame(self.message_utils.documents,columns = ["token-frase","classe"])
        df["texto-lstm"] = df["token-frase"].apply( lambda message : " ".join(message))
        df = df.drop("token-frase",axis="columns")
        MAX_LEN   = len(self.message_utils.vocabulary)
        tokenizer = Tokenizer(MAX_LEN,lower=True)
        tokenizer.fit_on_texts(df['texto-lstm'].values)
        X = tokenizer.texts_to_sequences(df['texto-lstm'].values)
        self.X = pad_sequences(X, maxlen=MAX_LEN)
        self.Y = pd.get_dummies(df['classe']).values

        model = Sequential()
        model.add(Embedding(MAX_LEN, 100, input_length=self.X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(self.Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def __train(self):
        self.model.fit(self.X,
                       self.Y, epochs=200, verbose=0)
    def get_model(self):
        return self.model

    def print_model(self):
        print(self.get_model().summary())

    def eval_model(self):
        print("Evaluate on train data")
        results = self.model.evaluate(
            self.X, self.Y, batch_size=1)
        print("test loss, test acc:", results)

    def __intent_prediction(self, user_message):
        print(">>> Normalized and Clean user_message: {}.".format(
            self.message_utils.full_clean_text(user_message)))
        user_message_bag = self.message_utils.bag_for_message(user_message)

        response_prediction = self.model.predict(
            np.array([user_message_bag]), verbose=0)[0]

        # print(response_prediction)

        results = [[index, response] for index, response in enumerate(
            response_prediction) if response > self.PROB_SAFE_VALUE]
        # print(results)
        # verifica nas previsões se não há 1 na lista, se não há envia a resposta padrão (anything_else)
        # ou se não corresponde a margem de erro

        if "1" not in str(user_message_bag) or len(results) == 0:
            results = [[0, response_prediction[0]]]

        results.sort(key=lambda x: x[1], reverse=True)
        # print([{"intent": self.message_utils.classes[r[0]], "probability": str(r[1])} for r in results])
        return [{"intent": self.message_utils.classes[r[0]], "probability": str(r[1])} for r in results]

    def get_reply(self, user_message):
        most_prob_intent = self.__intent_prediction(
            user_message)[0]['intent']  # a classe mais provável
        # lista de intenções
        list_of_intents = self.message_utils.corpus['intents']

        for idx in list_of_intents:
            if idx['tag'] == most_prob_intent:
                result = random.choice(idx['responses'])
                break

        return result, most_prob_intent

    def get_predict(self, user_message):
        # a classe mais provável
        return self.__intent_prediction(user_message)[0]['intents']



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
                    "patterns": ["oi, Quais são as minhas matérias ?","Quais são as minhas matérias ?","olá Quais são as minhas matérias de hoje ? ","Bom dia, Quais são as minhas disciplinas de hoje ? ", "Que aulas eu tenho Hoje","oi, ola, bom dia me fale minhas turmas", "que sala eu devo ir?", "Qual minha Sala ?","quais as minhas turmas ?"],
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

    tatu_zap = TatuIA("", message_utils=message_utils,lstm=False)
  
    #tatu_zap.print_model()

    #tatu_zap.eval_model()
    
    print(">>> Demo da funcionalidade de reconhecimento de intenção do TatuBot.")
    print(">>> Inicialmente a I.A foi treinada com somente duas inteções (welcome,my_classes).")

    while True:
        try:
            #print(">>> Envie uma mensagem para o TatuBot!")
            user_message = input("user: ")
            response, intent = tatu_zap.get_reply(user_message)
            if intent == "my_classes":
                user_ra = tatu_zap.message_utils.is_ra(user_message)
                if user_ra:
                    print("Tatu: Já estou processando as turmas para o ra {}.".format(user_ra))
                else:
                    while True:
                        print("Tatu: Você solicitou informações sobre suas turmas, agora insira seu ra!.")
                        expected_ra = input()
                        user_ra = tatu_zap.message_utils.is_ra(expected_ra)
                        if user_ra:
                            print("Tatu: Já estou processando as turmas para o ra {}.".format(user_ra))
                            break
            else:
                print("Tatu: {}.".format(response)) 

        except KeyboardInterrupt:
            break



if __name__ == "__main__":
    main()

