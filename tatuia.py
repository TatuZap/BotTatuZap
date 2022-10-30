from email import message
from tensorflow.keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout, SpatialDropout1D, LSTM, Embedding
from keras.models import Sequential
from keras import metrics
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

        #model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=[tf.keras.metrics.AUC()])
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
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
        # self.Y = np.array(tokenizer.texts_to_sequences(df['classe']))
        # print('self.X')
        # print(self.X)
        # print('self.Y')
        # print(self.Y)
        

        model = Sequential()
        model.add(Embedding(MAX_LEN, 32, input_length=self.X.shape[1]))
        model.add(LSTM(16))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.Y.shape[1], activation='softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.metrics.Recall()])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model

    def __train(self):
        self.model.fit(self.X,
                       self.Y, epochs=13, batch_size=1,verbose=1)
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
        # print(">>> Normalized and Clean user_message: {}.".format(
        #     self.message_utils.full_clean_text(user_message)))
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
        return self.__intent_prediction(user_message)[0]['intent']



def main():
    database = {
        "intents": [
                {
                    "tag": "welcome",
                    "patterns": ['oi','ola','boa tarde','bom dia','boa noite','saudações','fala','eae','salve','fala','fala meu bom','grande bot'],
                    "responses": ["Olá, serei seu assistente virtual, em que posso te ajudar?","Salve, qual foi ?", "Manda pro pai, Lança a braba", "No que posso te ajudar ?"],
                    "context": [""]
                },
                {
                    "tag": "my_classes",
                    "patterns": ['minhas grade na quarta',
  'informe a professor de manha',
  'quais as minhas turmas de tarde',
  'diga a  aula de tarde',
  'quero minhas sala de noite',
  ' professores ',
  'diz a turma na sexta',
  'diga a  turmas de tarde',
  'quero saber as professor agora',
  'qual minha disciplinas ',
  'quero saber as aula na sexta',
  'diz a local na sexta',
  'minhas turmas ',
  'minhas local de manha',
  'quero saber as horario de noite',
  'minhas disciplinas na segunda',
  'me fale as disciplina de tarde',
  'informe a grade na segunda',
  'qual é local na terca',
  'quero minhas salas na quarta',
  'quero saber as horario na segunda',
  'quero minhas disciplinas na quarta',
  'diga a  materia na segunda',
  'quais as minhas turmas de noite',
  'quero minhas professores de tarde',
  'qual minha sala de tarde',
  ' horario de tarde',
  'qual minha aula na quinta',
  'diga a  classes ',
  'me fale as turma de noite',
  'quero saber as professor na terca',
  ' turma ',
  'qual é aula na terca',
  'minhas turma ',
  'qual é professor na sexta',
  'quais as minhas professores na quinta',
  ' horario de noite',
  'qual minha local ',
  'qual é sala ',
  'quais as minhas professores ',
  'quero saber as materias ',
  'diga a  materias que devo ir',
  'me fale as materias na quarta',
  'quero minhas aulas de noite',
  'quero saber as turmas agora',
  'qual é aulas de noite',
  'diz a disciplinas ',
  'informe a classes que devo ir',
  'qual é salas de manha',
  'informe a local '],
                    "responses": ["Entendi, você deseja saber suas salas","Você deseja saber suas salas ?", "Ah, você quer saber qual sala ? ", "Suas Aulas ?"],
                    "context": [""]
                },
                {
                    "tag": "bus_info",
                    "patterns": ['quando busao',
                                'quando busao',
                                ' onibus',
                                'quero saber lotação',
                                'quando sai onibus',
                                'vai sair fretados',
                                'qual fretados',
                                'quero saber fretado',
                                ' lotação',
                                ' onibus',
                                'qual onibus',
                                'quando busao',
                                'informe lotação',
                                'que hora lotação',
                                ' lotação',
                                'quando onibus',
                                ' lotação',
                                'informe fretado',
                                'quando fretado',
                                ' lotação',
                                ' fretados',
                                'qual fretados',
                                'qual onibus',
                                'quero que sai busao',
                                'que hora fretado',
                                ' fretados',
                                'quero saber fretados',
                                ' onibus',
                                ' fretados',
                                'informe lotação',
                                'quando sai lotação',
                                'que hora fretados',
                                'informe fretados',
                                ' lotação',
                                'quando onibus',
                                'que hora fretados',
                                'quando sai lotação',
                                ' fretados',
                                ' fretado',
                                'quando sai onibus',
                                'vai sair busao',
                                'quero que sai busao',
                                'quando busao',
                                ' onibus',
                                'que hora onibus',
                                'vai sair fretados',
                                'informe fretado',
                                ' lotação',
                                'que hora fretado',
                                'vai sair fretados'],
                    "responses": ["Fretados","Horarios Fretado"], #provisório
                    "context": [""]
                },
                {
                    "tag": "disc_info",
                    "patterns": ['Gostaria de saber da ementa da disciplina','ementa da materia','quero saber da ementa','quero saber do plano de ensino','quais os requsitos da materia','qual a bibliografia da disciplina'],
                    "responses": ['Informações da disciplina X','Para a disciplina Y, as informações são as seguintes'],
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

