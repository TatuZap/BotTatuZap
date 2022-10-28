from tatuia import TatuIA 
from messageutils import MessageUtils # nossa classe de pré-processamento
from sklearn.metrics import classification_report

#import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
                    "patterns": ['me fale as materia', 'minhas disciplina', ' turmas', ' local', 'diz a horario', 'quero saber as disciplina', 'minhas turma', ' aulas', ' salas', ' classes', ' materias', 'diga a  salas', 'me fale as professores', 'quais as minhas disciplina', 'minhas turmas', 'quero minhas professor', 'quero saber as classes', 'qual é turmas'],
                    "responses": ["Entendi, você deseja saber suas salas","Você deseja saber suas salas ?", "Ah, você quer saber qual sala ? ", "Suas Aulas ?"],
                    "context": [""]
                },
                {
                    "tag": "bus_info",
                    "patterns": ['horário do fretado','hora dos fretados','qual hora do onibus','quando sai o busao','minha lotação'],
                    "responses": ["Fretados","Horarios Fretado"], #provisório
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

    
    message_utils = MessageUtils()
    message_utils.process_training_data(database,None)


    tatu_zap = TatuIA("", message_utils=message_utils,lstm = True)
    #tatu_zap.print_model()
    #tatu_zap.eval_model()
    X = ['informe a turmas', 'quero saber as sala', 'quais as minhas aulas', ' classes', ' professor', ' local', 'quais as minhas disciplina', 'quero saber as professores', 'informe a disciplina', 'quero saber as grade', ' aulas', 'qual é aulas', 'qual minha classes', 'quais as minhas materias', 'diz a materia', 'quero minhas disciplinas', 'minhas classe', 'minhas disciplinas', 'qual é materias', 'informe a materia', 'Oi', 'Oi, bom dia', 'Oi, boa tarde', 'bom dia', 'boa tarde', 'boa noite', 'oi, boa noite', 'olá, boa noite', 'oiiiii', 'Olá', 'oiii, como vai?', 'opa, tudo bem?', 'quando sai onibus', ' lotação', 'qual busao', 'quero saber fretado', 'qual fretado', 'informe fretado', 'quero que sai fretado', 'informe lotação', 'vai sair busao', 'quero saber fretados', ' busao', 'quando onibus', 'que hora lotação', 'que hora fretado',"asdasdasdas","cachorro voador","esquilo de chapeu", "gato de botas"]
    Y = ['my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'my_classes', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info', 'bus_info','anything_else','anything_else','anything_else','anything_else']

    
    list_predict = [(tatu_zap.get_predict(i)) for i in X]
        

    y_true = pd.Series(Y, name='Row_True')
    y_pred = pd.Series(list_predict, name='Col_Pred')
    df_confusion = pd.crosstab(y_true, y_pred)
    #df_confusion = pd.crosstab(y_pred, y_true,margins = 'True')

    print(df_confusion)
    print(classification_report(Y, list_predict))

if __name__ == "__main__":
    main()