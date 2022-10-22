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

    
    message_utils = MessageUtils()
    message_utils.process_training_data(database,None)


    tatu_zap = TatuIA("", message_utils=message_utils)
    #tatu_zap.print_model()
    #tatu_zap.eval_model()
    
    X = ["minhas matéria","QuaL matérias ? ","minha disciplina ", "Que aula tenho ","me fale turma", "que sala ", "Qual minha Sala","qual a turma ?","Oi","Ola","Opa","dia","asdasdasdas","cachorro voador","esquilo de chapeu", "gato de botas"]
    Y = ["my_classes","my_classes","my_classes", "my_classes","my_classes", "my_classes", "my_classes","my_classes","welcome","welcome","welcome","welcome","anything_else","anything_else","anything_else","anything_else"]
    
    list_predict = [(tatu_zap.get_predict(i)) for i in X]
        

    y_true = pd.Series(Y, name='Row_True')
    y_pred = pd.Series(list_predict, name='Col_Pred')
    df_confusion = pd.crosstab(y_pred, y_true)
    #df_confusion = pd.crosstab(y_pred, y_true,margins = 'True')

    print(df_confusion)
    print(classification_report(Y, list_predict))

if __name__ == "__main__":
    main()