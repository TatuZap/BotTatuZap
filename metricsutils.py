import tatuia as tatu
from messageutils import MessageUtils # nossa classe de pré-processamento


import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
def confusion_matrix(p,y_test):
    #p = model.model_predict(X_test)
    #p = np.rint(p)
    #p = p.astype(int)

    #pred_translated = np.argmax(p, axis=1)
    #y_translated = np.argmax(y_test, axis=1)

    cm = multilabel_confusion_matrix(p, y_test)
    #df_cm = pd.DataFrame(cm[0], index = ['welcome', 'my_classes'],columns = ['welcome', 'my_classes'])
    #plt.figure(figsize = (10,7))
    #sn.heatmap(df_cm, annot=True)
    print(cm[0])



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

    database_test = { 
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

    message_test = MessageUtils()
    message_test.process_training_data(database_test,None)

    tatu_zap = tatu.TatuIA("", message_utils=message_utils)
  
    tatu_zap.print_model()

    tatu_zap.eval_model()
    #print ("Salve")
    X = ["minhas matéria","QuaL matérias ? ","minha disciplina ", "Que aula tenho ","me fale turma", "que sala ", "Qual minha Sala","qual a turma ?","Oi","Ola","Opa","dia"]
    Y = ["my_classes","my_classes","my_classes", "my_classes","my_classes", "my_classes", "my_classes","my_classes","welcome","welcome","welcome","welcome"]
    


    #print(tatu_zap.get_reply("Quais são as minhas matérias ?"))

    list = []
    for i in X:
        list.append(tatu_zap.model_predict(i))


    #print(list)
    confusion_matrix(list,Y)
    

if __name__ == "__main__":
    main()