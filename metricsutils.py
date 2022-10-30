from tatuia import TatuIA 
from messageutils import MessageUtils # nossa classe de pré-processamento
from sklearn.metrics import classification_report

#import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geradorfrases as gerador


def main():
    # database = {
    #     "intents": [
    #             {
    #                 "tag": "welcome",
    #                 "patterns": ['oi','ola','boa tarde','bom dia','boa noite','saudações','fala','eae','salve'],
    #                 "responses": ["Olá, serei seu assistente virtual, em que posso te ajudar?","Salve, qual foi ?", "Manda pro pai, Lança a braba", "No que posso te ajudar ?"],
    #                 "context": [""]
    #             },
    #             {
    #                 "tag": "my_classes",
    #                 "patterns": ['materias', 'materia', 'sala', 'disciplina','professor','local','turma','turmas','professores','disciplinas','salas','aula','aulas','grade','horario','classe','classes','cadeira','cadeiras','sala de aula','local de estudo','disciplinas matriculadas'],
    #                 "responses": ["Entendi, você deseja saber suas salas","Você deseja saber suas salas ?", "Ah, você quer saber qual sala ? ", "Suas Aulas ?"],
    #                 "context": [""]
    #             },
    #             {
    #                 "tag": "bus_info",
    #                 "patterns": ['fretado','fretados','onibus','busao','lotação','coletivo','circular','transporte','carro','veículo'],
    #                 "responses": ["Fretados","Horarios Fretado"], #provisório
    #                 "context": [""]
    #             },
    #             {
    #                 "tag": "anything_else",
    #                 "patterns": [],
    #                 "responses": ["Desculpa, não entendi o que você falou, tente novamente!","Não compreendi a sua solicitação, talvez eu possa te ajudar"],
    #                 "context": [""]
    #             }
    #         ]
    #     }

    database = {
        "intents": [
                {
                    "tag": "welcome",
                    "patterns": [],
                    "responses": ["Olá, serei seu assistente virtual, em que posso te ajudar?","Salve, qual foi ?", "Manda pro pai, Lança a braba", "No que posso te ajudar ?"],
                    "context": [""]
                },
                {
                    "tag": "myclasses",
                    "patterns": [],
                    "responses": ["Entendi, você deseja saber suas salas","Você deseja saber suas salas ?", "Ah, você quer saber qual sala ? ", "Suas Aulas ?"],
                    "context": [""]
                },
                {
                    "tag": "businfo",
                    "patterns": [],
                    "responses": ["Fretados","Horarios Fretado"], #provisório
                    "context": [""]
                },
                # {
                #     "tag": "disc_info",
                #     "patterns": ['Gostaria de saber da ementa da disciplina','ementa da materia','quero saber da ementa','quero saber do plano de ensino','quais os requsitos da materia','qual a bibliografia da disciplina'],
                #     "responses": ['Informações da disciplina X','Para a disciplina Y, as informações são as seguintes'],
                #     "context": [""]
                # },
                {
                    "tag": "anything_else",
                    "patterns": [],
                    "responses": ["Desculpa, não entendi o que você falou, tente novamente!","Não compreendi a sua solicitação, talvez eu possa te ajudar"],
                    "context": [""]
                }
            ]
        }

    database = gerador.fill_database(database,300)
    #print(database)
    message_utils = MessageUtils()
    message_utils.process_training_data(database,None)


    tatu_zap = TatuIA("", message_utils=message_utils,lstm = False)
    tatu_zap.print_model()
    #tatu_zap.eval_model()
    #X = ['informe a turmas', 'quero saber as sala', 'quais as minhas aulas', ' classes', ' professor', ' local', 'quais as minhas disciplina', 'quero saber as professores', 'informe a disciplina', 'quero saber as grade', ' aulas', 'qual é aulas', 'qual minha classes', 'quais as minhas materias', 'diz a materia', 'quero minhas disciplinas', 'minhas classe', 'minhas disciplinas', 'qual é materias', 'informe a materia', 'Oi', 'Oi, bom dia', 'Oi, boa tarde', 'bom dia', 'boa tarde', 'boa noite', 'oi, boa noite', 'olá, boa noite', 'oiiiii', 'Olá', 'oiii, como vai?', 'opa, tudo bem?', 'quando sai onibus', ' lotação', 'qual busao', 'quero saber fretado', 'qual fretado', 'informe fretado', 'quero que sai fretado', 'informe lotação', 'vai sair busao', 'quero saber fretados', ' busao', 'quando onibus', 'que hora lotação', 'que hora fretado',"asdasdasdas","cachorro voador","esquilo de chapeu", "gato de botas"]
    #Y = ['myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'myclasses', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'welcome', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo', 'businfo','anything_else','anything_else','anything_else','anything_else']
    n = 200
    db_test = gerador.fill_treino(database,n)    
    X = db_test['intents'][0]['patterns'][0:n]+db_test['intents'][1]['patterns'][0:n]+db_test['intents'][2]['patterns'][0:n]+gerador.gerar_anything(n)
    Y = ['welcome']*n+['myclasses']*n+['businfo']*n+['anything_else']*n

    
    list_predict = [(tatu_zap.get_predict(i)) for i in X]
        

    y_true = pd.Series(Y, name='Row_True')
    y_pred = pd.Series(list_predict, name='Col_Pred')
    df_confusion = pd.crosstab(y_true, y_pred)
    #df_confusion = pd.crosstab(y_pred, y_true,margins = 'True')
    print('matriz confusao::')
    print(df_confusion)
    print(classification_report(Y, list_predict))

if __name__ == "__main__":
    main()
