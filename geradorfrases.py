import random as random
#from random import choice


def fill_database(database, n):
    random.seed()
    for intent in database['intents']:
        if intent["tag"] == 'myclasses':
            # sinonimo de QUERO
            list1 = ['quero', 'desejo', 'pretendo', 'cogito', 'exijo', 'necessito', 'preciso', 'procuro', 'interesso-me','','informe','diga','qual é']

            # sinonimos de SABER
            list2 = ['saber', 'entender', 'que me informe', 'conhecer', 'compreender','','','']

            # variacoes para QUE HORAS
            list3 = ['minhas salas', 'minhas turmas', 'minha grade', 'minhas salas de aula', 'minhas classes', 'minha próxima aula', 'a próxima aula',
                    'a turma seguinte', 'a seguinte sala', 'as salas', 'a sala','as sala','a salas','os professores','meus horários']

            # list1  = ['minhas', 'quero minhas', 'quais as minhas', 'qual minha', 'diga a ','informe a','quero saber as', 'me fale as','qual é', 'diz a','','','']
            # list2   = ['materias', 'materia', 'sala', 'disciplina','professor','local','turma','turmas','professores','disciplinas','salas','aula','aulas','grade','horario','classe','classes']
            # list3 = ['','','','agora','que devo ir','na segunda','na terca','na quarta','na quinta','na sexta','de manha','de tarde','de noite']

            for i in range(n):
                a = random.choice(list1)+' '+random.choice(list2)+' '+random.choice(list3)
                if a not in intent['patterns']:
                    intent['patterns'].append (a)
                else: i = i-1

        if intent["tag"] == 'businfo':
            # sinonimo de QUERO
            list_bus1 = ['quero', 'desejo', 'pretendo', 'necessito', 'preciso', 'procuro']
            # sinonimos de SABER
            list_bus2 = ['saber', 'entender', 'me informar', 'conhecer']
            # variacoes para QUE HORAS
            list_bus3 = ['que horas', 'que momento', 'quando']
            # sinonimos de PARTIR
            list_bus4 = ['','sai', 'parte']
            # sinonimos de onibus
            list_bus5 = ['o onibus', 'o busao', 'o fretado', 'o transporte', 'a lotação']

            #list_bus1 = ['','','','quero saber','informe','qual','que hora','quando','quando sai','quero que sai','vai sair']
            #list_bus2 = ['fretado','fretados','onibus','busao','lotação']
            
            for i in range(n):
                a = random.choice(list_bus1)+' '+random.choice(list_bus2)+' '+random.choice(list_bus3)+' '+random.choice(list_bus4)+' '+random.choice(list_bus5)
                #a = random.choice(list_bus1)+' '+random.choice(list_bus2)
                if a not in intent['patterns']:
                    intent['patterns'].append (a)
                else: i = i-1
        if intent["tag"] == 'welcome':
            list_wel1 = ['oi','ola','salve','eae']
            list_wel2 = ['bom dia','boa tarde','boa noite']
            list_wel3 = ['','','bot','tatu','tatuzap','']

            for i in range(n):
                a = random.choice(list_wel1)+' '+random.choice(list_wel2)+' '+random.choice(list_wel3)
                intent['patterns'].append (a)
                
    return database

def fill_treino(database, n):
    random.seed()
    for intent in database['intents']:
        if intent["tag"] == 'myclasses':
            # # sinonimo de QUERO
            # list1 = ['quero', 'desejo', 'pretendo', 'cogito', 'exijo', 'necessito', 'preciso', 'procuro', 'interesso-me','','informe','diga','qual é']

            # # sinonimos de SABER
            # list2 = ['saber', 'entender', 'que me informe', 'conhecer', 'compreender','','','']

            # # variacoes para QUE HORAS
            # list3 = ['minhas salas', 'minhas turmas', 'minha grade', 'minhas salas de aula', 'minhas classes', 'minha próxima aula', 'a próxima aula',
            #         'a turma seguinte', 'a seguinte sala', 'as salas', 'a sala','as sala','a salas','os professores','meus horários']

            list1  = ['minhas', 'quero minhas', 'quais as minhas', 'qual minha', 'diga a ','informe a','quero saber as', 'me fale as','qual é', 'diz a']+['']*10
            list2   = ['materias', 'materia', 'sala', 'disciplina','professor','local','turma','turmas','professores','disciplinas','salas','aula','aulas','grade','horario','classe','classes']
            list3 = ['agora','que devo ir','na segunda','na terca','na quarta','na quinta','na sexta','de manha','de tarde','de noite']+['']*7

            for i in range(n):
                a = random.choice(list1)+' '+random.choice(list2)+' '+random.choice(list3)
                intent['patterns'].append (a)


        if intent["tag"] == 'businfo':
            # # sinonimo de QUERO
            # list_bus1 = ['quero', 'desejo', 'pretendo', 'necessito', 'preciso', 'procuro']
            # # sinonimos de SABER
            # list_bus2 = ['saber', 'entender', 'me informar', 'conhecer']
            # # variacoes para QUE HORAS
            # list_bus3 = ['que horas', 'que momento', 'quando']
            # # sinonimos de PARTIR
            # list_bus4 = ['','sai', 'parte']
            # # sinonimos de onibus
            # list_bus5 = ['o onibus', 'o busao', 'o fretado', 'o transporte', 'a lotação']

            list_bus1 = ['quero saber','informe','qual','que hora','quando','quando sai','quero que sai','vai sair']+['']*6
            list_bus2 = ['fretado','fretados','onibus','busao','lotação']
            
            for i in range(n):
                #a = random.choice(list_bus1)+' '+random.choice(list_bus2)+' '+random.choice(list_bus3)+' '+random.choice(list_bus4)+' '+random.choice(list_bus5)
                a = random.choice(list_bus1)+' '+random.choice(list_bus2)
                intent['patterns'].append (a)

        if intent["tag"] == 'welcome':
            list_wel1 = ['oi','ola','salve','eae'] + ['']*4
            list_wel2 = ['bom dia','boa tarde','boa noite']+['']*5
            list_wel3 = ['bot','tatu','tatuzap']+['']*3

            for i in range(n):
                a = random.choice(list_wel1)+' '+random.choice(list_wel2)+' '+random.choice(list_wel3)
                intent['patterns'].append (a)
                
    return database

def gerar_anything(n):
    lista = []
    list1  = ['cachorro','esquilo','gato','camelo','cobra','tamandua','tatu','gaviao','bambi','touro','topeira']+['']*5
    list2   = ['voador','estiloso','esquisito','veloz','minha nossa','meus pets','de fogo','vingador','perneta']+['']*5
    list3 = ['de patinete','de patins','meu heroi','meu idolo','tchubilou','digdin','auu','dale do dele']+['']*7

    for i in range(n):
        a = random.choice(list1)+' '+random.choice(list2)+' '+random.choice(list3)
        lista.append (a)
    return lista


def print_dict(my_dict):
    keys, values = zip(*my_dict.items())
    print ("keys : ", str(keys))
    print ("values : ", str(values))


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

# database =fill_database(database, 15)
# print (database['intents'][2]['patterns'][0:10])
