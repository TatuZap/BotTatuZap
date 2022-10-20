all: install

install:
	pip install nltk 
	pip install -U spacy
	pip install unidecode 
	python -m spacy download pt_core_news_lg
	pip install tensorflow
	pip install pandas
	pip install numpy
	python -m pip install -U matplotlib
	