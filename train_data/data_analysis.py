import csv
# from transformers import AutoTokenizer
# from evaluation_scripts.src.helper import lg2wordtokenize
from sentence_transformers import SentenceTransformer
import re
from sentence_transformers    import SentenceTransformer,  losses, models
import csv
from sklearn.metrics.pairwise         import paired_cosine_distances
import os
import sys
import re
import pandas as pd


model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def tokenise_idiom( phrase ) :
  return re.sub( r'[\s|-]', '_', phrase ).lower() 
def load_csv( path ) : 
  header = None
  data   = list()
  with open( path, encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile ) 
    for row in reader : 
      if header is None : 
        header = row
        continue
      data.append( row ) 
  return header, data
def get_similarities( location, model, languages=None, tokenize1= False ) : 
  sentences1, sentences2 = prepare_eval_data( location, languages, tokenize=tokenize1) # apply idiom principle

  #Compute embedding for both lists
  # breakpoint()

  embeddings1 = model.encode(sentences1, show_progress_bar=False, convert_to_numpy=True)
  embeddings2 = model.encode(sentences2, show_progress_bar=False, convert_to_numpy=True)

  # Compute cosine-similarits
  cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

  return cosine_scores


def prepare_eval_data( location, languages, test_print=False, tokenize =  False) :
  header, data = load_csv( location )
  sentence1s = list()
  sentence2s = list()
  mwe1s = list()
  mwe2s = list()
  sims = list()
  mwe_dict = {}
  for elem in data : 
    if not languages is None and not elem[ header.index( 'Language' ) ] in languages : 
      continue
    sentence1 = elem[ header.index( 'sentence_1' ) ] 
    sentence2 = elem[ header.index( 'sentence_2' ) ] 
    mwe1      = elem[ header.index( 'MWE1'      ) ] 
    mwe2      = elem[ header.index( 'MWE2'      ) ] 
    


    language = elem[ header.index( 'Language' ) ]
    alternative1 = elem[ header.index( 'alternative_1' ) ] 
    alternative2 = elem[ header.index( 'alternative_2' ) ]  

    #Before training your model, you must first make predictions on the STS between alternative_1 and alternative_2 
    # and use those values as the similarities between corresponding sentence_1 and sentence_2. 
    # You might find the pre-processing scripts made available with the baseline useful for this purpose. 
    sim = elem[header.index('sim')]


    if tokenize:

      new_mwe1 = tokenise_idiom( mwe1 )
      new_mwe2 = tokenise_idiom( mwe2 )
      if mwe1 != 'None' : 
        replaced = re.sub( mwe1, new_mwe1, sentence1, flags=re.I)
        assert replaced != sentence1
        sentence1 = replaced
      if mwe2 != 'None' : 
        replaced = re.sub( mwe1, new_mwe2, sentence2, flags=re.I)
        assert replaced != sentence2
        sentence2 = replaced
    
    sims.append(sim)
    if sim == '1':
      sentence1s.append( sentence1 ) 
      sentence2s.append( sentence2 )
  print(len(sentence1s))
  return sentence1s, sentence2s


scores = get_similarities( 'train_data.csv', model, languages=['EN', 'PT'], tokenize1= True )

print(sum(scores)/len(scores))
# new_sentence1s = idiom_target(sentence1s, mwe1s)

# print(len(new_sentence1s))