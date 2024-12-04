import csv
# from transformers import AutoTokenizer
# from evaluation_scripts.src.helper import lg2wordtokenize
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr
from sentence_transformers    import SentenceTransformer,  losses, models
import csv
from sklearn.metrics.pairwise         import paired_cosine_distances
import os
import sys
import re
import pandas as pd
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import nltk
import torch

# model_name ='/local/wh1n18/Sheffield/MirrorWiC2/model/sentence-transformers/paraphrase-multilingual-mpnet-base-v2_mirror/temp/3/' #0.374

# model_name ='/local/wh1n18/Sheffield/MirrorWiC2/model/sentence-transformers/paraphrase-multilingual-mpnet-base-v2_mirror/temp/1/' #0.277

model_name = 'xlm-roberta-base'
# model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# model = SentenceTransformer(model_name)

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

def remove_tags(string):
  return string.replace("<b>", "").replace("</b>", "").replace("<strong>", "").replace("</strong>", "").replace("[", "").replace("]", "")

def replace_NC(sentence, replacement, Identifiers):

  start_index = sentence.find(Identifiers[0])
  end_index = sentence.find(Identifiers[1])
  return sentence[:start_index] + replacement + sentence[end_index+4:]

def get_similarities( ori_sentences, head_sens, modifier_sens) : 
  # head_sens, modifier_sens, head_scores, modifier_scores, ori_sentences = prepare_eval_data( location, languages, tokenize=tokenize1) # apply idiom principle

  #Compute embedding for both lists
  # breakpoint()
  embeddings_ori =   model.encode(ori_sentences, show_progress_bar=False, convert_to_numpy=True)
  embeddings_head = model.encode(head_sens, show_progress_bar=False, convert_to_numpy=True)
  embeddings_modifier = model.encode(modifier_sens, show_progress_bar=False, convert_to_numpy=True)

  # Compute cosine-similarits
  cosine_scores_modi = 1 - (paired_cosine_distances(embeddings_modifier, embeddings_ori))
  cosine_scores_head = 1 - (paired_cosine_distances(embeddings_head, embeddings_ori))

  return cosine_scores_modi, cosine_scores_head


def prepare_eval_data( location, test_print=False, tokenize =  False) :
  header, data = load_csv( location )
  ori_sentences = list()
  head_scores = list()
  modifier_scores = list()
  NC_compositional_scores = list()
  compounds = list()
  modifiers=list()
  heads = list()
  head_sens = list()
  modifier_sens = list()
  start_indexs = []
  for elem in data : 
    sentence = elem[ header.index( 'Input.examplesent1' ) ] 
    head_score = elem[ header.index( 'Answer.Qhead' ) ] 
    modifier_score      = elem[ header.index( 'Answer.Qmodifier'      ) ] 
    NC_compositional_score      = elem[ header.index( 'Answer.Qheadmodifier'      ) ] 
    


    compound = elem[ header.index( 'Input.compound' ) ]
    modifier = elem[ header.index( 'Input.modifier' ) ] 
    head = elem[ header.index( 'Input.head' ) ]  
    


    if "<b>" in sentence:
      index = sentence.find("<b>")
      start_index = len(sentence[:index].split())
      Identifiers =['<b>', '</b>']
      sentence_head = replace_NC(sentence, head, Identifiers)
      sentence_modifier = replace_NC(sentence, modifier, Identifiers)
    elif "<strong>" in sentence:
      index = sentence.find("<strong>")
      start_index = len(sentence[:index].split())
      Identifiers =['<strong>', '</strong>']
    elif "[" in sentence:
      index = sentence.find("[")
      start_index = len(sentence[:index].split())
      Identifiers =['[', ']']
    elif compound in sentence:
      index = sentence.find(compound)
      start_index = len(sentence[:index].split())
    else:
      print(sentence)
      continue

    NC_compositional_scores.append(float(NC_compositional_score))
    head_scores.append(head_score)
    modifier_scores.append(modifier_score)

    sentence_head = replace_NC(sentence, head, Identifiers)
    sentence_modifier = replace_NC(sentence, modifier, Identifiers)

    head_sens.append(sentence_head)
    modifier_sens.append(sentence_modifier)
    start_indexs.append(start_index)
    ori_sentences.append(remove_tags(sentence))
    compounds.append(compound)
    # breakpoint()
  return head_sens, modifier_sens, head_scores, modifier_scores, ori_sentences, compounds, start_indexs, NC_compositional_scores

def prepare_eval_data2( location, test_print=False, tokenize =  False) :
  header, data = load_csv( location )
  ori_sentences = list()
  head_scores = list()
  modifier_scores = list()
  NC_compositional_scores = list()
  compounds = list()
  modifiers=list()
  heads = list()
  head_sens = list()
  modifier_sens = list()
  for elem in data : 
    sentence = elem[ header.index( 'examplesent1' ) ] 
    head_score = elem[ header.index( 'answer-head' ) ] 
    modifier_score      = elem[ header.index( 'answer-modifier'      ) ] 
    NC_compositional_score      = elem[ header.index( 'answer-headModifier'      ) ] 
    


    compound = elem[ header.index( 'Input.compound' ) ]
    modifier = elem[ header.index( 'Input.modifier' ) ] 
    head = elem[ header.index( 'Input.head' ) ]  
    
    head_scores.append(head_score)
    modifier_scores.append(modifier_score)


    sentence_head = replace_NC(sentence, head)
    sentence_modifier = replace_NC(sentence, modifier)

    head_sens.append(sentence_head)
    modifier_sens.append(sentence_modifier)
    ori_sentences.append(remove_tags(sentence))
    # breakpoint()

  return head_sens, modifier_sens, head_scores, modifier_scores, ori_sentences
head_sens, modifier_sens, head_scores, modifier_scores, ori_sentences, compounds, start_indexs, NC_compositional_scores = prepare_eval_data('annotations_mean.csv') # apply idiom principle

# head_sensP, modifier_sensP, head_scoresP, modifier_scoresP, ori_sentencesP = prepare_eval_data2('annotations_mean_protuguese.csv') # apply idiom principle

# combine_ori_sent = ori_sentences
# combine_head_sens = head_sens
# combine_modifier_sens = modifier_sens 

# scores_modi, scores_head = get_similarities(ori_sentences, head_sens, modifier_sens )

# combine_modifier_scores = modifier_scores 
# combine_head_scores = head_scores 

# corel_modi, pvalue    =  spearmanr(modifier_scores, scores_modi) 
# breakpoint()
# corel_head, pvalue    =  spearmanr(head_scores, scores_head) 



# print(sum(scores_modi)/len(scores_modi))
# print(sum(scores_head)/len(scores_head))

# print(corel_modi)
# print(corel_head)


def NC_out(noun_compound, model):

  """ vector for the Noun compounds out of context"""
  embeddings = model(noun_compound)

  return embeddings

def NC_out_comp(noun_compuond, model):

  words = noun_compound.split()
  embeddings = []
  for word in words:
    embedding = model(word)
    embeddings.append(embedding)


  return sum(embeddings)

# def token_level():

#   return

def In_context_eval(sentences, model_name, MWEs, start_indexs):
  mwe_incontext_embeddings = []
  # init embedding
  embedding = TransformerWordEmbeddings(model_name)
  # breakpoint()
  mwe_outcontext_embeddings = []
  # create a sentence
  for sent, mwe, start_index in zip(sentences, MWEs, start_indexs):
    sentence = Sentence(sent)
    multi_we = Sentence(mwe)

    # embed words in sentence
    embedding.embed(sentence)

    embedding.embed(multi_we)

    # for sig_word in multi_we:
    #   mwe_outcontext_embeddings.append(sig_word)
    # breakpoint()
    mwe_outcontext_embeddings.append(torch.mean(torch.stack([sig.embedding.to('cpu') for sig in multi_we]), dim=0)) # average the word embeddings NC_out
    
    mwe = mwe.split()
    mwe_len= len(mwe)
    # sent = sent.split()
    # index = sent.index(mwe[0])

    # breakpoint()
    tokens = sentence[start_index: start_index+mwe_len]
    mwe_embedding = []
    for token in tokens:
        # print(token)
        # print(token.embedding)
        mwe_embedding.append(token.embedding.to('cpu'))
    # breakpoint()

    mwe_incontext_embeddings.append(torch.mean(torch.stack(mwe_embedding), dim=0))
  similarities = 1 - (paired_cosine_distances(torch.stack(mwe_incontext_embeddings), torch.stack(mwe_outcontext_embeddings)))
  # print(similarities)
  return similarities



similarities = In_context_eval(ori_sentences, model_name, compounds, start_indexs)
# breakpoint()
corel_context, pvalue    =  spearmanr(similarities, torch.tensor(NC_compositional_scores))
# breakpoint()
print(corel_context)
