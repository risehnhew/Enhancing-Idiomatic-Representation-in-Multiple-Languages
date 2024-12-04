import argparse
import csv
import time
import re
import os
from SubTask2Evaluator                import evaluate_submission
from sklearn.metrics.pairwise         import paired_cosine_distances
from sentence_transformers    import SentenceTransformer
import pandas as pd
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='mirrorwic train')

    # Required
    parser.add_argument('--model_name', 
                        help='model name or Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--use_Dgold',  action="store_true")
    parser.add_argument('--use_Dsliver',  action="store_true")
    parser.add_argument('--using_ext_miner',  action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=240, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=3, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--train_model', action="store_true", 
            help="train the model")
    parser.add_argument('--parallel', action="store_true") 
    parser.add_argument('--disable_prog_bar', action="store_true",
            help="disable progress bar") 
    parser.add_argument('--random_seed',
                        help='epoch to train',
                        default=1996, type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss|infoNCE}}",
                        default="ms_loss")
    parser.add_argument('--training_mode',
                        help="{pre_training|fine_tuning}}",
                        default="ms_loss")
    parser.add_argument('--infoNCE_tau', default=0.04, type=float) 
    parser.add_argument('--use_miner', action="store_true") 
    parser.add_argument('--miner_margin', default=0.2, type=float) 
    parser.add_argument('--triplet_margin', default= 0.1, type=float, help='The desired difference between the pos and neg distance')
    parser.add_argument('--type_of_triplets', default="all", type=str) 
    parser.add_argument('--STS_model', default="all-mpnet-base-v2", type=str)
    parser.add_argument('--device_name', default="0", type=str)
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean_pool|tokenmarker4layer|tokenmarker2layer}") 
    parser.add_argument('--add_idoms_to_tokenizer', action="store_true")
    parser.add_argument('--dropout_rate', default=0.1, type=float) 
    parser.add_argument('--is_shuffle', action="store_true")
    parser.add_argument('--pad_in_eos', action="store_true")
    parser.add_argument('--no_progress_bar', action="store_true")
    parser.add_argument('--mean_layer_nums', help='choose numbers of layer to mean', default=4, type=int)    

    args = parser.parse_args()
    return args


def tokenise_idiom( phrase ) :
  return 'ID' + re.sub( r'[\s|-]', '', phrase ).lower() + 'ID' # break compositionality


def prepare_eval_data( location, languages, test_print=False, tokenize =  True) :
  header, data = load_csv( location )
  sentence1s = list()
  sentence2s = list()
  for elem in data : 
    if not languages is None and not elem[ header.index( 'Language' ) ] in languages : 
      continue
    sentence1 = elem[ header.index( 'sentence1' ) ] 
    sentence2 = elem[ header.index( 'sentence2' ) ] 
    mwe1      = elem[ header.index( 'MWE1'      ) ] 
    mwe2      = elem[ header.index( 'MWE2'      ) ] 

    if test_print : 
      print( sentence1 ) 
      print( sentence2 ) 
      print( mwe1 ) 
      print( mwe2 ) 
    if tokenize:
      if mwe1 != 'None' : 
        replaced = re.sub( mwe1, tokenise_idiom( mwe1 ), sentence1, flags=re.I)
        assert replaced != sentence1
        sentence1 = replaced
      if mwe2 != 'None' : 
        replaced = re.sub( mwe1, tokenise_idiom( mwe2 ), sentence2, flags=re.I)
        assert replaced != sentence2
        sentence2 = replaced

    if test_print : 
      print( sentence1 ) 
      print( sentence2 ) 
      break

    sentence1s.append( sentence1 ) 
    sentence2s.append( sentence2 ) 

  return sentence1s, sentence2s


def prepare_eval_data2( location, languages, test_print=False, tokenize =  False) :
  header, data = load_csv( location )
  sentence1s = list()
  sentence2s = list()
  for elem in data : 
    if not languages is None and not elem[ header.index( 'Language' ) ] in languages : 
      continue
    sentence1 = elem[ header.index( 'sentence1' ) ] 
    sentence2 = elem[ header.index( 'sentence2' ) ] 
    mwe1      = elem[ header.index( 'MWE1'      ) ] 
    mwe2      = elem[ header.index( 'MWE2'      ) ] 

    if test_print : 
      print( sentence1 ) 
      print( sentence2 ) 
      print( mwe1 ) 
      print( mwe2 ) 
    if tokenize:
      if mwe1 != 'None' : 
        replaced = re.sub( mwe1, tokenise_idiom( mwe1 ), sentence1, flags=re.I)
        assert replaced != sentence1
        sentence1 = replaced
      if mwe2 != 'None' : 
        replaced = re.sub( mwe1, tokenise_idiom( mwe2 ), sentence2, flags=re.I)
        assert replaced != sentence2
        sentence2 = replaced

    if test_print : 
      print( sentence1 ) 
      print( sentence2 ) 
      break

    sentence1s.append( [sentence1, sentence2] ) 

  return sentence1s + sentence2s
def get_similarities( location, model, languages=None, tokenize1= True ) : 
  sentences1, sentences2 = prepare_eval_data( location, languages, tokenize=tokenize1) # apply idiom principle

  #Compute embedding for both lists
  # breakpoint()

  embeddings1 = model.encode(sentences1, show_progress_bar=False, convert_to_numpy=True)
  embeddings2 = model.encode(sentences2, show_progress_bar=False, convert_to_numpy=True)
  # breakpoint()

  # Compute cosine-similarits
  cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

  return cosine_scores

def write_csv( data, location ) : 
  with open( location, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer( csvfile ) 
    writer.writerows( data ) 
  # print( "Wrote {}".format( location ) ) 
  return


def insert_to_submission( languages, settings, sims, location ) : 
  header, data = load_csv( location ) 
  sims = list( reversed( sims ) )
  ## Validate with length
  updatable = [ i for i in data if i[ header.index( 'Language' ) ] in languages and i[ header.index( 'Setting' ) ] in settings ]
  assert len( updatable ) == len( sims ) 

  ## Will update in sequence - if data is not in sequence must update one language / setting at a time. 
  started_update = False
  for elem in data : 
    if elem[ header.index( 'Language' ) ] in languages and elem[ header.index( 'Setting' ) ] in settings : 
      sim_to_insert = sims.pop()
      elem[-1] = sim_to_insert
      started_update = True
    else :  
      assert not started_update ## Once we start, we must complete. 
    if len( sims ) == 0 : 
      break 
  assert len( sims ) == 0 ## Should be done here. 

  return [ header ] + data ## Submission file must retain header. 

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

def get_result(path, model, mode, languages, if_tokenize, gen_result= False, not_print = False):
  
    basic_path = '/mnt/parscratch/users/ac1whe/To_stanage/SubTaskB'
    outpath = path 
    dev_location = basic_path + '/EvaluationData/' + mode+ '.csv'
    formated_file_location = basic_path +"/EvaluationData/" + mode+ ".submission_format.csv"
    # breakpoint()
     
    dev_sims  = get_similarities(dev_location, model, languages, tokenize1 = True)
    submission_data = insert_to_submission( languages, [ 'fine_tune' ], dev_sims, formated_file_location )
    
    
    ## Evaluate development set. 

    if gen_result:
      # breakpoint()
      results_file    = os.path.join( outpath, mode+'.fine_tune_results-' + '.csv' )
      write_csv(submission_data, results_file )
      
      dev_gold_path = os.path.join( basic_path +'/EvaluationData/', mode+'.gold.csv' )


      results = evaluate_submission( results_file, dev_gold_path )

      ## Make results printable. 
      for result in results : 
        for result_index in range( 2, 5 ) : 
          result[result_index] = 'Did Not Attempt' if result[result_index] is None else result[ result_index ]
      # breakpoint()
      if not not_print:
        print(results[4])
        print(results[5])
        print(results[6])
      return results[6]
    else:
      if os.path.exists(outpath+ '/submission/') == False:
        os.makedirs(outpath+ '/submission/')
      results_file    = os.path.join( outpath+ '/submission/', 'task2_subtaskb.csv')
      write_csv(submission_data, results_file )
      return results_file
def get_result_trainer(eval_preds):
  


  preds, labels = eval_preds
  
  n = 160

  maxrange = n*args.epoch+1
  
  model1 = SentenceTransformer(args.output_dir+'/checkpoint-'+ str(n))
  result = get_result(path = args.output_dir, model= model1, mode = 'dev', languages = ['EN', 'PT'], if_tokenize = args.add_idoms_to_tokenizer, gen_result= True, not_print = True)

  best_result = result[3]
  # print('Best result:', best_result)
  all_score.append(round(result[2],3))
  idiom_score.append(round(result[3],3))
  STS_score.append(round(result[4],3))
  result_path = get_result(path = args.output_dir, model = model1, mode = 'test', languages = ['EN', 'PT', 'GL'], if_tokenize = args.add_idoms_to_tokenizer)    
  zip_file = zipfile.ZipFile(args.output_dir+ '/' + args.agg_mode[-6:] + str(int(number/n)) +'_' +args.model_name[-5:]+'.zip', 'w')
  zip_file.write(result_path,'./submission/task2_subtaskb.csv')
  zip_file.close()
      # if result[3] > best_result:
      #     model_wrapper.save_model(args.output_dir)
  print(all_score)
  print(colored(idiom_score, 'green'))
  print(STS_score)
  print(train_losses)
  print(sum(all_score)/len(all_score))
  
  return 1   

def get_gold_labels(gold_labels_loc ): 
  gold_headers, gold_data = load_csv( gold_labels_loc)
  gold_labels_all=[]
  filtered_submission_dict = dict() 
  for elem in gold_data : 
    filtered_submission_dict[ elem[ gold_headers.index( 'ID' ) ] ] = elem[ gold_headers.index( 'sim' ) ]
  for elem in gold_data :
    this_sim = elem[ gold_headers.index( 'sim' ) ]
    if this_sim == '' : 
      breakpoint()
      this_sim = filtered_submission_dict[ elem[ gold_headers.index( 'otherID' ) ] ]
    this_sim = float(this_sim)
    gold_labels_all.append(this_sim)
  # sims = [float(ele[3]) for ele in gold_data]
  return gold_labels_all



def get_triplet_similarities(data_path, model_name='paraphrase-multilingual-mpnet-base-v2'):
  from sentence_transformers import SentenceTransformer, util
  from sklearn.metrics.pairwise import cosine_similarity
  model = SentenceTransformer(model_name)
  header, data = load_csv( data_path )
  sentences1=[]
  sentences2=[]
  cosine_scores=[]
  for i, (label1, sentence1) in enumerate(data):
    for j, (label2, sentence2) in enumerate(data):
      if label1 == label2 and i!=j:
        sentences1.append(sentence1)
        sentences2.append(sentence2)
        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)
        cosine_scores.append(util.cos_sim(embeddings1, embeddings2).item())
  df = pd.DataFrame(cosine_scores, columns=['column_name'])
  print(df.nsmallest(100,'column_name'))
  print(df.describe())
  

  return cosine_scores

if __name__ == '__main__':
  
  get_triplet_similarities('../train_data/best_data_trainer.csv', model_name='thenlper/gte-large')

