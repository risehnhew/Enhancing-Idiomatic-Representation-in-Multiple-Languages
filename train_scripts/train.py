#!/usr/bin/env python
import argparse
import torch
import torch.optim as optim
from termcolor import colored
import random


from transformers import AutoTokenizer, AutoModel,AutoConfig,PreTrainedModel, TrainingArguments
from datasets import load_dataset

import os
from sentence_transformers    import SentenceTransformer
from help_funcs_wei import  parse_args
import zipfile
import logging

from help_funcs_wei import get_result_trainer,get_similarities, prepare_eval_data2, get_result,get_gold_labels

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.metric_learning_wei import MyTrainer, MetricLearningDataset_pairwise_trainer, MetricLearningDataset_pairwise_trainer_eval

logging.disable(logging.WARNING)
torch.manual_seed(33)
random.seed(33)
def main(args):
    print("type_of_triplets: {}".format(colored(args.type_of_triplets, 'green')))
    print("miner_margin: {}".format(colored(args.miner_margin, 'red')))
    print("triplet_margin: {}".format(colored(args.triplet_margin, 'green')))
    print("dropout_rate: {}".format(colored(args.dropout_rate, 'red')))
    print("device_name: {}".format(colored(args.device_name, 'green')))
    print("learning_rate:{}".format(colored(args.learning_rate, 'red')))
    print("Num_of_layers:{}".format(colored(args.agg_mode, 'green')))
    print("batch_size:{}".format(colored(args.train_batch_size, 'red')))
    print("Gold_data:{}".format(colored(args.use_Dgold, 'green')))
    print("Model Name:{}".format(colored(args.model_name, 'red')))
    print("Using eos as pad:{}".format(colored(args.pad_in_eos, 'green')))


    # torch.manual_seed(args.random_seed)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        


    #------------------------huggingface loading------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) # Using pre-trained model's tokenizer
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    tokenizer.model_max_length = args.max_length
    # breakpoint()
    if args.pad_in_eos:
        tokenizer.pad_token = tokenizer.eos_token

  

    # encoder = AutoModel.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name,
        STS_name = args.STS_model,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_cuda=torch.cuda.is_available()  ,
        loss=args.loss,
        infoNCE_tau=args.infoNCE_tau,
        use_miner=args.use_miner,
        miner_margin=args.miner_margin,
        triplet_margin=args.triplet_margin,
        type_of_triplets=args.type_of_triplets,
        agg_mode=args.agg_mode,
        output_hidden_states=True,
        )
    config.hidden_dropout_prob = args.dropout_rate
    config.attention_probs_dropout_prob = args.dropout_rate
    # breakpoint()
    if args.train_model:
        model = AutoModel.from_pretrained(args.model_name,config=config)

        if args.use_Dgold:
        # train_valid_test_files = {"train": args.train_dir,}
        # train_valid_test_files_gold = {"train": '/local/wh1n18/Sheffield/MirrorWiC2/train_data/gold_relable.csv',}
            train_valid_test_dataset = load_dataset('csv', data_files=['./train_data/best_data_trainer.csv','./train_data/gold_relable.csv'])
        else:
            train_valid_test_files = {"train": args.train_dir}
            train_valid_test_dataset = load_dataset('csv', data_files=train_valid_test_files)
        if args.use_Dsliver:
            train_valid_test_dataset = load_dataset('csv', data_files=['./train_data/best_data_trainer.csv','./train_data/gold_relable.csv', './train_data/sliver_relable.csv'])
        if args.add_idoms_to_tokenizer:
            idioms = []
            with open('./train_data/train_idioms_add.txt') as f1:
            # with open('./train_data/added_tokens.txt') as f1:

                for idiom in f1:
                    idioms.append(idiom.strip('\n'))

            
            idioms = list(set(idioms))
            if args.use_Dgold:
                idioms2=[]
                with open('./train_data/gold_mwes.txt') as f2: 

                    for idiom2 in f2:
                        idioms2.append(idiom2.strip('\n'))
                idioms+=list(set(idioms2))
                if args.use_Dsliver:
                    idioms3=[]
                    with open('./train_data/silver_mwes.txt') as f3: 

                        for idiom3 in f3:
                            idioms3.append(idiom3.strip('\n'))
                    idioms+=list(set(idioms3))
        
            #-------------------------------------------------------
            # add idioms in tokenizer
            old_len        = len( tokenizer )
            # breakpoint()
            num_added_toks = tokenizer.add_tokens( idioms ) 
            # print( "Old tokenizer length was {}. Added {} new tokens. New length is {}.".format( old_len, num_added_toks, len( tokenizer ) )  ) 
            model.resize_token_embeddings(len(tokenizer))
            print( "Found a total of {} idioms".format( len( idioms ) ))

        train_encodings = tokenizer(train_valid_test_dataset['train']['sentence'], truncation=True, padding=True)
        

        train_set = MetricLearningDataset_pairwise_trainer(
            train_encodings,
            train_valid_test_dataset['train']['labels'])

        arguments = TrainingArguments(remove_unused_columns=False, 
            output_dir= args.output_dir,
            per_device_train_batch_size = args.train_batch_size,
            save_strategy ='steps',
            save_steps = 480,
            num_train_epochs=args.epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            report_to="none",
            logging_strategy ="epoch",
            disable_tqdm=args.disable_prog_bar,
            local_rank=1,

            # do_eval = True,
            # evaluation_strategy='epoch'
            )
        trainer =  MyTrainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        tokenizer=tokenizer,
        # eval_dataset = eval_set,
        # compute_metrics =  get_result_trainer
            )
        
        trainer.train()
 
    # maxrange = 3360

    ## Make sure this worked. 
    # print( tokenizer.tokenize('This is a IDancienthistoryID'), flush=True )
    # print( tokenizer.tokenize( 'This is a IDcolégiomilitarID' ) )

    #-------------------------------------------------------


    
    # mask_token_id = tokenizer.encode("[MASK]")[1]
    # print ("[MASK] token ID:", mask_token_id)



    # breakpoint()
    step_global = 0
    best_result = 0
    idiom_score = []
    STS_score = []
    all_score = []
    train_losses = []  
    # basic_path = '/local/wh1n18/Sheffield/spare/SemEval_2022_Task2-idiomaticity/SubTaskB'
    # sentences = [[],[]]

    # eval_data = prepare_eval_data2( location = basic_path + '/EvaluationData/dev.csv', languages= ['EN', 'PT'], test_print=False, tokenize =  False)
    # # # breakpoint()

    # # eval_label = get_gold_labels('/local/wh1n18/Sheffield/spare/SemEval_2022_Task2-idiomaticity/SubTaskB/EvaluationData/dev.gold.csv')
    # eval_label = list(range(len(eval_data)))
    # eval_encodings = tokenizer(eval_data, truncation=True, padding=True)
    # # breakpoint()


    # eval_set = MetricLearningDataset_pairwise_trainer_eval(eval_encodings, eval_label)
    
    # model1 = SentenceTransformer(args.output_dir+'/temp/'+ str(1))

    # eval_metric = get_result_trainer(path = args.output_dir, model= model1, mode = 'dev', languages = ['EN', 'PT'], if_tokenize = args.add_idoms_to_tokenizer, gen_result= True)


    print('Starting evaluation...')

    if args.train_model:
        m = int(len(train_set)/args.train_batch_size) +1
        n = m*3
        # n = 480

        maxrange = m*args.epoch+1
        for number in range(n, maxrange, n):
        # for number in range(1, 15, 1):
            # breakpoint()
            model1 = SentenceTransformer(args.output_dir+'/checkpoint-'+ str(number))
            # model1 = SentenceTransformer(args.model_name)
            # breakpoint()
            if args.pad_in_eos:
                model1.tokenizer.pad_token = model1.tokenizer.eos_token

            result = get_result(path = args.output_dir, model= model1, mode = 'dev', languages = ['EN', 'PT'], if_tokenize = True, gen_result= True, not_print = True)

            best_result = result[3]
            # print('Best result:', best_result)
            all_score.append(round(result[2],3))
            idiom_score.append(round(result[3],3))
            STS_score.append(round(result[4],3))
            result_path = get_result(path = args.output_dir, model = model1, mode = 'test', languages = ['EN', 'PT', 'GL'], if_tokenize = True)    
            zip_file = zipfile.ZipFile(args.output_dir+ '/' + args.agg_mode[-6:] + str(int(number/n)) +'_' +args.model_name[-5:]+'.zip', 'w')
            # zip_file = zipfile.ZipFile(args.output_dir+ '/' + args.agg_mode[-6:] + '_' +args.model_name[-5:]+'.zip', 'w')

            zip_file.write(result_path,'./submission/task2_subtaskb.csv')
            zip_file.close()
            # if result[3] > best_result:
            #     model_wrapper.save_model(args.output_dir)
    else:

        model1 = SentenceTransformer(args.model_name)
        # breakpoint()
        if args.pad_in_eos:
            model1.tokenizer.pad_token = model1.tokenizer.eos_token

        result = get_result(path = args.output_dir, model= model1, mode = 'dev', languages = ['EN', 'PT'], if_tokenize = True, gen_result= True, not_print = True)

        best_result = result[3]
        # print('Best result:', best_result)
        all_score.append(round(result[2],3))
        idiom_score.append(round(result[3],3))
        STS_score.append(round(result[4],3))
        result_path = get_result(path = args.output_dir, model = model1, mode = 'test', languages = ['EN', 'PT', 'GL'], if_tokenize = True)    
        # zip_file = zipfile.ZipFile(args.output_dir+ '/' + args.agg_mode[-6:] + str(int(number/n)) +'_' +args.model_name[-5:]+'.zip', 'w')
        zip_file = zipfile.ZipFile(args.output_dir+ '/' +args.model_name[-10:]+'.zip', 'w')

        zip_file.write(result_path,'./submission/task2_subtaskb.csv')
        zip_file.close()

    print(all_score)
    print(colored(idiom_score, 'green'))
    print(STS_score)
    print(train_losses)
    print(sum(all_score)/len(all_score))

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
