import torch
from pytorch_metric_learning import miners, losses, distances
from pytorch_metric_learning.distances import CosineSimilarity
from transformers import  Trainer
from torch.utils.data import DataLoader, Dataset

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

import sys
sys.path.append("../") 
from help_funcs_wei import  parse_args
args2 = parse_args()

# metric = losses.TripletMarginLoss(margin=args.triplet_margin, distance=CosineSimilarity())


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.loss_cust=losses.TripletMarginLoss(margin=args2.triplet_margin, distance=CosineSimilarity())
        self.miner_cust=miners.TripletMarginMiner(margin=args2.miner_margin, type_of_triplets=args2.type_of_triplets, distance=CosineSimilarity())


    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        # breakpoint()
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        if args2.agg_mode =='tokenmarker4layer':
            hidden_state1=sum(hidden_states[-4:])/4    
        elif args2.agg_mode =='tokenmarker2layer':
            hidden_state1=sum(hidden_states[-2:])/2
        elif args2.agg_mode =='all_layer':
            hidden_state1=sum(hidden_states[:])/len(hidden_states)
        elif args2.agg_mode=='last_hidden':
            hidden_state1=outputs.last_hidden_state
        query_embed1 = hidden_state1.mean(1)  # query : [batch_size, embed_size]


        # miner = miners.TripletMarginMiner(margin=args.miner_margin, type_of_triplets=args.type_of_triplets, distance=CosineSimilarity())
        if args2.miner_margin>0:
            hard_pairs = self.miner_cust(query_embed1, labels)
            # breakpoint()
            return self.loss_cust(query_embed1, labels, hard_pairs)
        else:
            return self.loss_cust(query_embed1, labels)
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle":False
        }
        return DataLoader(train_dataset, **dataloader_params)


def get_result_trainer(eval_preds):
  basic_path = '/local/wh1n18/Sheffield/spare/SemEval_2022_Task2-idiomaticity/SubTaskB'

  

  # eval_preds
  # Compute cosine-similarits
  # dev_sims = 1 - (paired_cosine_distances(embeddings1, embeddings2))
  # submission_data = insert_to_submission( languages, [ 'fine_tune' ], dev_sims, formated_file_location )
  
  
  ## Evaluate development set. 


    
  # dev_gold_path = os.path.join( basic_path +'/EvaluationData/', mode+'.gold.csv' )

  return 1 

class MetricLearningDataset_pairwise_trainer(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, encodings, labels): #d_ratio, s_score_matrix, s_candidate_idxs):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class MetricLearningDataset_pairwise_trainer_eval(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, encodings, labels): #d_ratio, s_score_matrix, s_candidate_idxs):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)