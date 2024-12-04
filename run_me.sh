#!/bin/sh

model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2" 

rand=5
device_name='0'
export CUDA_VISIBLE_DEVICES=$device_name
# 1e-5 5e-6 2e-6 1e-6
for triplet_margin in  0.3
do
   for miner_margin in  0.4 
   do
      for model in $model
      do
      infoce=0.04
      maxlen=512 # tuning
      bs=64
      dropout=0.05
      # agg='tokenmarker4layer'
      # agg='tokenmarker2layer'
      # agg='last_hidden'
      # agg='all_layer'
      # lr=4e-06
      rs=33

      data=./train_data/best_data_trainer.csv

      # --loss "infoNCE" \
      # --learning_rate 2e-5 \
      # --miner_margin $margin \  
      # accelerate launch
      /users/ac1whe/anaconda3/envs/py1/bin/python  ./train_scripts/WiC_train_Wei.py \
         --model_name ${model} \
         --train_dir "${data}"  \
         --output_dir ./model/${model}/$device_name \
         --epoch 24 \
         --train_batch_size $bs \
         --learning_rate 2e-5 \
         --max_length ${maxlen} \
         --parallel \
         --random_seed ${rs} \
         --loss "triplet_loss" \
         --infoNCE_tau ${infoce} \
         --dropout_rate ${dropout} \
         --agg_mode 'tokenmarker2layer' \
         --miner_margin  $miner_margin\
         --type_of_triplets 'all'\
         --triplet_margin $triplet_margin\
         --training_mode "pre_training"\
         --use_miner  \
         --device_name $device_name \
         --train_model\
         --add_idoms_to_tokenizer        
         # --use_Dgold \
         # --use_Dsliver\
         # --train_model\
         # --pad_in_eos \
         
         # --disable_prog_bar \
         

         # --no_progress_bar \
         # --using_ext_miner \         
         # --is_shuffle \



      # eval_script=./SemEval_2022_Task2-idiomaticity/SubTaskB/get_sim.py
      # nohup bash create_data.sh & > output.txt
      done
   done
done


