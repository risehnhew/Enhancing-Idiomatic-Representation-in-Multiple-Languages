import gzip
import csv

train_samples = []
dev_samples = []
test_samples = []

# with open('stsbenchmark.tsv') as f:

# 	lines = csv.reader(f, delimiter="\t")
# 	with open('sts_train.txt','w') as f2:
# 		for i, line in enumerate(lines):
# 			if i>0 and len(line)==8:
# 				score = line[5]
# 				sent1 = line[6]
# 				# breakpoint()
# 				sent2 = line[7]
# 				f2.write(sent1 + '\t' + sent2 + '\t' + score + '\n')
sts_dataset_path ='./stsbenchmark.tsv.gz'
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    with open('sts_all.txt', 'w') as fall:
	    for row in reader:
	        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
	        inp_example = [row['sentence1'], row['sentence2'], str(score)]
	        fall.write(inp_example[0]+ '||' + inp_example[1]+ '||'+ inp_example[2]+ '\n')
	        if row['split'] == 'dev':
	            dev_samples.append(inp_example)
	        elif row['split'] == 'test':
	            test_samples.append(inp_example)
	        else:
	            train_samples.append(inp_example)

with open('sts_train.txt','w') as f1:
	for line in train_samples:
		f1.write(line[0]+ '||' + line[1]+ '||'+ line[2] + '\n')
with open('sts_dev.txt','w') as f2:
	for line in train_samples:
		f2.write(line[0]+ '||' + line[1]+ '||'+ line[2]+ '\n')
with open('sts_test.txt','w') as f3:
	for line in train_samples:
		f3.write(line[0]+ '||' + line[1]+ '||'+ line[2]+ '\n')
