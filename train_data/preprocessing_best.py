import csv
from annotation_eval import load_csv
import re

def pre_processing():
	header = ['labels', 'sentence']
	with open('best_data_trainer.csv', 'w') as f2:
		datawriter = csv.writer(f2)
		with open('best_data') as f:
			datawriter.writerow(header)
			for sent in f.readlines():
				label, name1, name2 = sent.rstrip("\n").split("||")
				name1 = name1.replace('[ ', '').replace('] ','')
				label = int(label.replace('en', '10').replace('pt', '20'))
				datawriter.writerow([label, name1])



def tokenise_idiom( phrase ) :
	return 'ID' + re.sub( r'[\s|-]', '', phrase ).lower() + 'ID' # break compositionality
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


def prepare_eval_data( location, languages, test_print=False, tokenize =  False):
	header, data = load_csv( location )
	sentence1s = list()
	sentence2s = list()
	mwe1s = list()
	mwe2s = list()
	sims = list()
	mwe_dict = {}
	for elem in data : 
		sentence1 = elem[ header.index( 'sentence_1' ) ] 
		sentence2 = elem[ header.index( 'sentence_2' ) ] 
		mwe1      = elem[ header.index( 'MWE1'      ) ] 
		mwe2      = elem[ header.index( 'MWE2'      ) ] 
		


		language = elem[ header.index( 'Language' ) ]
		# alternative1 = elem[ header.index( 'alternative_1' ) ] 
		# alternative2 = elem[ header.index( 'alternative_2' ) ]  

		#Before training your model, you must first make predictions on the STS between alternative_1 and alternative_2 
		# and use those values as the similarities between corresponding sentence_1 and sentence_2. 
		# You might find the pre-processing scripts made available with the baseline useful for this purpose. 
		# sim = elem[header.index('sim')]


		# breakpoint()
		if tokenize:
			if mwe1 != 'None' : 
				replaced1 = re.sub( mwe1, new_mwe1, sentence1, flags=re.I)
				if (replaced1 != sentence1):
					sentence1 = replaced1
				else:
					replaced2 = re.sub( mwe1, new_mwe1, sentence2, flags=re.I)
					sentence2 = replaced2

			if mwe2 != 'None' : 
				replaced = re.sub( mwe1, new_mwe2, sentence2, flags=re.I)
				assert replaced != sentence2
				sentence2 = replaced
			new_mwe1 = tokenise_idiom( mwe1 )
			new_mwe2 = tokenise_idiom( mwe2 )
		else:
			new_mwe1 = mwe1
			new_mwe2 = mwe2
		
		if mwe1 in mwe_dict:
			mwe_dict[new_mwe1].append([sentence1, sentence2])
		else:
			mwe_dict[new_mwe1] = [[sentence1, sentence2]]
		
		sentence1s.append( sentence1 ) 
		sentence2s.append( sentence2 )
		mwe1s.append(new_mwe1) 
		mwe2s.append(new_mwe2)
		# sims.append(sim)

	# breakpoint()
	return sentence1s, sentence2s, mwe1s, mwe2s,  mwe_dict
if __name__ == '__main__':
	data_location = 'train_data.csv'


	sentence1s, sentence2s, mwe1s, mwe2s, mwe_dict =  prepare_eval_data( location = data_location, languages='en', test_print=False, tokenize =  False)
	with open('train_data_mwes.txt', 'w') as f2:
		for mwe in set(mwe1s):
			f2.write(mwe+'\n')


	with open('train_data_relable.csv', 'w') as f:
		header = ['labels', 'sentence']
		writer = csv.writer(f)
		list1 = []
		list2 = []
		i = 0
		writer.writerow(header)
		for sent1, sent2, mwe1 in zip(sentence1s, sentence2s, mwe1s):
			label = '10'+str(i)
			label = int(label)
			
			writer.writerow([label, sent1])
			writer.writerow([label, sent2])
			# else:
			# 	writer.writerow([label, sent1])
			# 	i+=1
			# 	label = '30'+str(i)
			# 	label = int(label)
			# 	writer.writerow([label, sent2])
			i+=1

	print('finish')