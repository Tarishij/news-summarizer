
import os
from summarizer.pre_processor import PreProcessor
from summarizer.centroid_embeddings import summarize
from summarizer.mmr_summarizer import MMRsummarize
from summarizer.sentenceOrdering import sentencePositioning
import nltk
from bs4 import BeautifulSoup


main_folder_path = os.getcwd() + "/Input_DUC2002/input14"
Output_path='Output_DUC2002/output14/system'

n=20

f=open(os.getcwd()+"/Input_NumberofSentences.txt")
summary_length=int(f.read())
print(summary_length)

files = os.listdir(main_folder_path)

body=''
#print(files)
for file1 in files:
	f=open(main_folder_path+"/"+file1)
	raw=f.read()
	
	lines = BeautifulSoup(raw).find_all('text')
	line=' '
	for itr in lines:
		line=line+itr.get_text()
	#print(line)
	
	body=body+line
	f.close()
	#print(line)
	#print('\n\n')


#print(body)
#print('\n\n')



#applying k-means model
#model = PreProcessor()
print('\n\n1st stage\n\n')
result = PreProcessor(body,summary_length)
full = ''.join(result)
#print(full)
f = open(Output_path+'/task1_englishSyssum1.txt','w')
print (full,file=f)
f.close()





#applying centroid word embedding model
print('\n\n2nd stage\n\n')
result2=summarize(full,summary_length)
#print('\n\n2nd stage\n\n')
#print(result2)
f = open(Output_path+'/task1_englishSyssum2.txt','w')
print (result2,file=f)
f.close()




###applying MMR model to remove redundancy
print('\n\n3rd stage\n\n')
result3=MMRsummarize(result2,summary_length)
#print(result3)
f = open(Output_path+'/task1_englishSyssum3.txt','w')
print (result3,file=f)
f.close()


#applying Sentenceordering model
print('\n\n 4th stage\n\n')
result4=sentencePositioning(result3)
f = open(Output_path+'/task1_englishSyssum4.txt','w')
print (result4,file=f)
f.close()


