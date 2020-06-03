#----------------------------------------------------------------------------------
# Description:	Sentence class to store setences from the individual files in the
#				document cluster.
#----------------------------------------------------------------------------------

from nltk.corpus import stopwords

class sentence(object):

	#preprowords:stemmedwords
	def __init__(self, docName, preproWords, originalWords):
		self.docName = docName
		self.preproWords = preproWords
		self.wordFrequencies = self.sentenceWordFreq()
		self.originalWords = originalWords


	def getDocName(self):
		return self.docName
	
	
	
	def getPreProWords(self):
		return self.preproWords
	
	
	def getOriginalWords(self):
		return self.originalWords


	def getWordFreq(self):
		return self.wordFrequencies	
	

	def sentenceWordFreq(self):
		wordFreq = {}
		for word in self.preproWords:
			if word not in list(wordFreq.keys()):
				wordFreq[word] = 1
			else:
				# if word in stopwords.words('english'):
				# 	wordFreq[word] = 1
				# else:			
				wordFreq[word] = wordFreq[word] + 1
		return wordFreq
