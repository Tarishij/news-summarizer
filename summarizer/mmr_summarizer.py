import nltk
import os
import math
import string
import re
from summarizer.sentence import sentence
from nltk.corpus import stopwords



#.......Function for stemming sentences....each sentences is changed into coresponding stemming sentence
def stemming_Sentences(text_1):


    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    
    lines = sentence_token.tokenize(text_1.strip())    
    sentences = []
    porter = nltk.PorterStemmer()


    for line in lines:

        originalWords = line[:]
        line = line.strip().lower()
        sent = nltk.word_tokenize(line)

        stemmedSent = [porter.stem(word) for word in sent]        
        stemmedSent = [x for x in stemmedSent if x!='.'and x!='`'and x!=','and x!='?'and x!="'" 
            and x!='!' and x!='''"''' and x!="''" and x!="'s"]
        
        # list of sentence objects where each object contains original words as well as stemmed words
        if stemmedSent != []:
            sentences.append(sentence('file', stemmedSent, originalWords))                
    
    return sentences



# Function to find term frequency......frequency of each term
def TFs(sentences):
    
    tfs = {}  #  tfs dictonary which will store : [term : frequency]
    for sent in sentences:
        wordFreqs = sent.getWordFreq()
        
        for word in list(wordFreqs.keys()):
            # if word already present in the dictonary
            if tfs.get(word, 0) != 0:                
                        tfs[word] = tfs[word] + wordFreqs[word]
            # else if word is being added for the first time
            else:                
                tfs[word] = wordFreqs[word]    
    return tfs

#---------------------------------------------------------------------------------
# Description    : Function to find the inverse document frequencies of the words in
#                  the sentences present in the provided document cluster 
# Parameters    : sentences, sentences of the document cluster
# Return         : dictonary of word, inverse document frequency score
#---------------------------------------------------------------------------------


## .......To find inverse document frequency of each word    
def IDFs(sentences):
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}

    
    for sent in sentences:
        
        for word in sent.getPreProWords():

            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0)+ 1

    for word in words:
        n = words[word]
        try:
            idf = math.log10(float(N)/n)
        except ZeroDivisionError:
            idf = 0
        idfs[word] = idf
            
    return idfs

#....To find TF-IDF calue from term frequency and inverse document  ferquency
def TF_IDF(sentences):

    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}


    for word in tfs:
        tf_idfs=  tfs[word] * idfs[word]
        
        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval


#.......function to find sentence similarity between the summary and current sentence
def sentenceSim(sentence1, sentence2, IDF_w):
    numerator = 0
    denominator = 0    
    
    for word in sentence2.getPreProWords():        
        numerator+= sentence1.getWordFreq().get(word,0) * sentence2.getWordFreq().get(word,0) *  IDF_w.get(word,0) ** 2

    for word in sentence1.getPreProWords():
        denominator+= ( sentence1.getWordFreq().get(word,0) * IDF_w.get(word,0) ) ** 2

    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")    

#.......Function to generate a query by taking top words in it
def buildQuery(sentences, TF_IDF_w, n):
    scores = list(TF_IDF_w.keys())
    scores.sort(reverse=True)    
    
    i = 0
    j = 0
    queryWords = []

    while(i<n):
        words = TF_IDF_w[scores[j]]
        for word in words:
            queryWords.append(word)
            i=i+1
            if (i>n): 
                break
        j=j+1

    return sentence("query", queryWords, queryWords)


#....Find the best sentence which will be the first sentence in summary
def bestSentence(sentences, query, IDF):
    best_sentence = None
    maxVal = float("-inf")

    for sent in sentences:
        similarity = sentenceSim(sent, query, IDF)        

        if similarity > maxVal:
            best_sentence = sent
            maxVal = similarity
    sentences.remove(best_sentence)

    return best_sentence


#....Function to make summary by taking the sentences with highest MMR_Score
def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):    
    summary = [best_sentence]
    sum_len = len(best_sentence.getPreProWords())
    sentence_count=0
    MMRval={}
    while (sum_len < summary_length):    
        MMRval={}        

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)
      
        maxxer = max(MMRval, key=MMRval.get)
        summary.append(maxxer)
        sentences.remove(maxxer)
        sum_len += len(maxxer.getPreProWords())    
        sentence_count=sentence_count+1

    return summary


#......function to calculate MMR score 
def MMRScore(Si, query, Sj, lambta, IDF):    
    Sim1 = sentenceSim(Si, query, IDF)
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentenceSim(Si, sent, IDF)
        value.append(Sim2)

    r_expr = (1-lambta) * max(value)
    MMR_SCORE = l_expr - r_expr    

    return MMR_SCORE



#.....Main function
def MMRsummarize(doc2,summary_length=20):

      
        sentences =[]
        sentences=stemming_Sentences(doc2)
        #print(sentences)

       

        # calculate TF, IDF and TF-IDF scores

        IDF_w         = IDFs(sentences)
        TF_IDF_w     = TF_IDF(sentences)    

        # build query; set the number of words to include in our query
        query = buildQuery(sentences, TF_IDF_w, 10)
      
        # pick a sentence that best matches the query    
        best1sentence = bestSentence(sentences, query, IDF_w)        

        # build summary by adding more relevant sentences
        total_words_size=summary_length*20+20
        #print(total_words_size)
        summary = makeSummary(sentences, best1sentence, query, total_words_size, 0.6, IDF_w)
        
        #print(summary)
        
        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOriginalWords() + "\n"
        final_summary = final_summary[:-1]
        #print(final_summary)
        return final_summary
        
        
