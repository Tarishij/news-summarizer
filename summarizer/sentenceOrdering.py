# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:37:46 2020

@author: PRIYANKA JAIN
"""
from spacy.lang.en import English
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

def similarity(sent1,sent2):
  
# tokenization 
    X_list = word_tokenize(sent1)  
    Y_list = word_tokenize(sent2) 
  
# sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
  
# remove stop words from string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
  
# form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
  
# cosine formula  
    for i in range(len(rvector)): 
        c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    return cosine 
    
def coherence(doc):
    i=0
    sum=0
    while(i<len(doc)-1):
        sum=sum+similarity(doc[i],doc[i+1])
        i+=1
    
    sum=sum/len(doc)
    return sum

def sentencePositioning(input,language = English):
    
    nlp = language()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(input)
    sentences=[c.string.strip() for c in doc.sents]
    length=len(sentences)
    
    s=np.arange(1, length+1, 1)
    #index for s
    t=0
    start=0 #indexing from 0
    cohMax=0
    cohTmp=0
    dn=[sentences[start]]
    dTmp=[sentences[start]]
    l=len(dn)
    s[start]=0
    flag=0
    while(l<length):
        
        if(s[t]!=0):
            
            dTmp.append(sentences[t])
            cohTmp=coherence(dTmp)
        
            if(cohTmp>=cohMax):
                
                dn.append(sentences[t])
                cohMax=cohTmp
                s[t]=0
                flag=1
            else:
                dTmp.pop()
            
            t=t+1
            
            if(t==length):
                t=0
                if(flag==0):
                    break
                flag=0
            #check if all the elements of  s are 0
            if(not np.any(s)):
                break
                
        else:
            t=t+1
            if(t==length):
                t=0
                if(flag==0):
                    break
                flag=0
        l=len(dn)
    
    i=0
    
    while(i<length):
        if(s[i]!=0):
            dn.append(sentences[s[i]-1])
        i+=1
    
    return ''.join(dn)
    
