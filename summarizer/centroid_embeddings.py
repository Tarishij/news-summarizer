# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:27:22 2020

@author: PRIYANKA JAIN
"""

import string
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import flashtext
from nltk.corpus import stopwords
from summarizer.bert_parent import BertParent
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text_nltk(text):
        stopwords_remove=True
        language='english'
        sentences = sent_tokenize(text)
        extra_stopwords = ["''", "``", "'s"]
        sentences_cleaned = []
        if stopwords_remove:
            stopword_remover = flashtext.KeywordProcessor()
            for stopword in stopwords.words(language):
                stopword_remover.add_keyword(stopword, '')
            stopword_remover = stopword_remover
            
        for sent in sentences:
            if stopwords_remove:
                stopword_remover.replace_keywords(sent)
            words = nltk_word_tokenize(sent, language)
            words = [w for w in words if w not in string.punctuation]
            words = [w for w in words if w not in extra_stopwords]
            words = [w.lower() for w in words]
            sentences_cleaned.append(" ".join(words))
        return sentences_cleaned
        
def sent_tokenize(text):
        language='english'
        length_limit=10
        sents = nltk_sent_tokenize(text, language)
        
        sents_filtered = []
        for s in sents:
            if s[-1] != ':' and len(s) > length_limit:
                sents_filtered.append(s)
        return sents_filtered
    
def get_topic_idf(sentences):
    topic_threshold=0.3
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences)
    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()
    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())

    feature_names = vectorizer.get_feature_names()

    relevant_vector_indices = np.where(centroid_vector > topic_threshold)[0]

    word_list = list(np.array(feature_names)[relevant_vector_indices])
    return word_list

# Sentence representation with sum of word vectors
def compose_vectors(words,centroid_words,centroid_words_vector):
    composed_vector = np.zeros(centroid_words_vector.shape, dtype="float32")
    count = 0
    for w in words:
        if w in centroid_words:
            composed_vector = composed_vector + centroid_words_vector[centroid_words.index(w)]
            count += 1
    if count != 0:
        composed_vector = np.divide(np.sum(composed_vector,axis=0), count)
    else:
        composed_vector = np.sum(composed_vector,axis=0)
    return composed_vector
    
def create_embedding_vector(centroid_words):
    model = BertParent('bert-large-uncased')
    hidden = model(centroid_words)
    return hidden
    
def summarize(text, summary_length=20, limit_type='word'):
    raw_sentences = sent_tokenize(text)
    clean_sentences = preprocess_text_nltk(text)

    centroid_words =get_topic_idf(clean_sentences)
    #print(summary_length*20)
    limit=summary_length*20+100
        ###get bert  embeddings for centroid worsd
    centroid_words_vector=create_embedding_vector(centroid_words)
    centroid_vector=np.divide(np.sum(centroid_words_vector,axis=0),len(centroid_words))

    sentences_scores=[]
    for i in range(len(clean_sentences)):
            words = clean_sentences[i].split()
            sentence_vector = compose_vectors(words,centroid_words,centroid_words_vector)
            
            cos_lib = cosine_similarity([sentence_vector],[centroid_vector]) 
           
            ##similarity score of sentence
            score=cos_lib[0][0]
            sentences_scores.append((i, raw_sentences[i], score))

    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
        
    count = 0
    sentences_summary = []
        
    for s in sentence_scores_sort:
        if count > limit:
            break
        sentences_summary.append(s)
        count += len(s[1].split())
        
    sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)
    summary = "\n".join([s[1] for s in sentences_summary])
   
    return summary

            

  
