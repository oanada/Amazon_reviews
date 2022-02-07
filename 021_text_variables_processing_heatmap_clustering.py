# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:24:25 2021

@author: oanad
"""

import pandas as pd
import math
import re


XXX_All = pd.read_pickle(r"C:\Users_Folders\Cursuri_toate\YORK_MLcertificate\Course_02\Project/trimmed_cellphone.pkl")


################################################################################################
## retain only essential fields and add other relevant fields
XXX_All=XXX_All[['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style',
        'reviewText', 'summary', 'vote','category',  'description', 'title','brand', 'feature',  'main_cat','price']]

#XXX_All['years_2_today']=(XXX_All['reviewTime']-pd.to_datetime("now"))/pd.TimeDelta('1D')
XXX_All["reviewTime"]=pd.to_datetime(XXX_All["reviewTime"])
XXX_All["year"]=XXX_All["reviewTime"].dt.year
XXX_All["vote"]=XXX_All["vote"].fillna('0')
def replace_comma(x):
        return int(x.replace(",", ""),10)
def vote_capped_replace_comma(x):
        return min(x,10)
XXX_All["vote"]=XXX_All["vote"].apply(replace_comma)
XXX_All["vote_capped_10"]=XXX_All["vote"].apply(vote_capped_replace_comma)




################################################################################################
### remove tablets
temp_loc=XXX_All["main_cat"]!="Computers"
XXX_All=XXX_All[temp_loc]

### remove ipods
temp_loc=XXX_All["main_cat"]!="Apple Products"
zzz=XXX_All[temp_loc]




################################################################################################
### group into one category the periferic brands
def category_brands(x):
    if x in ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone']:
        return x
    else:
        return 'Other'
XXX_All['main_brands'] = XXX_All['brand'].apply(category_brands)


  ### group into one category non cell phones 
def category_groups(x):
    if x =='''['Cell Phones & Accessories', 'Cell Phones', 'Carrier Cell Phones']''':
        return 'Carrier Cell Phones'
    elif x=='''['Cell Phones & Accessories', 'Cell Phones', 'Unlocked Cell Phones']''':
        return 'Unlocked Cell Phones'
    elif x=='''['Cell Phones & Accessories', 'Cell Phones']''':
        return 'Cell Phones'
    else:
        return 'Other'    
XXX_All['category'] = XXX_All['category'].apply(category_groups)

################################################################################################
### count the review length
def review_length(x):
    if isinstance(x, str):
        return len(x)
    else:
        return -1
XXX_All['reviewText_length'] = XXX_All['reviewText'].apply(review_length)
     
def review_length_bins(x):
    if isinstance(x, str):
        return 1+math.floor(min(3000,len(x))/10)*10
    else:
        return -1    

XXX_All['reviewText_length_bins'] = XXX_All['reviewText'].apply(review_length_bins)


XXX_All.sort_values(['overall', 'reviewText_length'], ascending=[True, True], inplace=True)



################################################################################################
# Select subset of review for code testing
temp_loc=(XXX_All["reviewText_length"]>50) & (XXX_All["reviewText_length"]<1500) &(XXX_All["year"]>=2018) #& (XXX_All["overall"]<=3)
xxx_test=XXX_All[["overall","reviewText","reviewText_length"]][temp_loc]
xxx_test.drop_duplicates











################################################################################################
## reviewText column - Text pre-processing with NLTK
#  https://newscatcherapi.com/blog/spacy-vs-nltk-text-normalization-comparison-with-code-examples


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
stemmer = PorterStemmer()


import spacy


#-------------------------------------------
# create a joint stop words list nltk & spacy
# --------------------------------------------

stop_words_list_spacy= spacy.load('en_core_web_sm').Defaults.stop_words
stop_words_list_nltk= set(stopwords.words("english"))
stop_words_list= set.union(stop_words_list_spacy,stop_words_list_nltk)


# add some dataset specific stopwords
stop_words_list= set.union(stop_words_list,['phone','iphone','mobile','product','feature']) 




# ---------------------------------------
#  Tokenization
# ---------------------------------------
def nltk_tokenize(x):    
    try: return word_tokenize(x.lower())        
    except ValueError: return([])
#xxx_test["reviewText_tokn"]=xxx_test['reviewText'].apply(nltk_tokenize)
xxx_test["reviewText_normalized"]=xxx_test['reviewText'].apply(nltk_tokenize)
xxx_test.sort_values(by=['reviewText'])




# ---------------------------------------
# Removing the stopwords
# ---------------------------------------
def nltk_spacy_remove_stopwords(x,stop_words_list): 
    normalized_tokens = []  
    for word in x:  
        if word not in stop_words_list:  normalized_tokens.append(word)
    return normalized_tokens
#xxx_test["reviewText_tokn_stem_lem_stopwrds"]=xxx_test['reviewText_tokn_stem_lem'].apply(nltk_remove_stopwords)    
xxx_test["reviewText_normalized"]=xxx_test['reviewText_normalized'].apply(nltk_spacy_remove_stopwords,args=([stop_words_list]))   




#--------------------- ------------------
# Lemmatization
# ---------------------------------------
def nltk_lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    nltk_lemma_list = []
    for word in x:
        nltk_lemma_list.append(lemmatizer.lemmatize(word))
    return nltk_lemma_list

#xxx_test["reviewText_tokn_stem_lem"]=xxx_test['reviewText_tokn_stem'].apply(nltk_lemmatize)
xxx_test["reviewText_normalized"]=xxx_test['reviewText_normalized'].apply(nltk_lemmatize)

#add another step for removing stop words
xxx_test["reviewText_normalized"]=xxx_test['reviewText_normalized'].apply(nltk_spacy_remove_stopwords,args=([stop_words_list]))   





# ---------------------------------------
# Removing punctuations
# ---------------------------------------
def remove_punctuations(x):
    punctuations=['?',':','!',',','.','*','!!','* *',';','|','(',')','--','..',"&","-","''","""""",'``',"$",">","<",'.....','[', '...', ']',"u"]
    for word in x:
        if word in punctuations:
            x.remove(word)
    return x
#xxx_test["reviewText_tokn_stem_lem_stopwrds_punct"]=xxx_test['reviewText_tokn_stem_lem_stopwrds'].apply(remove_punctuations)    
xxx_test["reviewText_normalized"]=xxx_test['reviewText_normalized'].apply(remove_punctuations)



# ---------------------------------------
# Removing all numbers tokens
# ---------------------------------------
def remove_all_number_words(x):
    set_words_to_keep = []
    for word in x:
        if len(word)!=sum(c.isdigit() for c in word):
            set_words_to_keep.append(word)
    return set_words_to_keep
#xxx_test["reviewText_tokn_stem_lem_stopwrds_punct"]=xxx_test['reviewText_tokn_stem_lem_stopwrds'].apply(remove_punctuations)    
xxx_test["reviewText_normalized"]=xxx_test['reviewText_normalized'].apply(remove_all_number_words)



# ---------------------------------------
#  Stemming
# ---------------------------------------
def nltk_PorterStemmer(x):
    stemed_tokens = []
    for word in x:  stemed_tokens.append(stemmer.stem(word))
    return stemed_tokens

#xxx_test["reviewText_tokn_stem"]=xxx_test['reviewText_tokn'].apply(nltk_PorterStemmer)
xxx_test["reviewText_normalized"]=xxx_test['reviewText_normalized'].apply(nltk_PorterStemmer)









################################################################################################
# Online Session 2 - Coding Exercise
# (The following links will open in new window)

# Understanding Feature Engineering (Part 3) - Traditional Methods for Text Data [18 minutes]

# You do not need to submit this code again, since it was covered in the first class. For your reference, code is available in Github here.

# This article with Jupyter notebook presents the traditional features engineering techniques in text data. After presenting
# text pre-processing, it will cover the more traditional methods such as bag-of-word or n-grams, and tf-idf. It will also cover 
# the powerful method of document similarity and document clustering using topic models.

# https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/feature%20engineering%20text%20data/Feature%20Engineering%20Text%20Data%20-%20Traditional%20Strategies.ipynb
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt


def join_processed_text(x):
    return " ".join(x)
xxx_test["reviewText_normalized_string"]=xxx_test["reviewText_normalized"].apply(join_processed_text)

 # corpus needs to be an array
corpus = np.array(xxx_test["reviewText_normalized_string"])
corpus_df = xxx_test[['reviewText_normalized_string', 'overall']]
corpus_df.rename(columns={"reviewText_normalized_string": "Document", "overall": "Category"})
corpus_df.reset_index()

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(corpus)


# ---------------------------------------
## Bag of Words Model
# ---------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix = cv_matrix.toarray()





# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
OUT_vocab=pd.DataFrame(cv_matrix, columns=vocab)


# ---------------------------------------
# Bag of N-Grams Model
# ---------------------------------------
# you can set the n-gram range to 1,2 to get unigrams as well as bigrams
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_corpus)

bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
OUT_bigrams=pd.DataFrame(bv_matrix, columns=vocab)
OUT_bigrams_vocab=vocab



# ---------------------------------------
# TF-IDF Model for teh dictionmary
# ---------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
OUT_TDIDF=pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)



# ----------------------------------------
#Document Similarity
# ----------------------------------------
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap=sns.color_palette("rocket_r", as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(similarity_matrix,cmap=cmap)

# # Draw the full plot




# ----------------------------------------------
# Clustering documents using similarity features
# ----------------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(similarity_matrix, 'ward')
OUT_clustering_similarity_matrix=pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2', 
                         'Distance', 'Cluster Size'], dtype='object')

plt.figure(figsize=(20, 4))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=1.0, c='k', ls='--', lw=0.5)

from scipy.cluster.hierarchy import fcluster
max_dist = 7

cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
OUT_clusters=pd.concat([corpus_df, cluster_labels], axis=1)


def colors_conversion(x): 
    if x==1: return 'r'
    elif x==2: return 'm'
    elif x==3: return 'y'
    elif x==4: return 'b'
    elif x==5: return 'g'
    else: return 'w'
    xxx=corpus_df["overall"].apply(colors_conversion)
g = sns.clustermap(similarity_matrix,cmap="rocket_r",method='ward',dendrogram_ratio=(.1, .2),\
                   col_colors=corpus_df["overall"].apply(colors_conversion).to_numpy(),\
                   row_colors=corpus_df["overall"].apply(colors_conversion).to_numpy())
g.ax_row_dendrogram.remove()



# for future:   https://seaborn.pydata.org/examples/structured_heatmap.html



# # ---------------------------------------
# # Topic Models for vocabulary
# # ---------------------------------------
# from sklearn.decomposition import LatentDirichletAllocation

# lda = LatentDirichletAllocation(n_components=3, max_iter=5000, random_state=0)
# dt_matrix = lda.fit_transform(cv_matrix)
# features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
# features

# # Show topics and their weights
# tt_matrix = lda.components_
# for topic_weights in tt_matrix:
#     topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
#     topic = sorted(topic, key=lambda x: -x[1])
#     topic = [item for item in topic if item[1] > 12]
#     print(topic)
#     print() 
#     print()
    
    
    
# # ---------------------------------------    
# #Clustering documents using topic model features , 1-gram
# # ---------------------------------------


# from sklearn.cluster import KMeans

# km = KMeans(n_clusters=5, random_state=0)
# km.fit_transform(features)
# cluster_labels = km.labels_
# cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
# corpus_df['cluster_labels']=cluster_labels['ClusterLabel']
  
    
    
    
    
# # ---------------------------------------
# # Topic Models for bigrams
# # ---------------------------------------
# from sklearn.decomposition import LatentDirichletAllocation

# lda_bigrams = LatentDirichletAllocation(n_components=3, max_iter=5000, random_state=0)
# dt_matrix = lda_bigrams.fit_transform(bv_matrix)
# features_bigrams = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
# features_bigrams

# # Show topics and their weights
# tt_matrix = lda_bigrams.components_
# for topic_weights in tt_matrix:
#     topic = [(token, weight) for token, weight in zip(OUT_bigrams_vocab, topic_weights)]
#     topic = sorted(topic, key=lambda x: -x[1])
#     topic = [item for item in topic if item[1] > 2.5]
#     print(topic)
#     print() 
#     print()
        
    
#  # ---------------------------------------    
#  #Clustering documents using topic model features ,bigram
#  # ---------------------------------------  
    
# from sklearn.cluster import KMeans

# km = KMeans(n_clusters=5, random_state=0)
# km.fit_transform(features_bigrams)
# cluster_labels_bigrams = km.labels_
# cluster_labels_bigrams = pd.DataFrame(cluster_labels_bigrams, columns=['ClusterLabel'])
# corpus_df[['cluster_labels_bigrams']]=cluster_labels_bigrams
 




    
# ####################################
# #Training a random forest  model

# RANDOM_STATE = 123
# #Train random forest classification model
# model = RandomForestClassifier(max_depth=4, random_state=RANDOM_STATE)
# model.fit(OUT_vocab, y_train)
# # Diagnosis prediction
# y_predict = model.predict(X_test)
# # Probability of malignant tissue produced by the model
# y_prob = [probs[1] for probs in model.predict_proba(X_test)]
    

# #########################################################
# #Evaluate model

# #Accuracy on test set
# print(f"Test accuracy: {accuracy_score(y_test, y_predict).round(2)}")
# # Confusion matrix test set
# pd.DataFrame(
# confusion_matrix(y_test, y_predict),
# columns=['Predicted Benign', 'Predicted Malignant'],index=['Benign', 'Malignant']
# )
# # Compute area under the curve
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# #Set default figure size
# plt.rcParams['figure.figsize'] = (8,8)
# # Plot ROC curve
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
# lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title("Diagnosing Breast Cancer")
# plt.legend(loc="lower right")
# plt.show()

# ############################################
# # Model-specific feature importance
# # Feature importance dataframe
# imp_df = pd.DataFrame({'feature': X_train.columns.values,
# 'importance': model.feature_importances_})
# # Reorder by importance
# ordered_df = imp_df.sort_values(by='importance')
# imp_range=range(1,len(imp_df.index)+1)
# ## Barplot with confidence intervals
# height = ordered_df['importance']
# bars = ordered_df['feature']
# y_pos = np.arange(len(bars))
# # Create horizontal bars
# plt.barh(y_pos, height)
# # Create names on the y-axis
# plt.yticks(y_pos, bars)
# plt.xlabel("Mean reduction in tree impurity in random forest")
# plt.tight_layout()
# # Show graphic
# plt.show()