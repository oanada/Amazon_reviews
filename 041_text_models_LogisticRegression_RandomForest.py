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
temp_loc=(XXX_All["reviewText_length"]>50) & (XXX_All["reviewText_length"]<1500) &(XXX_All["year"]>=2017) #& (XXX_All["overall"]<=3)
xxx_test=XXX_All[["overall","reviewText","reviewText_length"]][temp_loc]
xxx_test.drop_duplicates















################################################################################################
## reviewText column - Text pre-processing with NLTK
#  https://newscatcherapi.com/blog/spacy-vs-nltk-text-normalization-comparison-with-code-examples
################################################################################################

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
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
stop_words_list= set.union(stop_words_list,['phone','iphone','mobile','product','feature',"samsung","galaxy"]) 




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



 
##########################################################################
#Training a logistic regression model to extract features linked to the overall score
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
import sklearn


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
import seaborn as sns

# instantiate the model (using the default parameters)
logreg = LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.5,max_iter=300,class_weight="balanced")

# fit the model with data
logreg.fit(OUT_vocab, corpus_df["overall"])

#
y_pred=logreg.predict(OUT_vocab)
y_pred_proba = logreg.predict_proba(OUT_vocab)[::,1]

cnf_matrix = metrics.confusion_matrix(corpus_df["overall"].to_numpy(), y_pred)
sns.heatmap(cnf_matrix,cmap="YlGnBu",)

print("Precision:",metrics.precision_score(corpus_df["overall"].to_numpy(), y_pred,average=None))
print("Recall:",metrics.recall_score(corpus_df["overall"].to_numpy(), y_pred,average=None))

    
Logreg_coeffs=np.absolute(logreg.coef_)
#Logreg_coeffs = pd.DataFrame(Logreg_coeffs[0:2,:], columns = OUT_vocab.columns)
Logreg_coeffs = pd.DataFrame(Logreg_coeffs[3:4,:], columns = OUT_vocab.columns)
Logreg_coeffs=Logreg_coeffs.sum().sort_values(ascending=False)


ordered_df=Logreg_coeffs[0:30]
#ordered_df=ordered_df[31:60]
## Barplot with confidence intervals
height = ordered_df
bars = ordered_df.index
y_pos = np.arange(len(bars))
# Create horizontal bars
fig, ax = plt.subplots(figsize=(5,10))
plt.barh(y_pos, height)
# Create names on the y-axis
plt.yticks(y_pos, bars)
plt.xlabel("Abs sum of coefficients across the best 2 rating categories - top 30 ")
#plt.xlabel("Abs sum of coefficients across the worst 3 rating categories - top 30 ")
# Show graphic
#plt.xlabel("Mean reduction in tree impurity in random forest top 31-60 importance")
plt.show()



    
##########################################################################
#Training a random forest model to extract features linked to the overall score

from sklearn.ensemble import RandomForestClassifier




RANDOM_STATE = 123
#Train random forest classification model


model = RandomForestClassifier(max_depth=30, random_state=RANDOM_STATE)
model.fit(OUT_vocab, corpus_df["overall"])
# Diagnosis prediction
y_predict = model.predict(OUT_vocab)
# Probability of malignant tissue produced by the model
y_prob = [probs[1] for probs in model.predict_proba(OUT_vocab)]

    
##########################################################################
#Evaluate model
y_test=corpus_df["overall"].to_numpy()

#Accuracy on test set
print(f"Test accuracy: {accuracy_score(y_test, y_predict).round(2)}")
# Confusion matrix test set

confusion_matrix(y_test, y_predict)
sns.heatmap(confusion_matrix(y_test, y_predict),cmap="YlGnBu")

print("Precision:",metrics.precision_score(corpus_df["overall"].to_numpy(), y_predict,average=None))
print("Recall:",metrics.recall_score(corpus_df["overall"].to_numpy(), y_predict,average=None))
############################################
# Model-specific feature importance
# Feature importance dataframe
imp_df = pd.DataFrame({'feature': OUT_vocab.columns.values,
'importance': model.feature_importances_})
# Reorder by importance
ordered_df = imp_df.sort_values(by='importance',ascending=False)
ordered_df=ordered_df[0:30]
#ordered_df=ordered_df[31:60]
## Barplot with confidence intervals
height = ordered_df['importance']
bars = ordered_df['feature']
y_pos = np.arange(len(bars))
# Create horizontal bars
fig, ax = plt.subplots(figsize=(5,10))
plt.barh(y_pos, height)
# Create names on the y-axis
plt.yticks(y_pos, bars)
plt.xlabel("Mean reduction in tree impurity in random forest top 30 importance")
# Show graphic
#plt.xlabel("Mean reduction in tree impurity in random forest top 31-60 importance")
plt.show()




