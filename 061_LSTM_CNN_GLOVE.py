
# based on https://github.com/msahamed/yelp_comments_classification_nlp/blob/master/word_embeddings.ipynb



import plotly as py
import plotly.graph_objs as go
from plotly.offline import *


import matplotlib as plt

# NLTK
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# Keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation


import text_normalizer as tn # adapted text normalizer

###############################################################################
###############################################################################
## Load and clean data


###############################################################################
## Load and clean data: retain only essentia fields and add other relevant fields
XXX_All = pd.read_pickle(r"C:\Users_Folders\Cursuri_toate\YORK_MLcertificate\Course_02\Project/trimmed_cellphone.pkl")

XXX_All=XXX_All[['overall', 'verified', 'reviewTime', 'reviewerID', 'asin',
        'reviewText', 'summary', 'vote','category','title','brand','main_cat']]

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




###############################################################################
### Load and clean data: remove tablets
temp_loc=XXX_All["main_cat"]!="Computers"
XXX_All=XXX_All[temp_loc]

### remove ipods
temp_loc=XXX_All["main_cat"]!="Apple Products"
zzz=XXX_All[temp_loc]

### check what is in the "All Electronics" category
temp_loc=XXX_All["main_cat"]=="All Electronics"
zzz=XXX_All[temp_loc]



###############################################################################
## Load and clean data:  group into one category the periferic brands
def category_brands(x):
    if x in ['Samsung','BLU','Apple',"LG","Motorola",'Huawei','BlackBerry','HTC','Nokia','Asus','Sony','ZTE','Tracfone']:
        return x
    else:
        return 'Other'
XXX_All['main_brands'] = XXX_All['brand'].apply(category_brands)



###############################################################################
# colapse all star ratings <= 4 into one category and =5 into another category
XXX_All['overall_2_categories'] = XXX_All['overall'].map(lambda x : 0 if int(x) < 5 else 1)


###############################################################################
# Removing certain html addresses
XXX_All["reviewText"] = XXX_All["reviewText"].str.replace(r'https://www.amazon.com/dp/B00CIF9MJK/ref=cm_cr_ryp_prd_ttl_sol_23', '')
XXX_All["reviewText"] = XXX_All["reviewText"].str.replace(r'https://images-na.ssl-images-amazon.com/images/I/315VpPewBYL._SS300_.jpg', '')



###############################################################################
### count the review length
def review_length(x):
    if isinstance(x, str):
        return len(x)
    else:
        return -1
XXX_All['reviewText_length'] = XXX_All['reviewText'].apply(review_length)


###############################################################################
## Load and clean data: Select subset of review for code testing

temp_loc=(XXX_All["reviewText_length"]>20) & (XXX_All["reviewText_length"]<500)&(XXX_All["year"]>=2017) 
xxx_test=XXX_All[["overall_2_categories","reviewText"]][temp_loc]
xxx_test.drop_duplicates



##############################################################################
## Load and clean data: process word corrections  and other text normalization steps

reviewText_clean=tn.normalize_corpus(np.array(xxx_test['reviewText']),html_stripping=False, 
                     contraction_expansion=True,apply_correct_words=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=False, special_char_removal=True, 
                     stopword_removal=False)
df=pd.DataFrame(reviewText_clean,columns =['text'])
labels=xxx_test['overall_2_categories'] 
del(XXX_All)

###############################################################################
## Load and clean data: Tokenize text data

print("\n\n#######################################\nTokenize text data")
def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text
df['text'] = df['text'].map(lambda x: clean_text(x))

print("\ndf.head()")
print(df.head())



INPUT_vocab_size = 2000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=INPUT_vocab_size)
tokenizer.fit_on_texts(df['text'])
        
tokenizer = Tokenizer(num_words= INPUT_vocab_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)
print('\ndata.shape',data.shape)





######################################
## Build neural network with LSTM
######################################
print("\n\n#######################################\nBuild neural network with LSTM\n")


## Network Architechture

model_lstm = tf.keras.Sequential()
model_lstm.add(tf.keras.layers.Embedding(INPUT_vocab_size, 100, input_length=50))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Train the network
print('\nLSTM\n')
model_lstm.fit(data, np.array(labels), validation_split=0.4, epochs=3)



##  neural network with LSTM and CNN

def create_conv_model():
    model_conv = Sequential()
    model_conv.add(tf.keras.layers.Embedding(INPUT_vocab_size, 100, input_length=50))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv 
model_conv = create_conv_model()
print('\n\nLSTM and CNN\n')
model_conv.fit(data, np.array(labels), validation_split=0.4, epochs = 3)
print('\nAdding the CNN layer does not lead to significant improvements in accuracy.\n')


##  Save processed Data
df_save = pd.DataFrame(data)
df_label = pd.DataFrame(np.array(labels))
result = pd.concat([df_save, df_label], axis = 1)
result.to_csv('train_dense_word_vectors.csv', index=False)





######################################
## Use pre-trained Glove word embeddings
######################################
print("\n\n#######################################\nUse pre-trained Glove word embeddings\n")

## Get embeddings from Glove
# https://www.damienpontifex.com/posts/using-pre-trained-glove-embeddings-in-tensorflow/
# had to manually remove a row  and transform data into csv

# re-wrote code for reading pretrained glove.6B.100d.txt

GLOVE_results = pd.read_csv('glove_6B_100d.csv')
embeddings_index = dict()

for index, row in GLOVE_results.iterrows():
    word = row[0]
    coefs = np.asarray(row[1:], dtype='float32')
    embeddings_index[word] = coefs

print('Loaded %s word vectors from glove_6B_100d mapping.\n' % len(embeddings_index))


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((INPUT_vocab_size, 100))
for word, index in tokenizer.word_index.items():
    if index > INPUT_vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


## Develop model- same model architecture with a convolutional layer on top of the LSTM layer.

model_glove = Sequential()
model_glove.add(tf.keras.layers.Embedding(INPUT_vocab_size, 100, input_length=50, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_glove.fit(data, np.array(labels), validation_split=0.4, epochs = 3)

print('\nUsing Glove embedding does not lead to significant improvements in accuracy .\n')


