# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score
import time
from pymongo import MongoClient



# In[2]:


starttime=time.time()



# In[2]:
#
# import os
# cwd = os.getcwd()
# print(cwd)
# price = pd.read_csv("../ml-model/price_data.csv",encoding = "ISO-8859-1")
# news = pd.read_csv("../ml-model/news_data.csv",encoding = "ISO-8859-1")


# In[3]:
def convertDataTypeForPrice(p):
    return {
        'exchange': p['exchange'],
        'id': p['id'],
        'price': float(p['price']),
        'timestamp': int(p['timestamp'])
           }

def convertDataTypeForNews(n):
    return {
        'title': n['title'],
        'time': int(n['time']),
        'text': n['text'],
        'weight': int(n['weight']),
        'source': n['source']
    }

def mongoToPd(db, collectionName, convertFunc, query=None):
    collection = db[collectionName]
    dataList = list(collection.find({}))
    return pd.DataFrame(list(map(convertFunc, dataList)))

client = MongoClient('localhost', 27017)
db = client['cs5412']
# news_data_db = db['news_data']
# price_data_db = db['price_data']
#
# news = pd.DataFrame(list(news_data_db.find({}, {'_id': 0, 'title': 1, 'time': 1, 'text': 1, 'weight': 1, 'source': 1})))
# price = pd.DataFrame(list(price_data_db.find({}, {'_id': 0, 'id': 1, 'price': 1, 'timestamp': 1, 'exchange': 1})))

news = mongoToPd(db, 'news_data', convertDataTypeForNews)
price = mongoToPd(db, 'price_data', convertDataTypeForPrice)


# In[4]:


price = price.drop(['id','exchange'], 1)
price=price.sort_values(by=['timestamp'],ascending=True)
price_now = price['price'][price.last_valid_index()]
timestamp_now = price['timestamp'][price.last_valid_index()]

# In[5]:


price=price[-81:]


# In[6]:


price


# In[7]:


price = price.set_index('timestamp').diff(periods=1)
price['timestamp'] = price.index
price = price.reset_index(drop=True)


# In[8]:


price = price[price.price.notnull()]
price.shape


# In[9]:


price_label = []
headlines = []
for row in price.iterrows():
    daily_headlines = []
    timestamp = row[1]['timestamp']
    price_label.append(row[1]['price'])
    i=0
    for row_ in news[news.time<=timestamp].iterrows() and news[news.time>(timestamp-80)].iterrows():
        #print (row_[1]['weight'])
        if(i<30):
            i=i+1
            daily_headlines.append(row_[1]['text'])
        else:
            continue

    justb = np.array(daily_headlines)
    #print (justb.shape)
    headlines.append(justb)
       #daily_headlines.append(row_[1]['text'])
       # justb = np.array(daily_headlines)
       # print (justb.shape)


# In[10]:


headlines = np.array(headlines)


# In[11]:


contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


# In[12]:


def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'0,0', '00', text)
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# In[13]:


import re
from nltk.corpus import stopwords
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)


# In[14]:


clean_headlines = np.array(clean_headlines)


# In[15]:


def add_avg_sentiment_list(data):
    sid = SentimentIntensityAnalyzer()
    avgs = []
    for i in range(0,len(data)):
        inner_avg=[]
        for j in range(0,len(data[0])):
            sentiments=(sid.polarity_scores(str(data[i][j]))['compound'])
            inner_avg.append(float(sentiments))
        avgs.append(inner_avg)
    avgs = np.array(avgs)
    print(avgs.shape)
    return avgs


# In[16]:


sentiment_included = add_avg_sentiment_list(clean_headlines)


# In[17]:


import numpy
mylist_label = np.asarray((price_label))
mylist_label=numpy.reshape(mylist_label, (len(mylist_label),1))


# In[18]:


sum_set_timestep=sentiment_included.sum(axis=1)
sum_set_timestep=numpy.reshape(sum_set_timestep,(len(sum_set_timestep),1))


# In[19]:


sum_set_timestep=numpy.reshape(sum_set_timestep,(1,len(sum_set_timestep)))


# In[20]:


from keras.models import model_from_json


# In[21]:


json_file = open('stock_lstm_noscale_5min.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("stock_lstm_noscale_5min.h5")
print("Loaded model from disk")


# In[22]:


loaded_model.compile(loss='mean_squared_error', optimizer='adam')


# In[23]:


lastone = loaded_model.predict(numpy.reshape(sum_set_timestep, (len(sum_set_timestep), 80, 1)), verbose = 0)


# In[24]:


import matplotlib.pyplot as plt
import math
plt.figure(figsize=(12,4))
endtime=time.time()
print (endtime-starttime)
plt.plot(numpy.reshape(lastone, lastone[0].shape)[math.ceil(endtime-starttime):]*10,'r')
plt.title("Predicted (red) ")
plt.xlabel("Time")
plt.ylabel("Predicted Price Difference")
#plt.axis([endtime-starttime,80, 0, 10])
#plt.show()
plt.savefig('predictions.png')


# In[26]:
price_predicted = str(price_now + lastone[0][-1][0])
timestamp_next = str(timestamp_now + 60)
pred = {'price_predicted': price_predicted, 'timestamp': timestamp_next}
result = db['prediction_data'].insert_one(pred)

a=numpy.reshape(lastone, lastone[0].shape)
a
