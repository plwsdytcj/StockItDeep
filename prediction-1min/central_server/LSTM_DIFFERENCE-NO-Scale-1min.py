
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import svm
from sklearn.model_selection import cross_val_score


# In[2]:


price = pd.read_csv("price_data.csv",encoding = "ISO-8859-1")
news = pd.read_csv("news_data.csv",encoding = "ISO-8859-1")


# In[3]:


price = price.drop(['id','exchange'], 1)
price=price.sort_values(by=['timestamp'],ascending=True)
price.head()


# In[4]:


price_diff=price
price_diff.head()


# In[5]:


price = price.set_index('timestamp').diff(periods=1)
price['timestamp'] = price.index
price = price.reset_index(drop=True)
price.head()


# In[6]:


price = price[price.price.notnull()]
price[0:100]


# In[7]:


price.shape


# In[8]:


price_label = []
headlines = []
for row in price.iterrows():
    daily_headlines = []
    timestamp = row[1]['timestamp']
    price_label.append(row[1]['price'])
    i=0
    for row_ in news[news.time<=timestamp].iterrows() and news[news.time>(timestamp-80)].iterrows():
        #print (row_[1]['weight'])
        if(i<80):
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


# In[9]:


headlines = np.array(headlines)
headlines.shape


# In[10]:


price_label


# In[11]:


print(headlines[0])


# In[12]:


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


# In[13]:


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


# In[14]:


import re
from nltk.corpus import stopwords
clean_headlines = []

for daily_headlines in headlines:
    clean_daily_headlines = []
    for headline in daily_headlines:
        clean_daily_headlines.append(clean_text(headline))
    clean_headlines.append(clean_daily_headlines)


# In[15]:


clean_headlines = np.array(clean_headlines)
clean_headlines.shape


# In[16]:


clean_headlines[0]


# In[17]:


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


# In[18]:


sentiment_included = add_avg_sentiment_list(clean_headlines)
print("sentiment_include")
print(sentiment_included.shape)


# In[19]:


print(sentiment_included[:10])


# In[20]:


print(price_label[0:10])


# In[21]:


import numpy
mylist_label = np.asarray((price_label))
print(mylist_label[0:10])
print(mylist_label.shape)
mylist_label=numpy.reshape(mylist_label, (len(mylist_label),1))
mylist_label[0:10]


# In[22]:


def divide_train_test(data):
	train = data[:4500]
	test = data[4500:]
	return train, test


# In[23]:


train_set, test_set = divide_train_test(sentiment_included)


# In[24]:


train_set.shape


# In[25]:


def divide_train_label(data):
	train = data[:4500]
	test = data[4500:]
	return train, test


# In[26]:


train_label, test_label = divide_train_test(mylist_label)


# In[27]:


print("train_labels")
print(train_label.shape)
print(train_label[0:30])


# In[28]:


sum_set_timestep=train_set.sum(axis=1)
sum_set_timestep=numpy.reshape(sum_set_timestep,(len(sum_set_timestep),1))
sum_set_timestep


# In[29]:


train_label_timestep=train_label
train_label_timestep


# In[30]:


seq_length = 80
dataX = []
dataY = []
for i in range(0, len(sum_set_timestep) - seq_length-seq_length, 1):
    seq_in=[]
    seq_in = sum_set_timestep[i:i + seq_length]
    seq_in=numpy.reshape(seq_in,(len(seq_in),))
    dataX.append(seq_in)
print (len(dataX))
print (dataX[300])

for i in range(seq_length, len(train_label_timestep) - seq_length, 1):
    seq_out = train_label_timestep[i:i + seq_length]
    seq_out=numpy.reshape(seq_out,(len(seq_out),))
    dataY.append(seq_out)
print (len(dataY))
print (dataY[300])


# In[31]:


dataX=numpy.reshape(dataX,(len(dataX),80))
dataY=numpy.reshape(dataY,(len(dataY),80))
print (dataX.shape)
print (dataY.shape)


# In[32]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[33]:


from keras.layers import TimeDistributed
def lstm_train_timestep(data, labels):
    print(data.shape)
    print(labels.shape)
    X = numpy.reshape(data, (len(data), 80, 1))
    Y = numpy.reshape(labels, (len(labels), 80, 1))
# normalize
    #X = X / float(len(alphabet))
# one hot encode the output variable
    #y = np_utils.to_categorical(dataY)
# create and fit the model
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X,Y, epochs=30, batch_size=5, verbose=2)
    return model


# In[34]:


lstm_clsfr_timestep = lstm_train_timestep(dataX, dataY)


# In[35]:


lstm_predictions_timestep = lstm_clsfr_timestep.predict(numpy.reshape(dataX, (len(dataX), 80, 1)), verbose = 0) 


# In[36]:


lastone = lstm_clsfr_timestep.predict(numpy.reshape(dataX[-1:], (len(dataX[-1:]), 80, 1)), verbose = 0)
lastone


# In[37]:


lastone=numpy.reshape(lastone, lastone[0].shape)
lastone.shape


# In[41]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.plot(lastone,'r')
plt.title("Predicted (red) vs Actual (green) Opening Price Changes")
plt.xlabel("Testing instances")
plt.ylabel("Change in Opening Price")
plt.show()


# In[43]:


stock_json = lstm_clsfr_timestep.to_json()
with open("stock_lstm_noscale_5min.json", "w") as json_file:
    json_file.write(stock_json)
# serialize weights to HDF5
lstm_clsfr_timestep.save_weights("stock_lstm_noscale_5min.h5")

