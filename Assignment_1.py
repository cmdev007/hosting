#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import os,string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer


# In[2]:


class PreProcess:
    def __init__(self,Dir):
        #1. Read and combine the text between the <TEXT> tags by using Beautiful Soup from all the documents.
        self.Dir=Dir
        #self.data=self.fetchComb()
        #self.dataFiltered=self.remPunNum()
        #self.dataTokenSimple=self.dataTokenS()
        #self.dataTokenNLTK=self.dataTokenN()
        #self.lowCase()
        #self.dataStop=self.remStop()
        #self.dataStem=self.Stemming()
        #self.dataLem=self.Lemmatization()
        
        #return self.dataLem
    
    def fetchComb(self):
#         if self.Dir[-1]!="/":
#             self.Dir+='/'
        global data
        data=''
        #for i in os.listdir(self.Dir):
        with open(self.Dir) as buff:
            databuff=BeautifulSoup(buff,'xml')
            data+=databuff.TEXT.text
        return data
    
    #2. Remove punctuation and numerical values from your text.
    def remPunNum(self):
        self.fetchComb()
        global dataFiltered
        dataFiltered=''
        for i in data:
            if (i not in string.punctuation) and (i not in string.digits):
                dataFiltered+=i
        return dataFiltered
    
    #3. Perform tokenization on the text.
    def dataTokenS(self):
        self.remPunNum()
        #simple method
        global dataTokenSimple
        dataTokenSimple=dataFiltered.split()
        return dataTokenSimple
    
    def dataTokenN(self):
        self.remPunNum()
        #NLTK method
        global dataTokenNLTK
        dataTokenNLTK=word_tokenize(dataFiltered)
        return dataTokenNLTK
    
    #4. Convert the text to lowercase
    def lowCase(self):
        self.dataTokenN()
        global dataTokenNLTK
        for i in range(len(dataTokenNLTK)):
            dataTokenNLTK[i]=dataTokenNLTK[i].lower()
    
    #5. Remove stop-words from the text
    def remStop(self):
        self.lowCase()
        global dataStop
        dataStop=[]
        for i in dataTokenNLTK:
            if i not in stopwords.words('english'):
                dataStop.append(i)
        return dataStop
                
    #6. Perform Stemming (use Porter Stemmer) and Lemmatization (separately) and display the differences that
    #you observe while using the two approaches. (For this you have to use NLTK library).
    def Stemming(self):
        self.remStop()
        global dataStem
        dataStem=[]
        for i in dataStop:
            dataStem.append(PorterStemmer().stem(i))
        return dataStem
            
    def Lemmatization(self):
        self.remStop()
        global dataLem
        dataLem=[]
        for i in dataStop:
            dataLem.append(WordNetLemmatizer().lemmatize(i))
        return dataLem


# In[3]:


#output=PreProcess('mini_dataset').Lemmatization()

