from bs4 import BeautifulSoup
import os,string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
import numpy as np
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from Assignment_1 import PreProcess

class DocRepresent(PreProcess):
    def __init__(self,Dir):
        #1. Read and combine the text between the <TEXT> tags by using Beautiful Soup from all the documents.
        self.Dir=Dir
        #self.corpus=self.corpusCreater()
        
    def manualTFIDF(self):
        global c
        global uniIdent
        c=1
        indexer=os.listdir(self.Dir)
        indexer.sort()
        for i in indexer:
            StemData=PreProcess(self.Dir+i).Lemmatization()
            globals()['data%s' %c] = StemData
            c+=1
        
        #iterating through all the documents and finding unique words
        uniIdent=[]
        for i in range(1,c):
            buff=globals()['data%s' %i]
            for j in buff:
                if j not in uniIdent:
                    uniIdent.append(j)
        uniIdent.sort()
            
        #Creating TF matrix
        mat1=np.zeros((c-1,len(uniIdent)))
        for i in range(1,c):
            indexDict=dict() #making a dictionary to keep count of words in each document and initializing it with 0
            for j in uniIdent:
                indexDict[j]=0
            buffDict=indexDict
            for j in globals()['data%s' %i]:
                buffDict[j]+=1
            rowData=[]
            for j in uniIdent:
                if len(globals()['data%s' %i])!=0:
                    rowData.append(buffDict[j]/len(globals()['data%s' %i]))
                else:
                    rowData.append(buffDict[j])
            mat1[i-1]=rowData
            
        #creating IDF matrix
        mat2=np.zeros((c-1,len(uniIdent)))
        indexList=[]
        for i in uniIdent:
            count=0
            for j in range(1,c):
                if i in globals()['data%s' %j]: #iterating through all the documents to find the existance of that word in docs
                    count+=1
            indexList.append(log10((c-1)/count))
        for i in range(mat2.shape[0]):
            mat2[i]=indexList
        
        #multiplying elements of TF and IDF matrix to create TF-IDF matrix
        matMain=np.multiply(mat1,mat2)
        #for i in range(matMain.shape[0]):
            #for j in range(matMain.shape[1]):
                #matMain[i,j]=mat1[i,j]*mat2[i,j]
        return matMain,indexer
                
    def corpusCreater(self):
        global indexer
        #creating corpus to feed to the library
        global corpus
        corpus=[]
        indexer=os.listdir(self.Dir)
        indexer.sort()
        for i in indexer:
            Data=PreProcess(self.Dir+i).fetchComb()
            corpus.append(Data)
        return corpus
    
    def autoTFIDF(self):
        global vectorizer
        self.corpusCreater()
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(corpus)
        return X,indexer

    def Transform(self,doc,method='auto'):
        if method=='auto':
            X=self.autoTFIDF()
            X=X.toarray()
        elif method=='man':
            X=self.manualTFIDF()
        return X[doc-1]
    
    def topFive(self,method='auto'):
        if method=='auto':
            X=self.autoTFIDF()
            X_indexer=vectorizer.get_feature_names()
        elif method=='man':
            X=self.manualTFIDF()
            X_indexer=uniIdent
        try:
            X_mat=X.toarray()
        except:
            X_mat=X
        X_df=pd.DataFrame(X_mat)
        Top5=[]
        for i in range(5):
            buff=[]
            globals()['X1_%s' %i]=X_df.transpose()[[i]].nlargest(5,i).transpose()
            for j in range(5):
                if globals()['X1_%s' %i].iloc[0,j]==0:
                    buffData=None
                else:
                    buffData=X_indexer[globals()['X1_%s' %i].columns[j]]
                buff.append(buffData)
            Top5.append(buff)
        return Top5





