#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:40:14 2019

@author: lindsey
"""

from bert_embedding import BertEmbedding
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
import time
from sklearn.neighbors import NearestNeighbors

######################################################################################## 
# construct the argument parser and parse the arguments
########################################################################################
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--input", required=True,help="description of book in strings")
#ap.add_argument("-o", "--output", required=False,help="path to output image")
#args = vars(ap.parse_args())                         

class mastercode:
    bookTitle='user_input_title_here'  
    description=str('this should be retrieved from user input. this is a test. ')#args["input"]#
    tfidf="/home/lindsey/insightproject/tfidfweights.csv"
    database="/home/lindsey/insightproject/weighted43witharray.pkl"
    m=768

##reading the threeDPositions files and: aggregate into one master files
    def __init__(self,description):
        self.description=description
###bert embedding parag
    def pa_to_sen(self):
        """ input paragraph in string
            output list of m by 2
            where m is number of sentences
            n= [0] element is txt, n=[1] is 768 by w where w is number of words                             
        """
        bert_embedding = BertEmbedding()
        self.sentences=self.description.split('. ')
        self.sent_bert = bert_embedding(self.sentences)
        self.num_sen=len(self.sent_bert)
        return self.sent_bert,self.num_sen,self.sentences
    ##return weights dependingi f its in dictionary##
    def wei(self,txt):
        """ input the word 
        output the weight associated with the word
        if word is in tfidf then give assigned values
        if word is not in tfidf then give it assigned 1
        need to update this is database is changed                            
        """
        self.txt=txt
        self.read=pd.read_csv(self.tfidf,index_col=0)
        self.weights = self.read.to_dict("split")
        self.weights = dict(zip(self.weights["index"], self.weights["data"]))
        if self.txt in self.weights:
            self.weight=(self.weights[self.txt])[0]
        else:
            self.weight=1
        return self.weight
    ##given a word array and its word txt determine the sum##
    def sum_words(self,wordtxt,word,total):
        self.word=word
        self.wordtxt=wordtxt
        self.total=total
        self.weight= self.wei(self.wordtxt)
        self.total += self.weight*self.word
        return self.total
    
    def avg_words(self,total,length):
        self.total=total
        self.length=length
        self.avg=self.total/self.length
        return self.avg
    
    def f2hex(self,f2rgb, f):
        self.f=f
        rgb = f2rgb.to_rgba(self.f)[:3]#change colormap f2rgb to rgba color, where a is opacity
        return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])##scaling values from 0 to 1 back to 0 to 255
    
    def master(self):
    ##retrive description from users##
        ###load current databse###
        self.current_data=pd.read_pickle(self.database)
        self.booknumber=max(np.unique(np.array(self.current_data['book#'])))+1 
        self.i=max(np.unique(np.array(self.current_data['index'])))+1 
        ##initialize dataframes for storing new data##
        self.avgs=pd.DataFrame()
        self.sent_bert,self.num_sen,self.sentences=self.pa_to_sen()
        ##getting new data##
        for s in np.arange(self.num_sen-1):
            self.sentence=self.sentences[s]
            self.words=(self.sent_bert[s])[1]
            self.wordtxts=(self.sent_bert[s])[0]
            self.length=len(self.words)
            self.total=np.zeros(self.m,)##initialize total array for each sentence
            if(self.length==0):
                continue
            for w in np.arange(self.length):
                self.word=self.words[w]
                self.wordtxt=self.wordtxts[w]
                self.total=self.sum_words(self.wordtxt,self.word,\
                                          self.total)##update total every words
                self.avg=self.avg_words(self.total,self.length)
            self.current=pd.DataFrame( {'book#':self.booknumber, \
                                   'booktitle':self.bookTitle,\
                                   'sentence#': s,\
                                   'array':[self.avg],
                                   "index":self.i,\
                                   'sentence':[self.sentence]},\
                index=[str(self.i)])
            self.avgs=self.avgs.append(self.current,ignore_index=True)
        
        
        
        # to append newdata at the end of olddata dataframe 
        self.result=self.current_data.append(self.avgs, ignore_index = True) 
        return self.result, self.avgs
    
    def dim_reduc(self):
        self.result,self.avgs= self.master()
        ####assign colors based on book###
        self.norm = colors.Normalize(vmin=0, \
                                     vmax=max(np.array(self.result['book#'])))
        self.f2rgb = cm.ScalarMappable(norm=self.norm, \
                                       cmap=cm.get_cmap('prism'))       
                              
        ####assign sizes and colors to all data####
        
        self.rangeofbook=np.array(self.result['book#'])
                                    
        for b in self.rangeofbook:
            if b < self.booknumber:
                self.result.loc[self.result['book#']==b,'scale']=25
            elif b == self.booknumber:
                self.result.loc[self.result['book#']==b,'scale']=500
                                  
        for b in self.rangeofbook:
            self.color=self.f2hex(self.f2rgb,b)
            self.result.loc[self.result['book#']==b,'color']=self.color
        
        
        ###CURRENT DATA POINT IN 2D DIMENSION####
        self.newarr=np.array(self.avgs['array'].values.tolist())
        tsne_model = TSNE(perplexity=40, n_components=2, \
                          init='pca', n_iter=2500, random_state=23)
        self.new_values_P = tsne_model.fit_transform(self.newarr)
        self.Point=self.new_values_P.tolist()
            
        ######preparring inputdata for plotting#####
        self.arr = np.array(self.result['array'].values.tolist())  
        ####reduce dimenson of input data#####  
        self.new_values_S = tsne_model.fit_transform(self.arr)
        self.x=self.new_values_S[:,0]
        self.y=self.new_values_S[:,1]
        self.Samples=self.new_values_S.tolist()
        return self.Samples,self.Point,self.x,self.y,self.result
    
    def plot(self):
        self.Samples,self.Point, self.x,self.y,self.result=self.dim_reduc()      
        self.fig = plt.figure(figsize=(12.0, 7.0),dpi=100,linewidth=0.5)
        self.ax = self.fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            self.ax.spines[axis].set_linewidth(0.5)
        
        self.ax.set_xlabel('x',fontsize=11,linespacing=3.2)
        self.ax.set_ylabel('y',fontsize=11,linespacing=3.2)
        #
        for self.index, self.row in self.result.iterrows():#interating row by row in
            self.color=self.row['color']
            self.label=self.row['index']
            self.size=self.row['scale']
            self.a=self.ax.scatter(self.x[self.index],self.y[self.index], \
                                   marker='o',c=str(self.color),s=self.size)
            self.ax.annotate(self.label,(self.x[self.index],\
                                         self.y[self.index]))
        #plt.show()####or plt.save for web inquiry
        plt.savefig('/home/lindsey/insightproject/website/static/myfigure.png')
        
    
    def output_neighbors(self):
        #######################
        self.Samples,self.Point, self.x,self.y,self.result=self.dim_reduc()
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(self.Samples) 
        self.Suggestions=neigh.kneighbors(self.Point,return_distance=False) 
        
        self.yoursentence=[]
#        self.returnsentences=[]
        for i in np.arange(len(self.Suggestions)):
            self.Neighbors=self.Suggestions[i]
            self.row=self.avgs.loc[self.avgs.index==i]
            self.sen1=list(self.row['sentence'].values)
            self.yoursen=str('your sentence \
                             : {}'.format(" ".join(str(x) for x in self.sen1)))
            self.yoursentence.append(self.yoursen)
            for j in self.Neighbors:
                self.frame=self.result.loc[self.result.index==j]
                self.bookname=list(self.frame['booktitle'].values)
                self.sennum=list(self.frame['index'].values)
                self.sen2=list(self.frame['sentence'].values)
                self.returnsen=str('Closest Neightbor{}:{} sentence\
                                   : {}'.format(self.sennum,\
                                   " ".join(str(x) for x in self.bookname),\
                                   " ".join(str(x) for x in self.sen2)))
                self.yoursentence.append(self.returnsen)
        
        return self.yoursentence#,self.returnsentences
        


##print out neighboring result