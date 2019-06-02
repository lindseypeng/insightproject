from bert_embedding import BertEmbedding
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
import re
from ast import literal_eval
                               
###bert embedding parag
def pa_to_sen(description):
    """ input paragraph in string
        output list of m by 2
        where m is number of sentences
        n= [0] element is txt, n=[1] is 768 by w where w is number of words                             
    """
    bert_embedding = BertEmbedding()
    sentences=description.split('. ')
    sent_bert = bert_embedding(sentences)
    num_sen=len(sent_bert)
    return sent_bert,num_sen,sentences
##return weights dependingi f its in dictionary##
def wei(tfidf,txt):
    """ input the word 
    output the weight associated with the word
    if word is in tfidf then give assigned values
    if word is not in tfidf then give it assigned 1
    need to update this is database is changed                            
    """
    read=pd.read_csv(tfidf,index_col=0)
    weights = read.to_dict("split")
    weights = dict(zip(weights["index"], weights["data"]))
    if txt in weights:
        weight=(weights[txt])[0]
    else:
        weight=1
    return weight
##given a word array and its word txt determine the sum##
def sum_words(wordtxt,word,tfidf,total):
    weight=wei(tfidf,wordtxt)
    total += weight*word
    return total

def avg_words(total,length):
    avg=total/length
    return avg

def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]#change colormap f2rgb to rgba color, where a is opacity
    return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])##scaling values from 0 to 1 back to 0 to 255


##retrive description from users##
bookTitle='user_input_title_here'  
description=str('this should be retrieved from user input. this is a test. ')
tfidf="/home/lindsey/insightproject/tfidfweights.csv"
m=768
###load current databse###
current_data=pd.read_pickle('/home/lindsey/insightproject/weighted43witharray.pkl')
booknumber=max(np.unique(np.array(current_data['book#'])))+1 
i=max(np.unique(np.array(current_data['index'])))+1 
##initialize dataframes for storing new data##
avgs=pd.DataFrame()
avgsstacks=np.empty((1000,m))
sentexts=pd.DataFrame()
sent_bert,num_sen,sentences=pa_to_sen(description)
##getting new data##
for s in np.arange(num_sen-1):
    sentence=sentences[s]
    words=(sent_bert[s])[1]
    wordtxts=(sent_bert[s])[0]
    length=len(words)
    total=np.zeros(m,)##initialize total array for each sentence
    if(length==0):
        continue
    for w in np.arange(length):
        word=words[w]
        wordtxt=wordtxts[w]
        total=sum_words(wordtxt,word,tfidf,total)##update total every words
        avg=avg_words(total,length)
    current=pd.DataFrame( {'book#':booknumber, 'booktitle':bookTitle,\
                           'sentence#': s,'array':[avg],"index":i,\
                           'sentence':[sentence]},index=[str(booknumber)])
    avgs=avgs.append(current,ignore_index=True)



# to append newdata at the end of olddata dataframe 
result=current_data.append(avgs, ignore_index = True) 

####assign colors based on book###
norm = colors.Normalize(vmin=0, vmax=max(np.array(result['book#'])))
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('prism'))
for b in np.array(result['book#']):
    color=f2hex(f2rgb,b)
    result.loc[result['book#']==b,'color']=color        
                      
####assign sizes and colors to all data####

rangeofbook=np.array(result['book#'])
                            
for b in rangeofbook:
    if b < booknumber:
        result.loc[result['book#']==b,'scale']=25
    elif b == booknumber:
        result.loc[result['book#']==b,'scale']=500
                          
for b in rangeofbook:
    color=f2hex(f2rgb,b)
    result.loc[result['book#']==b,'color']=color


###CURRENT DATA POINT IN 2D DIMENSION####
newarr=np.array(avgs['array'].values.tolist())
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values_P = tsne_model.fit_transform(newarr)
Point=new_values_P.tolist()
    
######preparring inputdata for plotting#####
arr = np.array(result['array'].values.tolist())  ##check shape .shape
####reduce dimenson of input data#####  
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values_S = tsne_model.fit_transform(arr)
x=new_values_S[:,0]
y=new_values_S[:,1]
Samples=new_values_S.tolist()
#####plotting####################################      
fig = plt.figure(figsize=(12.0, 7.0),dpi=100,linewidth=0.5)
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

ax.set_xlabel('x',fontsize=11,linespacing=3.2)
ax.set_ylabel('y',fontsize=11,linespacing=3.2)
#
for index, row in result.iterrows():#interating row by row in
    color=row['color']
    label=row['index']
    size=row['scale']
    a=ax.scatter(x[index],y[index], marker='o',c=str(color),s=size)
    ax.annotate(label,(x[index],y[index]))


plt.show()####or plt.save for web inquiry
#######################
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=3)
neigh.fit(Samples) 
Suggestions=neigh.kneighbors(Point,return_distance=False) 

for i in np.arange(len(Suggestions)):
    Neighbors=Suggestions[i]
    point=Point[i]
    row=avgs.loc[avgs.index==i]
    sen1=list(row['sentence'].values)
    print('your sentence : {}'.format(" ".join(str(x) for x in sen1)))
    for j in Neighbors:
        frame=result.loc[result.index==j]
        bookname=list(frame['booktitle'].values)
        sennum=frame['index']
        sen2=list(frame['sentence'].values)
        print('{}:{} sentence: {}'.format(j," ".join(str(x) for x in bookname)," ".join(str(x) for x in sen2)))
        
