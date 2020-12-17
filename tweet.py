
# coding: utf-8

# In[2]:


# 导入库

#数据处理
import numpy as np 
import pandas as pd #数据计算与处理
import re   #用于字符串的匹配，调用方法

#自然语言处理工具
import nltk  
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer #词干提取
from collections import Counter  #统计频数

#可视化工具
import seaborn as sns 
import matplotlib.pyplot as plt  
import cufflinks as cf  
cf.go_offline() #如果使用online模式，那么生成的图形是有限制的，我们这里先设置为offline模式，这样就避免了出现次数限制问题。
from wordcloud import WordCloud, STOPWORDS  #词云图
from IPython.core.display import display, HTML  #显示功能
import plotly.graph_objects as go  

#建模相关
from tensorflow.keras.preprocessing.text import Tokenizer  #分词器
from tensorflow.keras.preprocessing.sequence import pad_sequences   #序列与处理
import tensorflow as tf      #机器学习库
from tqdm import tqdm


# In[ ]:


#加载数据

train = pd.read_csv("D:/机器学习/NLP/nlp-getting-started/train.csv")
test = pd.read_csv("D:/机器学习/NLP/nlp-getting-started/test.csv")
submission =  pd.read_csv("D:/机器学习/NLP/nlp-getting-started/sample_submission.csv")


# In[ ]:


train.head()   #查看训练集样本


# id：每条推文的唯一标识符
# text：推文的内容
# location：发送推文的位置（可以为空白）
# keyword：推文中的关键字（可以为空白）
# target：仅在train.csv中，表示一条推文描述的灾难是否真实，真实为（1）或虚假为（0）

# In[ ]:


test.head()    #查看测试集样本


# 此处无target列

# In[ ]:


#数据可视化

#绘制真实虚假灾难数量对比的饼图
counts_train = train.target.value_counts(sort=False) #查看target列中有多少个不同值并计算每个不同值有在该列中有多少重复值
labels = counts_train.index   
values_train = counts_train.values 

data = go.Pie(labels=labels, values=values_train ,pull=[0.03, 0])                  #输入饼图数据
layout = go.Layout(title='Comparing Tweet is a real disaster (1) or not (0) in %') #设置标题

fig = go.Figure(data=[data], layout=layout)                                        #绘制饼图
fig.update_traces(hole=.3, hoverinfo="label+percent+value")                        #设置饼图的显示，标签，百分比，值
fig.update_layout(annotations=                                                     #在饼图的中心添加注释
                  [dict(text='Train', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()                                                                         #显示饼图


# In[ ]:


#绘制推文内容的词云
STOPWORDS.add('https')  #将https加入停用词表
STOPWORDS.add('co')
STOPWORDS.add('amp')
#定义函数：绘制词云图
def Plot_world(text):
    
    comment_words = ' '
    stopwords = set(STOPWORDS)    #创建停用词元素集
    
    for val in text:    #遍历text中的值
        val = str(val)  #将每个值转换成字符串
        tokens = val.split()    #分割值

        for i in range(len(tokens)):    
            tokens[i] = tokens[i].lower()   #遍历分割后的字符串，并将每个词转换成小写

        for words in tokens: 
            comment_words = comment_words + words + ' '   #遍历字符串中的所有词,将每个词用空格隔开


    wordcloud = WordCloud(width = 4000, height = 3000,    #生成词云
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 

    #绘制词云                        
    plt.figure(figsize = (12, 12), facecolor = 'white', edgecolor = 'white' ) 
    plt.imshow(wordcloud) 
    plt.axis("off")            #不显示坐标轴
    plt.tight_layout(pad = 0)  #图像自动调整，设置间距
 
    plt.show() 
    wordcloud.to_file('wordcloud.png')
#运行函数    
text = train.text.values
Plot_world(text)


# In[ ]:


#数据清洗

#删除网址
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


# In[ ]:


train['text']=train['text'].apply(lambda x : remove_URL(x))


# In[ ]:


#删除HTML标签
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[ ]:


train['text']=train['text'].apply(lambda x : remove_html(x))


# In[ ]:


#删除表情
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[ ]:


train['text']=train['text'].apply(lambda x: remove_emoji(x))


# In[ ]:


#删除标点符号
import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


# In[ ]:


train['text']=train['text'].apply(lambda x : remove_punct(x))


# In[ ]:


#拼写校正（速度慢）
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# In[ ]:


train['text']=train['text'].apply(lambda x : correct_spellings(x))


# In[ ]:


#将清洗后的数据保存
train.to_csv('clean_train.csv')


# In[3]:


#重新载入数据
train = pd.read_csv('clean_train.csv')


# In[4]:


#文本序列化
 
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()              #分词器
tokenizer.fit_on_texts(train.text)   #用本文内容学习字典
word_index = tokenizer.word_index    #将词映射为索引值
print('Number of unique words:',len(word_index))


# In[5]:


training_sequences = tokenizer.texts_to_sequences(train.text)  #将训练集的文本转化为序列
MAX_LEN=20
training_padded = pad_sequences(training_sequences,      #填充序列
                                maxlen=MAX_LEN,          #设置序列长度
                                padding='post',          #需要补充时确定补0的位置 结尾
                                truncating='post')       #需要截断时确定截断位置  结尾


# In[6]:


#嵌入glove字典
embedding_dict={}
with open('D:/机器学习/NLP/IMDB/glove.6B.100d.txt','r',encoding='utf-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()


# In[7]:


#匹配GloVe向量
num_words=len(word_index)+1
embedding_dim=100
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in tqdm(word_index.items()):
    if i < num_words:
        #返回glove字典中相应的word对应的词向量
        embedding_vector = embedding_dict.get(word)  
        if embedding_vector is not None:
        #在嵌入索引中找不到的单词将为全零。
            embedding_matrix[i] = embedding_vector

embedding_matrix.shape


# In[8]:


#建立模型
from keras.models import Sequential 
from keras.layers import Embedding,Dense,Dropout,LSTM
from keras import optimizers,initializers

def create_model():
    model = Sequential() 
                        #input_dim即词汇量，输入数组中的词典大小是14666，即有14666个不同的词，所以我的input_dim便要比14666要大1，
    model.add(Embedding(input_dim=num_words,
                        #output_dim是密集嵌入的尺寸，就如同CNN最后的全连接层一样，上面设置的100，便将每一个词变为用1x100来表示的向量，
                        output_dim=100,
                        #嵌入矩阵的初始化的方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。
                        embeddings_initializer=initializers.Constant(embedding_matrix), 
                        #输入序列的长度为20，本层将每个text表示为一个20×100的矩阵
                        input_length=MAX_LEN,trainable=False))
    #正则化，防止过拟合
    model.add(Dropout(0.2))    
    #LSTM层，长短期记忆
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    #全连接层
    model.add(Dense(1, activation='sigmoid')) 
    #编译模型
    model.compile(loss='binary_crossentropy',  #一般搭配sigmoid
                  optimizer='adam',            #优化器
                  metrics=['accuracy'])        #模型评估标准
    return model

#模型摘要
model=create_model()
model.summary()


# In[9]:


#划分训练集测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(training_padded,train['target'].values,test_size=0.2)


# In[10]:


#参数选择与调优———网格搜索和交叉验证
#Scikit-Learn里有一个API 为model.selection.GridSearchCV，可以将keras搭建的模型传入，作为sklearn工作流程一部分。
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV 
#包装keras模型以便在scikit-learn中使用keras，该模型用作scikit-learn中的estimator
model = KerasClassifier(build_fn=create_model, verbose=0) 
#定义网格搜索超参数
batch_size = [5, 10, 50, 100] 
epochs = [5, 10, 20, 50] 
#参数字典
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
#网格搜索，5折交叉验证
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1) 


# In[11]:


#训练获得最优模型
grid_result = grid.fit(training_padded, train['target'].values) 


# In[12]:


#查看训练结果信息
results= pd.DataFrame(grid_result.cv_results_)
results.head(16)


# In[13]:


#不同参数组合的准确率图像
results[['mean_train_score','mean_test_score']].plot(ylim=[0.75,0.85])


# In[14]:


#模型评估
#评估查看最终选择的结果和交叉验证的结果
print("在交叉验证中验证的最好结果：\n", grid_result.best_score_)
print("最好的参数模型：\n", grid_result.best_params_)
#print("测试集准确率：\n", grid_result.score(X_test,y_test))


# In[15]:


#对比SVM,KNN,多层感知器,朴素贝叶斯，随机森林方法
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm,neighbors,neural_network,naive_bayes,ensemble
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import *

model1= svm.SVC()
model1.fit(X_train,y_train)
predict1=model1.predict(X_test)

model2=neighbors.KNeighborsClassifier()
model2.fit(X_train,y_train)

model3=neural_network.MLPClassifier()
model3.fit(X_train,y_train)

model4=naive_bayes.GaussianNB()
model4.fit(X_train,y_train)

model5=ensemble.RandomForestClassifier()
model5.fit(X_train,y_train)

model6=ensemble.GradientBoostingClassifier()
model6.fit(X_train,y_train)


# In[16]:


#模型评估
print('模型1测试集准确率:',model1.score(X_test, y_test))
print('模型2测试集准确率:',model2.score(X_test, y_test))
print('模型3测试集准确率:',model3.score(X_test, y_test))
print('模型4测试集准确率:',model4.score(X_test, y_test))
print('模型5测试集准确率:',model5.score(X_test, y_test))
print('模型6测试集准确率:',model6.score(X_test, y_test))
print('模型1测试集AUC:',metrics.roc_auc_score(y_test,predict1))


# In[ ]:


#test.csv结果预测
test = pd.read_csv("D:/机器学习/NLP/nlp-getting-started/test.csv")
testing_sequences2 = tokenizer.texts_to_sequences(test.text)
testing_padded2 = pad_sequences(testing_sequences2, maxlen=MAX_LEN, padding='post', truncating='post')
predictions = grid_result.predict(testing_padded2)


# In[ ]:


submission['target'] = (predictions > 0.5).astype(int)
submission


# In[ ]:


submission.to_csv("submission.csv", index=False, header=True)

