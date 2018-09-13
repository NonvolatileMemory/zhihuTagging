from mxnet import init
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from mxnet import nd
from mxnet.gluon import rnn
import pickle
from mxnet import gluon
from mxnet import autograd

ctx = mx.gpu(2)
batch_size = 1

class SMN_Last(nn.Block):
    def __init__(self,**kwargs):
        super(SMN_Last,self).__init__(**kwargs)
        with self.name_scope():
            
            self.Embed = nn.Embedding(525000,256)
            # agg param
            self.mlp_1 = nn.Dense(units=2000,flatten=True,activation='relu')
            self.mlp_2 = nn.Dense(units=16586,flatten=True)
            # lstm param

    def forward(self,x):
        """
        return shape:(batch_size,2000,2)
        """
        # Encode layer
        question = x[:,0:30]
        question = self.Embed(question)
        #question = nd.mean(question,axis=1)


        res = self.mlp_2(self.mlp_1(question))


        return res
#Train Model

#Train Model
SMN = SMN_Last()
SMN.load_params("86ft4.params",ctx=ctx)

import pickle
pkl_file = open('worddict.pkl','rb')
vocab    = pickle.load(pkl_file)

topicVocab = {}
with open("topicVocab",'r') as f:
    count = 1
    for line in f:
        topicVocab[count] = line
        count = count + 1
import jieba
while True:
    print("请输入问题： ")
    ques_list = []
    question = input()
    question = question.replace(" ","")
    seg_list = jieba.cut(question, cut_all=False, HMM=True)
    question = " ".join(seg_list)
    for word in question.split(" "):
        if(vocab.get(word,0)>=74):
            ques_list.append(vocab.get(word,"0"))
        else:
            ques_list.append(vocab.get(word,"0"))
    question = [int(k) for k in ques_list]
    question = [0]*(30-len(question)) + question
    print(question)
    question = nd.array(question, ctx=ctx)
    question = question.reshape((1,30))
    topic = SMN(question)
    tt = topic
    topic = nd.topk(topic, axis=1, ret_typ='indices', k=10)
    logits =  nd.topk(tt, axis=1, ret_typ='value', k=10)
    print(topic)
    print(logits)
    topic = topic.asnumpy()
    question_outputs=topic.reshape(-1)
    print("推荐的标签是：")
    for num in range(question_outputs.shape[0]):
        print(topicVocab[question_outputs[num]+1])

