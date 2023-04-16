import pandas as pd
import numpy as np
import random
import string
import time
from memory_profiler import profile 

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

@profile
def main():
    S = 50000
    df=pd.DataFrame({
        "did":[''.join(random.choices(string.ascii_uppercase + string.digits, k = 30)) for i in range(S)],
        "active":np.random.random(S),  
        "vector":[np.random.random(64) for i in range(S)]
    })
    
    df1 = pd.DataFrame({
        "did":df['did'][:S//4].tolist() + [''.join(random.choices(string.ascii_uppercase + string.digits, k = 30)) for i in range(S//4)],
        "token":[''.join(random.choices(string.ascii_uppercase + string.digits, k = 140)) for i in range(S//2)]
    })
    df = df.sample(frac=1).reset_index(drop=True)
    df1 = df1.sample(frac=1).reset_index(drop=True)

    # print(df)
    # print(df1)
    # print(df1.info(memory_usage = 'deep'))

    start = time.time()
    data = pd.merge(df,df1,how = 'right')
    # print(data)
    cadidate = data[data['active'].isnull()].copy()
    choose = data[~data['active'].isnull()].copy()   
    print(str(S)+'规模数据筛选耗时：',time.time() - start)

    

    start = time.time()
    res = []
    for i in range(10):
        para1 = np.random.random((64,320))-0.5
        b1 = np.random.random(320)-0.5
        para2 = np.random.random((320,4))-0.5
        b2 = np.random.random(4)-0.5
        para3 = np.random.random((320,4))-0.5
        b3 = np.random.random(4)-0.5

        vector= np.array(choose['vector'].tolist())
        x = np.matmul(vector,para1) + b1
        x = np.maximum(x,0)
        y1 = np.matmul(x,para2) + b2
        y2 = np.matmul(x,para3) + b3
        y1 = 1. / (1. + np.exp(-y1))
        y2 = softmax(y2, axis=1)
        y = (y1*y2).sum(axis = 1, keepdims = True)
        res.append(y)
    res = np.concatenate(res,axis = 1)
    # print(res)
    print(str(S)+'规模模型计算耗时：',time.time() - start)

    start = time.time()
    index = np.argmax(res, axis = 1)
    vals = np.max(res, axis = 1)
    choose['idx'] = index
    choose['vals'] = vals
    choose.sort_values('vals',ascending = False,inplace=True)
    print(str(S)+'规模排序耗时：',time.time() - start)



main()



