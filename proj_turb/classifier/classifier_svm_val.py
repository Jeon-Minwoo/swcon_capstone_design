from random import random
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, GridSearchCV
import numpy as np
import pandas as pd

##data preprocessing
def svm_():
    train_df = pd.read_csv('solution_data.csv')
    #train_df['pf'] = train_df['pf'].map({'p0': 1, 'f':0})
    train_df['target'] = train_df['target'].map({'p0':2, 'f2':1, 'f1':0})
    #train_df['target'] = train_df['target'].map({'p0':1, 'f2':0, 'f1':0})

    #print(train_df.iloc[:, [1]].values)
    
    # train_df.iloc[:, [1]] /=10
    # train_df.iloc[:, [2]] /=100
    # train_df.iloc[:, [3]] /=10
    # train_df.iloc[:, [4]] /=10
    # train_df.iloc[:, [5]] /=10
    # train_df.iloc[:, [6]] /= 1000
    # train_df.iloc[:, [7]] /= 100
    # train_df.iloc[:, [9]] /= 1000
    
    input = train_df.iloc[:,2:-1].values
    
    target = train_df.iloc[:, -1].values

    # C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # kernel = ['rbf', 'linear']
    # hyper={'kernel':kernel,'C':C,'gamma':gamma}
    # gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
    # gd.fit(input, target)
    # print(gd.best_score_)
    # print(gd.best_estimator_)

    # for i in range(0, len(input)):
    #     print(input[i], target[i])
    s = svm.SVC(gamma=0.1, C=0.05, kernel='linear')
    s.fit(input, target)
    return s

if __name__ == '__main__':
    train_df = pd.read_csv('solution_data.csv')
    #train_df['pf'] = train_df['pf'].map({'p0': 1, 'f':0})
    train_df['target'] = train_df['target'].map({'p0':2, 'f2':1, 'f1':0})
    #train_df['target'] = train_df['target'].map({'p0':1, 'f2':0, 'f1':0})
    
    #print(train_df.iloc[:, [-1]].values)
    kfold = KFold(n_splits=4, shuffle=True)
    
    input = train_df.iloc[:,2:-1].values
    target = train_df.iloc[:, -1].values
    
    s = svm_()
    res = s.predict(input)
    print(res)

    scores = cross_val_score(s, input, target,scoring='accuracy', cv=kfold, n_jobs=-1)
    print(scores*100)
    print('avg: ', sum(scores, 0.0)/len(scores)*100)
    
    # print(input[3].reshape(1,-1))
    # res_ = s.predict(input[3].reshape(1,-1))
    # print(res_)

    #conf = np.zeros((2,2))
    conf = np.zeros((3,3))
    for i in range(len(res)):
        conf[res[i]][target[i]] +=1
    print(conf)

    no_correct= 0
    for i in range(3):
        no_correct += conf[i][i]
    accuracy = no_correct/len(res)
    print(accuracy*100)

