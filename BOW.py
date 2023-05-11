from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report

Vec_sw = 'english'
Vec_max_f = 10000
Vec_ngr = (1,1)

def bow_plus(df): 
    def party_adjust(r):
        if 'DEMOC' in r:
            return 1.0
        elif "REPUB" in r:
            return 0.0
        else: 
            return 2.0

    docs = df['headlines']

    df['party'] =df['party'].fillna('-1')
    parties = df['party'].apply(lambda r: party_adjust(r))
    print('Parties D, R, I:',Counter(parties))

    CountVec = TfidfVectorizer(use_idf=True, 
                               smooth_idf=False,
                               ngram_range=Vec_ngr, # to use bigrams ngram_range=(2,2)
                               stop_words=Vec_sw,
                               max_features=Vec_max_f)
        
    Count_data = CountVec.fit_transform(docs)

    cvdf = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())
    cvdf['PARTYX'] = parties
    cvdf['PARTYX'].fillna(-1,inplace=True)
    cvdf['winner'] = df['winner']
    return cvdf

def train_w_BOW(trainBOW,devBOW,testBOW, model):
    tdBOW = pd.concat([trainBOW, devBOW])

    Xtrain = tdBOW.loc[:,tdBOW.columns != 'winner'].to_numpy()
    ytrain = tdBOW['winner']
    Xtest = testBOW.loc[:,testBOW.columns != 'winner'].to_numpy()
    ytest = testBOW['winner']

    model.fit(X=Xtrain, y=ytrain)
    preds = model.predict(Xtest)
    print(classification_report(preds,ytest))
