from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report

# TFIDF VEC params
Vec_sw = 'english'
Vec_max_f = 10000
Vec_ngr = (1,1)


def bow_plus(df): 
    # take a pandas df;
    # build a TdidfVectorize Countvec
    # a weighted 10000 vocab vector of unigrams + party affiliation as 0,1,2

    def party_adjust(r):
        # democrat = 1, republican = 0, independent = 2
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
    # train/dev/test BOW sets and a model
    # train by fitting to train+dev
    # eval on test
    # print classification report
    tdBOW = pd.concat([trainBOW, devBOW])

    Xtrain = tdBOW.loc[:,tdBOW.columns != 'winner'].to_numpy()
    ytrain = tdBOW['winner']
    Xtest = testBOW.loc[:,testBOW.columns != 'winner'].to_numpy()
    ytest = testBOW['winner']

    model.fit(X=Xtrain, y=ytrain)
    preds = model.predict(Xtest)
    print(classification_report(preds,ytest))

    return model
