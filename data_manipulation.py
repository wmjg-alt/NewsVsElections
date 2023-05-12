import pandas as pd
import matplotlib.pyplot as plt
import re
from unidecode import unidecode

candidate_list_split = " ~ "


def masking_names(row):
    # mask out candidate and opponent's names in all headlines
    target = re.sub(r'[^\w\s]', '', row['candidate'])
    opponent = re.sub(r'[^\w\s]', '', row['opponents'])
    text = re.sub(r'[^\w\s]', '', row['headlines'])

    text = text.lower()
    opponent = opponent.lower()
    target = target.lower()

    text = re.sub(r'\d+', '', text)

    for wd in target.split(' '):
        tmp = [t if t != wd and t != wd+"s" else 'the candidate' for t in text.split(' ')]
        text = " ".join(tmp)
        
    for wd in opponent.split(' '):
        tmp = [t if t != wd and t != wd+"s" else 'the opponent' for t in text.split(' ')]
        text = " ".join(tmp)
        
    return text


def fill_opponent(row):
    # gen a df column containing split of ALL candidates
    tmp =" "
    others = row['candidates'].split(candidate_list_split)
    for nm in others:
        if nm != row['candidate']:
            tmp = tmp + nm
    return tmp


def preproc(df:pd.DataFrame, shuffle:bool=True):
    # normalize data in pandas df
    # gen extra informational columns
    # shuffle df, if shuffle
    df['headlines'] = df['headlines'].replace('', pd.NA)
    df.dropna(subset=['headlines'], inplace=True) #vs fillna BLANK

    df['headlines'] = df['headlines'].apply(lambda x: unidecode(x))
    df['opponents'] = df.apply(lambda x: fill_opponent(x), axis=1)
    df['headlines'] = df.apply(lambda x: masking_names(x), axis=1)

    tmp = df.groupby(['year','state','district','totalvotes'], sort=False)

    print("Num of races:", len(tmp))

    mx = tmp['candidatevotes'].transform(max)
    ct = tmp['candidatevotes'].transform('count')
    df['winner'] = mx
    df['numcandidates'] = ct

    print("Avg num of participants in a race:",df['numcandidates'].mean())

    df['votedisparity'] = df['winner'] - df['candidatevotes']
    df['winner'] = df['winner'] == df['candidatevotes']

    df = df.drop([c for c in df.columns if "Unnamed" in c], axis=1)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def balanced_train_dev_test(df:pd.DataFrame, 
                            t_perc:float,
                            label_column:str='winner', 
                            dev_perc:float=0.15):
    # a balance train_dev_test splitter
    # df : data in pd.Dataframe
    # t_perc: percent of train split
    # label_column: df column with labels to separate
    # dev_perc: dev set percentage (can be 0)
    # balance equal percents of 2 class labels in train/dev/test
    # return train, dev, test
    s = len(df)
    c1_df = df[df[label_column] == True]
    c2_df = df[df[label_column] == False]

    assert(s == len(c1_df)+ len(c2_df))

    c1_tr_sz  = int(len(c1_df)*t_perc)                  # size of train set of winners
    c2_tr_sz  = int(len(c2_df)*t_perc)                  # size of train set of losers
    c1_dev_sz = int((len(c1_df)*dev_perc) + c1_tr_sz)   # end index of dev set of winners
    c2_dev_sz = int((len(c2_df)*dev_perc) + c2_tr_sz)   # end index of dev set of losers

    train   = pd.concat([c1_df[:c1_tr_sz], c2_df[:c2_tr_sz]])
    train   = train.sample(frac=1).reset_index(drop=True)
    print('TRAIN',len(train))

    dev     = pd.concat([c1_df[c1_tr_sz:c1_dev_sz], c2_df[c2_tr_sz:c2_dev_sz]])
    dev     = dev.sample(frac=1).reset_index(drop=True)
    print('DEV',len(dev))

    test    = pd.concat([c1_df[c1_dev_sz:], c2_df[c2_dev_sz:]])
    test    = test.sample(frac=1).reset_index(drop=True)
    print('TEST',len(test))

    return train, dev, test
