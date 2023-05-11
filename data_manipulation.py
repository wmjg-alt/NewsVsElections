import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
from unidecode import unidecode
warnings.filterwarnings("ignore")

candidate_list_split = " ~ "

def masking_names(row):
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
    tmp =" "
    others = row['candidates'].split(candidate_list_split)
    for nm in others:
        if nm != row['candidate']:
            tmp = tmp + nm
    return tmp

def preproc(df, shuffle=True):
    df['headlines'] = df['headlines'].replace('', pd.NA)
    df.dropna(subset=['headlines'], inplace=True) #vs fillna BLANK
    df['headlines'] = df['headlines'].apply(lambda x: unidecode(x))
    df['opponents'] = df.apply(lambda x: fill_opponent(x), axis=1)
    df['headlines'] = df.apply(lambda x: masking_names(x), axis=1)
    tmp = df.groupby(['year','state','district','totalvotes'], sort=False)
    print("num of races:", len(tmp))
    mx = tmp['candidatevotes'].transform(max)
    ct = tmp['candidatevotes'].transform('count')
    df['winner'] = mx
    df['numcandidates'] = ct
    print("avg num of participants in a race:",df['numcandidates'].mean())
    df['votedisparity'] = df['winner'] - df['candidatevotes']
    df['winner'] = df['winner'] == df['candidatevotes']
    df = df.drop([c for c in df.columns if "Unnamed" in c], axis=1)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def balanced_train_dev_test(df, t_perc, label_column='winner', dev_perc=0.15):
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


if __name__ == "__main__":
    pass
