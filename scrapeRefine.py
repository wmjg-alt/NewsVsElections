from zipfile import ZipFile
import re
import pandas as pd
from data_manipulation import candidate_list_split

#HELPERS
def read_zip(zip_fn, extract_fn=None):
    zf = ZipFile(zip_fn)
    if extract_fn:
        return zf.read(extract_fn)
    else:
        return [zf.read(name) for name in zf.namelist()][0]

def preprocess_headlines(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    text = re.sub(r'\d+', '', text)
    return text
    
def name_norm(n):
    return re.sub(r'[^A-Za-z0-9 ]+','',n)

def masking_names(row):
    target = re.sub(r'[^\w\s]', '', row['candidate'])
    opponent = re.sub(r'[^\w\s]', '', row['opponents'])
    text = re.sub(r'[^\w\s]', '', row['headlines'])

    text = text.lower()
    opponent = opponent.lower()
    target = target.lower()

    text = re.sub(r'\d+', '', text)

    for wd in target.split(' '):
        tmp = [t if t != wd and t != wd+"s" else 'thecandidate' for t in text.split(' ')]
        text = " ".join(tmp)
        
    for wd in opponent.split(' '):
        tmp = [t if t != wd and t != wd+"s" else 'theopponent' for t in text.split(' ')]
        text = " ".join(tmp)
        
    return text


def race_matching(candidate0, year, state, source, headlines):
    def candidate_matcher(c, y, src,  state, col):
        if state != None:
            state= state.strip()
            tmp = src.index[((src[col] == c) & (src['state'] == state)) & (src['year'] == y)]
            if len(tmp) == 1:
                return tmp[0]
        else:
            tmp = src.index[(src[col] == c) & (src['year'] == y)]
            if len(tmp) == 1:
                return tmp[0]
        
        return None
    
    candidates = source['candidate'].unique()
    presidents = source[source['office'] == "US PRESIDENT"]['candidate'].unique()

    r = candidate_matcher(candidate0, year, source, state,'candidate')
    if r:
        return r

    r = candidate_matcher(candidate0, year, source, state, 'candidate_norm')
    if r:
        return r
    
    return -1
    
def fill_opponent(row):
    tmp =" "
    others = row['candidates'].split(candidate_list_split)
    for nm in others:
        if nm != row['candidate']:
            tmp = tmp + nm
    return tmp


def fetch_preprocess_targets(f, dated):
    def perc(row):
        if row['totalvotes'] == 0:
            return 0
        return row['candidatevotes'] / row['totalvotes']

    df = pd.read_csv(f,encoding = "ISO-8859-1")
    if 'president' in f:
        df['district'] = ['nationwide']*len(df)
    
        
    df['percentvote'] = df.apply(lambda row: perc(row), axis=1)

    df['date'] = df.apply(lambda row: dated[row['year']], axis=1)
    
    df = df[df['candidate'].notna()]
    df['candidate_norm'] = df['candidate'].apply(lambda n: name_norm(n))
    
    if 'party_detailed' in df.columns:
        df['party'] = df['party_detailed']
    
    if 'runoff' in df.columns:
        df = df[df['runoff'] !=True]
    
    if 'special' in df.columns:
        df = df[df['special'] ==False] 
    
    if 'stage' in df.columns:
        df = df[df['stage'].str.lower() == 'gen']
    
    df = df.drop(['version',
                  'notes',
                  'writein',
                  'unofficial',
                  'party_detailed',
                  'party_simplified', 
                  'runoff',
                  'special',
                  'mode',
                  'stage',
                  'fusion_ticket'], axis=1, errors='ignore')    #state_fips	state_cen	state_ic?
    
    df = df[~(df['year'] < 1999)]  

    df = df.sort_values(['year',
                         'state',
                         'district',
                         'totalvotes',
                         'percentvote'],ascending=False).groupby(['year',
                                                                  'state',
                                                                  'district',
                                                                  'totalvotes'])#.head(2).reset_index(drop=True) 
    
    grouped_df = []
    for name, group in df:
        tmp = group.head(2)
        if tmp['percentvote'].sum() >= 0.70:
            pass
        else:
            tmp = group.head(3)

        tmp['candidates'] = [candidate_list_split.join(list(tmp['candidate'].astype('string')))] * len(tmp)
        grouped_df.append(tmp)
    
    df = pd.concat(grouped_df)
    
    #df = df[df['percentvote'] >= 0.15]
    #df = df[df['percentvote'] <= 0.85]
    return df

def merge_news_elections(election_news, elections_source):
    elections_source['headlines'] = [""] * len(elections_source)
    elections_source['file'] = [""] * len(elections_source)
    elections_source = elections_source.reset_index(drop=True)
    fails = []
    
    for df in election_news:
        candidate = df.loc[0,'candidate']
        file_c = df.loc[0,'file']
        state = df.loc[0, 'state']
        date = file_c.split('_')[-1][:-4] #drop .ZIP
        year = int(date[-4:])    #grab last 4 of date
        headlines = list(set([preprocess_headlines(text) for text in df['Headline']]))
        bagofwords = '    '.join(headlines)
        e_idx = race_matching(candidate, year, state, elections_source, headlines)
        if e_idx != -1:
            elections_source.at[e_idx,'headlines']  = bagofwords
            elections_source.at[e_idx,'file']       = file_c
        else:
            fails.append(df)
    print('FAILS',len(fails))
    return elections_source, fails

def model_initializer():
    n_estimators = ""
    learning_rate= ""
    pass

def train_test_model():#fit, predict, score
    pass

def vocabulary(df):
    all_docs = df['headlines']
    pass
