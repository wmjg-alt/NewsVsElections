from zipfile import ZipFile
import re
import pandas as pd
from data_manipulation import candidate_list_split, masking_names, fill_opponent

head_split = '    '

#HELPERS
def read_zip(zip_fn, extract_fn=None):
    # NewsData is directory of zipped files
    # read then with from zip_fn file name
    # return list of read files in zipped file
    zf = ZipFile(zip_fn)
    if extract_fn:
        return zf.read(extract_fn)
    else:
        return [zf.read(name) for name in zf.namelist()][0]


def preprocess_headlines(text):
    #regex norm headlines
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    text = re.sub(r'\d+', '', text)
    return text


def name_norm(n):
    #regex norm names
    return re.sub(r'[^A-Za-z0-9 ]+','',n)


def race_matching(candidate0, year, state, source):
    # Race: a candidate, in year, in state
    # match to the ELECTION info in source df
    # pass to candidate_matcher and return

    def candidate_matcher(c, y, src,  state, col):
        # find the target sub df (president or other)
        # with c=candidate, y=year, src =election source, state=state, 
        # col= column to check for candidate in
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
    

def fetch_preprocess_targets(f, dated):
    # Bring in and clean target df 
    # adding columns for text cleanup, percentage of total vote, and more
    # dropping extraneous races and columns

    def perc(row):
        # candidate's votes / total votes
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
                  'fusion_ticket'], axis=1, errors='ignore') 
    
    df = df[~(df['year'] < 1999)]  

    df = df.sort_values(['year',
                         'state',
                         'district',
                         'totalvotes',
                         'percentvote'],ascending=False).groupby(['year',
                                                                  'state',
                                                                  'district',
                                                                  'totalvotes'])
    
    grouped_df = []
    # drop low performing 3rd candidates
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
    # FULL conjoin functionality;
    # election_news: list of dfs of Headline News
    # elections_source: Raw info about Elections
    # build BOW of headlines with SEP and build DF of elect info + NEWS
    # REPORT drops, failure, issues
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
        bagofwords = head_split.join(headlines)
        e_idx = race_matching(candidate, year, state, elections_source)
        if e_idx != -1:
            elections_source.at[e_idx,'headlines']  = bagofwords
            elections_source.at[e_idx,'file']       = file_c
        else:
            fails.append(df)
    print('FAILS',len(fails))
    return elections_source, fails

'''
def model_initializer():
    n_estimators = ""
    learning_rate= ""
    pass

def train_test_model():#fit, predict, score
    pass

def vocabulary(df):
    all_docs = df['headlines']
    pass
'''
