import torch
from torch.utils.data import IterableDataset, TensorDataset
from bert_retrain import tokenize_from_end

class ADataset(IterableDataset):
    def __init__(self, dat, llmtoken=None, max_embed=510, shuffle=None):
        if llmtoken:
            dat = dat.drop(columns=[col for col in dat.columns if col not in ['headlines', 'winner']])
        self.data = dat
        self.llmtoken = llmtoken
        self.max_embed = max_embed
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if self.llmtoken:
            f = tokenize_from_end(self.data.loc[idx,'headlines'], self.llmtoken, max_embed=self.max_embed)['input_ids']
            f = f['input_ids'].squeeze()
        else:
            bow = self.data.loc[idx,self.data.columns != 'winner']
            bow = bow.astype(float)
            f = torch.tensor(bow.values).float()
        l = torch.tensor(self.data.loc[idx, 'winner'])
        return f,l
    
    def __iter__(self):
        for idx,d in self.data.iterrows():
            dd = {}
            if self.llmtoken:
                dd['features'] = tokenize_from_end(d['headlines'], self.llmtoken, self.max_embed)['input_ids'].squeeze()
            else:
                bow = d.loc[self.data.columns != 'winner']
                bow = bow.astype(float)
                dd['features'] = torch.tensor(bow.values).float()
                #print(dd['features'].shape, dd['features'].max(), dd['features'].min())
            dd['label'] = torch.tensor(int(d['winner']))
            yield dd['features'], dd['label']
