from flask import Flask, render_template
import pandas as pd
import random
from DatasetIter import ADataset
import skorch as sk
from skorchers import skorch_net_maker, parameters_setup
from collections import Counter
from nltk.corpus import stopwords

sw = stopwords.words('english')

app = Flask(__name__)

llm_best = "distilbert-base-uncased"
model = 'LSTM'
T = pd.read_csv('models/train.csv').reset_index(drop=True)
D = pd.read_csv('models/dev.csv').reset_index(drop=True)
S = pd.read_csv('models/test.csv').reset_index(drop=True)

data = pd.read_csv('NEWfull_with_headlines.csv', encoding='utf8')

num_epochs =6
device='cuda'

(llmtoken,  llmmodel, 
            max_embed,
            hidden_layers,
            lr, 
            dr, 
            train_DS,
            dev_DS,
            test_DS)    = parameters_setup(model, 
                                            llm_best,
                                            T,
                                            D,
                                            S)

net = skorch_net_maker(model,
                        llmtoken, 
                        llmmodel,
                        llm_best,
                        num_epochs, 
                        max_embed, 
                        hidden_layers,
                        lr, dr, device,
                        dev_DS)

f_path = 'best_performers/'+"LSTM"+'_FINAL_'+llm_best
net.initialize()
net.load_params(
    f_params=f_path+"_model.pt", f_optimizer=f_path+"_opt.pt", f_history=f_path+'history.json')
preds = net.predict(test_DS)
S['pred'] = list(preds)

@app.route('/showdown')
def showdown():
    return

@app.route('/')
def home():
    example = S.sample(1)
    idx = S.first_valid_index()
    example = example.drop(columns=[col for col in example.columns if col not in ['year',
                                                                          'state',
                                                                          'office',
                                                                          'party',
                                                                          'candidate',
                                                                          'opponents',
                                                                          'percent_vote',
                                                                          'headlines',
                                                                          'winner',
                                                                          'pred']])
    ct = Counter([wd for wd in list(example['headlines'].str.split())[0] if wd not in sw])
    cdf = pd.DataFrame.from_dict(ct, orient='index').reset_index()
    return render_template('home.html',  tables=[example.to_html(classes='data', header="true"), 
                                                cdf.to_html(classes='data', header="true")])


if __name__ == "__main__":
    del net
    app.run(host='0.0.0.0')