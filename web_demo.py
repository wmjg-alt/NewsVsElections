import pandas as pd
import random
import skorch as sk

from flask import Flask, render_template
from collections import Counter
from nltk.corpus import stopwords

from DatasetIter import ADataset
from skorchers import skorch_net_maker, parameters_setup

sw = stopwords.words('english')

app = Flask(__name__)

# best performer
llm_best = "bert-base-uncased"
model = 'LSTM'

#load in test/dev/train
T = pd.read_csv('models/train.csv').reset_index(drop=True)
D = pd.read_csv('models/dev.csv').reset_index(drop=True)
S = pd.read_csv('models/test.csv').reset_index(drop=True)

data = pd.read_csv('NEWfull_with_headlines.csv', encoding='utf8')

num_epochs =6
device='cuda'

# skorch parameters
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

# init net with parameters
net = skorch_net_maker(model,
                        llmtoken, 
                        llmmodel,
                        llm_best,
                        num_epochs, 
                        max_embed, 
                        hidden_layers,
                        lr, dr, device,
                        dev_DS)

''' overwrite params with the best performing model '''
f_path = 'best_performers/'+"LSTM"+'_FINAL_'+llm_best
net.initialize()
net.load_params(
    f_params=f_path+"_model.pt", f_optimizer=f_path+"_opt.pt", f_history=f_path+'history.json')

#predict the test set live
preds = net.predict(test_DS)
S['pred'] = list(preds)

@app.route('/showdown')
def showdown():
    # meant to implement a Trump vs Biden "if the election were today"
    # Have the Data for 4/11/23 - 5/11/23, but ran out of time
    return

@app.route('/')
def home():
    ''' flask home page, random sample the predicted test set; 
        render test df and term Counter
    '''
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
    # host flask app locally
    app.run(host='0.0.0.0')