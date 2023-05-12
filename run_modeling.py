import pandas as pd
import pickle

import model as m
from bert_retrain import train_bert, tokenize_from_end
from BOW import bow_plus, train_w_BOW
from data_manipulation import preproc, balanced_train_dev_test
from skorchers import skorch_net_maker, parameters_setup
from save_load import save

from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV


device ='cuda'
num_epochs = 16

if __name__ == "__main__":
    LOAD_TDS = True # LOAD IN TRAIN DEV TEST SET PREMADE FOR CONSISTENCY

    if not LOAD_TDS:
        data = pd.read_csv('NEWfull_with_headlines.csv', encoding='utf8')

        data = preproc(data)
        data_bow = bow_plus(data)
        print()

        #..... train dev test split proper
        print('train/dev/test splitting base data')
        train, dev, test            = balanced_train_dev_test(data, 0.70, dev_perc=0.15)
        train.to_csv('models/train.csv')
        dev.to_csv('models/dev.csv')
        test.to_csv('models/test.csv')
        print()

        print('train/dev/test splitting with BOW features')
        trainBOW, devBOW, testBOW   = balanced_train_dev_test(data_bow, 0.70, dev_perc=0.15)
        print()
    
    else:
        train = pd.read_csv('models/train.csv').reset_index(drop=True)
        dev =pd.read_csv('models/dev.csv').reset_index(drop=True)
        test = pd.read_csv('models/test.csv').reset_index(drop=True)
    
    print('--------------train/test', 'bagofword ML model')
    BOWmodel = LogisticRegressionCV(cv=7,n_jobs=2, max_iter=250, solver='sag')   
    BOWmodel = train_w_BOW(trainBOW,devBOW,testBOW, BOWmodel)

    with open('models/BOW_SVC.pkl','wb') as f:
        pickle.dump(BOWmodel,f)
    print()
    
    print('----------- train test on BERT model')
    m_o = train_bert(train, dev, test, epochs=num_epochs)
    s = save(m_o['model'],m_o['optimizer'], model_file='models/retrained.bert')
    print('saved model?:',s)
    print()
    
    for (model, llm_choice) in [('lstm','bow'),
                                ('lstm','distilbert-base-uncased'), 
                                ('lstm','bert-base-uncased'), 
                                ('lstm','openai-gpt'),
                                ('lstm','gpt2'), 
                                #('cnn','bow'),
                                #('cnn','distilbert-base-uncased'), 
                                #('cnn','bert-base-uncased'),
                                #('cnn','openai-gpt'), 
                                #('cnn','gpt2'),
                                ]:

        if llm_choice == "bow":
            T = trainBOW
            D = devBOW
            S = testBOW
        else:
            T = train
            D = dev
            S = test

        print("---------NN-EXPERIMENT", model, llm_choice, sep='\t')
        
        (llmtoken,  llmmodel, 
                    max_embed,
                    hidden_layers,
                    lr, 
                    dr, 
                    train_DS,
                    dev_DS,
                    test_DS)    = parameters_setup(model, 
                                                   llm_choice,
                                                   T,
                                                   D,
                                                   S)
        
        net = skorch_net_maker(model,
                               llmtoken, 
                               llmmodel,
                               llm_choice,
                               num_epochs, 
                               max_embed, 
                               hidden_layers,
                               lr, dr, device,
                               dev_DS)

        net.fit(train_DS, y=None)

        print('training done, reloading best checkpoint...')
        f_path = 'best_performers/'+model.upper()+'_FINAL_'+llm_choice
        net.initialize()
        net.load_params(
            f_params=f_path+"_model.pt", f_optimizer=f_path+"_opt.pt", f_history=f_path+'history.json')

        #   Check the load's accuracy, get a report
        y_pred = net.predict(test_DS)
        y_true = [int(s[1]) for s in test_DS]

        print()
        print(list(y_pred[:15]))
        print(y_true[:15])
        print("-------- TEST CLASSIFICATION REPORT --------------")
        print(classification_report(list(y_true), 
                                    y_pred,
                                    zero_division=  True))
    