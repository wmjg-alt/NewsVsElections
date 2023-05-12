import model as m
import torch
import torch.nn as nn
import skorch as sk

from torch.utils.data import DataLoader
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from transformers import BertTokenizer, DistilBertTokenizer, OpenAIGPTTokenizer,GPT2Tokenizer
from transformers import BertModel, DistilBertModel, OpenAIGPTModel,  GPT2Model

from DatasetIter import ADataset


def skorch_net_maker(model_name:str,
                     llmtoken, 
                     llmmodel, 
                     llm_choice:str, 
                     num_epochs:int, 
                     max_embed:int, 
                     hidden, 
                     lr:float, dr:float, 
                     device:str, 
                     dev_DS):
    ''' build a skorch net from all the parameters customized in params_setup'''
    #select classifier
    if llm_choice == "bow":
        classifier = m.CNNTextClassifier if model_name == 'cnn' else m.LSTMTextClassifier
    else:
        classifier = m.LLMCNNTextClassifier if model_name == 'cnn' else m.LLMLSTMTextClassifier
    
    # Define some metrics
    f1_micro = EpochScoring("f1_micro", lower_is_better=False,)
    f1_macro = EpochScoring("f1_macro", lower_is_better=False,)

    # BUILD the skorch net form all the parameter preamble
    # with lots of reporting metrics and checkpoints
    net = sk.NeuralNetClassifier(
                classifier,
                module__llm = llmmodel,
                module__num_classes=2,
                module__vocab_size=10001,
                module__hidden_layer_sizes=hidden,
                module__embedding_dim=768 if llmmodel else 10000,
                lr=lr,
                module__dr=dr,
                optimizer=torch.optim.SGD if llm_choice not in ['bow'] else torch.optim.AdamW,
                optimizer__weight_decay=0.01,
                max_epochs=num_epochs,
                criterion=nn.CrossEntropyLoss,
                batch_size=4 if model_name != "gpt2" else 2,
                device=device,
                iterator_train=DataLoader,
                iterator_valid=DataLoader,
                train_split=predefined_split(dev_DS),
                verbose=True,
                callbacks=[ f1_micro, 
                            f1_macro,
                            sk.callbacks.Checkpoint(monitor='valid_acc_best',
                                                    fn_prefix=model_name.upper()+'_FINAL_'+llm_choice,
                                                    dirname='best_performers',
                                                    f_params="_model.pt",
                                                    f_optimizer="_opt.pt",
                                                    f_history='history.json')
                ],
    )
    return net


def parameters_setup(model_name:str, llm_choice:str, T, D, S):
    # very custome parameter selection
    # model_name: lstm, cnn
    # llm_choice: 'bow', 'bert-base-uncased' as below
    # handle separately with match
    # T: train set
    # D: dev set
    # S: test set
    # return bundle of parameters and DATASETS of T,D,S

    match llm_choice:
        case 'bow':
            llmtoken=None
            llmmodel=None
            max_embed=10000
            lr = 0.00033    #lock
            dr = 0.3
        case 'gpt2':
            llmtoken = GPT2Tokenizer.from_pretrained(llm_choice)
            llmmodel = GPT2Model.from_pretrained(llm_choice,)
            max_embed = 500 #1000 broke my GPU
            lr = 0.00033       #lock
            dr = 0.2
        case 'openai-gpt':
            llmtoken = OpenAIGPTTokenizer.from_pretrained(llm_choice)
            llmmodel = OpenAIGPTModel.from_pretrained(llm_choice,)
            max_embed = 500
            lr = 0.0045     #lock
            dr = 0.2
        case 'bert-base-uncased':
            llmtoken = BertTokenizer.from_pretrained(llm_choice)
            llmmodel = BertModel.from_pretrained(llm_choice,)
            max_embed = 500
            lr = 0.0005    #lock
            dr = 0.25
        case 'distilbert-base-uncased':
            llmtoken = DistilBertTokenizer.from_pretrained(llm_choice)
            llmmodel = DistilBertModel.from_pretrained(llm_choice,)
            max_embed = 500
            lr = 0.003      #lock
            dr = 0.3
        case _:
            raise Exception("need to choose an llm")
        
    if llmtoken and llmtoken.pad_token is None:
        llmtoken.add_special_tokens({'pad_token': '[PAD]', 'unk_token':'[UNK]'})
        llmmodel.resize_token_embeddings(len(llmtoken))

    train_DS    = ADataset(T[:], llmtoken=llmtoken, max_embed=max_embed)
    dev_DS      = ADataset(D, llmtoken=llmtoken, max_embed=max_embed)
    test_DS     = ADataset(S, llmtoken=llmtoken, max_embed=max_embed)

    hidden_layers  = [4,3,2] if model_name=='cnn' else [350]

    return llmtoken, llmmodel, max_embed, hidden_layers,lr, dr, train_DS, dev_DS, test_DS