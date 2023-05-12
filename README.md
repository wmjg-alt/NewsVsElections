# NEWS VS ELECTIONS
    A Project Predicting the Outcomes of Elections from News Headlines

    If the following are True:
        - News Media are Opinion Leaders
        - People read Headlines, not articles
    It must follow that Headlines themselves are informative in the outcome of Elections.

    So we have a 2 class problem: a month of headlines predicting winners (1) and losers (0)

## Outline:
- Data on results of elections was sourced from MIT Election Lab (/Elections)
        
        * Senate    races for 2000-2020
        * House     races for 2000-2020
        * President races for 2000-2020
    
- The Data_Ingest functionality is for controlling a selenium browser to access a certain data source with auth credentials
        
        * Searching for Candidate Elections from the MIT data
            * Web Scraping up to 1000 Headlines per candidate per election
            * For the month leading up to election date
            * Only results marked "Government & Public Administration"

        * Building a corpus of Headlines associated with political candidates by race by year
            * Corpus unavailable to public
        
        * ingest_news tools no longer functional due to lockout
    
- The ipynb data_collate_prep demonstrates how the 5000 files were prepped and processed
        
        * Using pandas to associate Candidate-Elections with their headlines in a big dataframe
        * Output in NEWfull_with_headlines.csv  (also unavailable)
            * about 4600 candidate elections with close to 200k headlines
        * A simple estimator for naively selecting a model from sklearn 
            * (LogisticRegressionCV selected)
        
    - scrapeRefine contains functionality for merging Elections with NewsData

            * regex normalizing candidates, headlines, more
            * matching dates and states and names
            * dropping corrupt/incomplete/erroneous data

    - Data_manipulation contains functionality for 
            
            * masking out candidates' and their opponents' names in the data set
            * And building datasets with balanced data represenations (no bias for winners/losers, etc)

    - statfile jupyter notebook demos balances and imbalances in the dataset

- Naive jupyter notebook demos a naive model

        * Predicting based on raw number of headlines
        * Less than 60% accuracy
        * Target to beat

- THE TASK: 3 Big Experiments
    - Bag of Words TFIDF Logistic Regression (tests of SVC also present)
    - Custom LSTM and CNN with LLM Embeddings
    - Retraining an LLM (BERT-base-cased)

## BAG OF WORDS

    BOW.py creates a bag of words out of headlines

        * 10000 vocab vector
        * tdidf vectorized
        * with Party affiliation as one extra feature

## CUSTOM EMBEDDED MODELS
    
    Skorchers.py contains functionality for:

        * implementing a cross functional selection of hugging face models and tokenizers
        * experimentally selected hyperparameters
        * returning a skorcher net with all the features for 10 experiments
            * the Bag of Words, bert-base-uncased, distilbert-base-uncased, openai-gpt, gpt2 embeddings
            * used in the custom LSTM or CNN
    
    model.py contains the LSTM and CNN models (torch.nn)

        * LSTM
            * dropout, Bidirectional 3-layer LSTM, with Max pooling
        * CNN
            * Conv1d blocks of dropout, 100 filters of 4, 3, and 2 widths with Max pooling

    DatasetIter.py implements a Dataset class

        * serving up the tokenized headlines and labels to Dataloaders

## RETRAINING BERT

    bert_retrain.py contains all the functionality for the LLM retraining

        * A different Classification Dataset serving up tokenized ids, attention masks, labels
        * Custom BERT module with training and evaluation loops
    
    save_load.py
        
        * a wrapper of torch's save and load functionality

# MAINLY
- with the data in Elections/ NewsData/
- pip install -r requirements.txt
- python run_modeling.py

It all takes place in run_modeling.py; sets up and calls all experiments
train/dev/test output to models/
best performing models saved in models/ and best_performers/


|       MODEL              |                  |   acc      |
|--------------------------|------------------|------------|
|  LogisticRegressionCV    |     BOW          |      0.75  |
|  BERT Retraining         |                  | 0.74       |
|   LSTM                   |    BOW           |   0.74     |
|   LSTM                   |      distilbert  |      0.74  |
|   **LSTM**                   |   **BERT**           |     **0.76**   |**
|   LSTM                   |    GPT           |     0.73   |
|   LSTM                   |    GPT2          |     0.68   |

NOTE: CNN deemed worse than LSTM without many epochs, results not consistent with rest

NOTE: GPT was not given sufficient training time/best modelling due to hardware limitations



## A mini demo of the predictions on the test set can be run on FLASK with web_demo.py
