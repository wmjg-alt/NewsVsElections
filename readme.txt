NEWS VS ELECTIONS

Predicting Election Outcomes with News Data (HEADLINES)

With the 2022 Midterms oncoming, the goal was to define an NLP task which predicted Elections from News. In other terms: What role does News play in Election outcomes? The argument goes that if major news sources are, in fact, Opinion Leaders in the modern American political landscape AND the average news reader typically skims articles or just read headlines, then Headlines alone should be predictive of political outcomes.

ingest_news.ipynb -- uses a credentials file to web scrape News Headlines for Election candidates, by election date for the month before the election

naive.ipynb -- uses sklearn models and predicts election winners and losers based only on the number of headlines a candidate received in the month before the election

BOW_Classifiers.ipynb -- puts the Headlines to work as a bag of words (with candidate names masked) and featurizes the bags to the top 10k vocabulary terms before training and testing many sklearn models

statfile.ipynb -- does some data viz for some questions of bias in the election data republican vs democrat win numbers.


naive's best hit 0.6 accuracy -- target to beat
an SVM in BOW_Classifiers hit 0.8 accuracy

The hypothesis that news sources are Opinion Leaders seems to hold water, because we not only reached 80% accuracy in our test set, but we beat the raw How-much-news classifier. That means, it’s not THAT a candidate is talked about; what was said matters significantly more.
