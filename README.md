#ext Analysis of News Articles 
This section delves into the analysis of news articles extracted from the "articles1.csv" dataset sourced from Kaggle. Utilizing NLTK (Natural Language Toolkit) for tokenization, the text was standardized to lowercase for consistency.

**Tokenization Results**  
Total Tokens: 38,222,245  
Total Types (Unique Tokens): 226,908  
Token Ratio: 0.0059  
Top 3 Most Frequent Tokens:  
(',', '-', '.')  
Tokens Appeared Only Once: 89,887  
Top 3 Most Frequent Words (excluding stopwords):  
'Said'  
'Trump'  
'People'  
Lexical Diversity: 0.0052  
Top 3 Most Frequent Bigrams:  
('of', 'the')  
('in', 'the')  
('to', 'the')  


**Word Representation**  
Preprocessing and Model Building  
Imported the Amazon.csv data into Google Colab for preprocessing.  
Utilized NLTK for stemming, lemmatization, and stopword removal.  
Implemented Logistic Regression (LR) and Support Vector Machine (SVM) models.  
Employed Bag of Words (BOW) for feature extraction.  
Split the data into a training set (80%) and a test set (20%).  

**Model Evaluation**  
Confusion Matrix Results:  
BOW LR Accuracy: 67.05%  
BOW SVM Accuracy: 65.3%  
TF-IDF LR Accuracy: 66.55%  
TF-IDF SVM Accuracy: 67.65%  
N-gram LR Accuracy: 68.05%  
N-gram SVM Accuracy: 66.35%  

**Interpretation and Discussion**  
The confusion matrices offer insights into model performance.  
LR performs optimally for N-gram, while SVM demonstrates superiority with TF-IDF.  
The utilization of the full dataset might yield altered results.  
Hyperparameter tuning was conducted using grid search.  
Best Accuracy  
LR:  
BOW (67%)  
N-gram (68%)  
TF-IDF (66.55%)  
SVM:  
BOW (65.3%)  
N-gram (66.35%)  
TF-IDF (67.65%)  
