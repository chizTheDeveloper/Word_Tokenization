**Part 1: Text Analysis of News Articles**

I downloaded the "articles1.csv" from Kaggle's dataset, the articles.csv contained mostly text data. 
I choose to use NLTK (Natural Language Toolkit)  as my tokenization library due to ease of use for tokenization. 
Since I would be tokenizing a news article I made sure to convert all text to lowercase during tokenization to ensure uniformity in the representation of words.

**The first 10 lines of the tokenizer's output for the whole corpus.**

['WASHINGTON', '—', 'Congressional', 'Republicans', 'have', 'a', 'new', 'fear', 'when', 'it']

**The total number of tokens and types (unique tokens) in the corpus.**

Total number of tokens: 38222245
Total number of types: 226908

**The type/token ratio for the corpus.**

Token Ratio: 0.005936542973862472

**A list of the top 3 most frequent tokens, along with their frequencies.
Top 3 Most Frequent Tokens:**

 ,      - 1859063
the - 1662375
.       - 1457522

**The number of tokens that appeared only once in the corpus.**

Number of Tokens Appeared Only Once: 89887

Top 3 Most Frequent Words: 
'The' - 1873736

 ‘To'   - 891302
 
  'Of'   - 810486

**The lexical diversity (type/token ratio) when using only words.**

Lexical Diversity: 0.0052486779677832635

**A list of the top 3 most frequent words (excluding stopwords and punctuation),
along with their frequencies**


Top 3 Most Frequent Words (excluding stopwords): 

'Said'       -  207527

'Trump'  -  149726

'People'   -  77330

 The lexical density (type/token ratio when using only word tokens without stopwords) 
lexical_density  : 0.00957795115795021

 A list of the most frequent 3 bigrams (excluding stopwords and punctuation) and their frequencies 
Top 3 Most Frequent Bigrams  

('of', 'the') -  193629

('in', 'the') -  159726  

('to', 'the') -   89586


**Part 2: Word Representation**

●	I imported my Amazon.csv data into Google Colab and installed the libraries I would need for preprocessing.

●	I imported the NLTK library, which allows me to apply stemming via the PorterStemmer function and lemmatization via WordNetLemmatizer.
●	 I also used NLTK for stopword removal and imported my data handling library as pandas. 

●	I would be evaluating the model being built I made sure to import the accuracy score and confusion matrix(to compare model confidence). 

●	I would be using the Logistic Regression (LR) and Support Vector Machine (SVM) models, so I imported my Sklearn libraries.

●	I imported and checked the size of the data which consisted of 568454 instances and 10 attributes. Due to processing issues, I was only able to work with 10,000 of the Amazon review data due to constant crashes and training time.

●	The data was split into a training set (80%) and a test set (20%), just a generally good split since no data imbalance was present from the Kaggle overview. 

●	I created a pre-process function that converted all text to lowercase, removed punctuation, applied “word_tokenize” to the text, removed stop words, processed stemming on new word vocabulary and lemmatized it.

●	Applied the pre-process function to the train and test data

●	Performed feature extraction via Bag of Words (BOW) 


**confusion matrix for all the models.**

BOW
Logistic Regression Accuracy for BOW: 0.6705
Logistic Regression Confusion Matrix for BOW:

 [[  89    9    3    6   89]
 
 [  22    9   12   12   40]
 
 [  18   10   22   32   84]
 
 [   4    6   20   64  219]
 
 [  13    5   20   35 1157]]

Support Vector Machine Accuracy for BOW: 0.653
Support Vector Machine Confusion Matrix for BOW:
 [[ 101   15    4    6   70]
 [  25   14   12   11   33]
 [  23   17   22   20   84]
 [   7   11   20   49  226]
 [  22   17   35   36 1120]]

TDIF
Logistic Regression Accuracy for TDIF: 0.6655
Logistic Regression Confusion Matrix for TDIF:
 [[  69    4    0    4  119]
 [  13    2    4   13   63]
 [  11    0    7   29  119]
 [   1    1    7   44  260]
 [   4    0    4   13 1209]]


SVM Accuracy for TDIF: 0.6765
SVM Confusion Matrix for TDIF:
 [[ 100    3    2    3   88]
 [  21    2    7    7   58]
 [  16    5   12   17  116]
 [   4    2    5   34  268]
 [   7    0    3   15 1205]]


Ngram
LR Accuracy for n-gram: 0.6805
LR Confusion Matrix for n-gram:
 [[  97   12    3    5   79]
 [  25   11   12   12   35]
 [  15    7   23   32   89]
 [   3    6   16   71  217]
 [  15    6   16   34 1159]]

Support Vector Machine Accuracy for n-gram: 0.6635
Support Vector Machine Confusion Matrix for n-gram:
 [[ 100   15    3    3   75]
 [  24   16   12    9   34]
 [  20   16   23   26   81]
 [   5    7   23   63  215]
 [  21    8   27   49 1125]]

** interpretation and discussion of the results.
**
The Matrix looks weird to me, I'm used to a 2-by-2 matrix. I was thinking of applying PCA to reduce it 2x2 so I can easily tell the confidence of True, False, Positive and Negatives.
So far LR works best for N-gram and SVM works best on TF-IDF. If I used the full dataset the result might be different.
I just used grid search for hyperparameter tuning as it searches through a predefined grid of hyperparameter values, evaluating the model's performance for each combination through cross-validation and selecting the best one


 best accuracy for all the used methods:**
**
Method 	LR 	 SVM
BOW 	67%	65%
n-gram	68%	66%
TF-IDF	66%	67%



