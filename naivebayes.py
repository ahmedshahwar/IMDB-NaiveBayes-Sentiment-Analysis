import numpy as np
import pandas as pd
from nltk.corpus import stopwords  # For stopwords removal
from collections import defaultdict  # For efficient dictionary creation
import re  # For punctuation removal

# from sklearn.model_selection import train_test_split  # For train-test split
# from sklearn.metrics import accuracy_score, recall_score, precision_score  # For evaluation metrics

# --------------------------------------------------------------------- # 
#               Implementing Naive Bayes Classification                 #  
# --------------------------------------------------------------------- #

class NaiveBayesBinary:
  """
  This class implements a Naive Bayes classifier for binary classification.
  """
  def __init__(self):
    """
    Initialize the classifier with empty dictionaries for class probabilities and word probabilities.
    """
    self.class_prob = {}  # Dictionary to store class probabilities
    self.word_prob = defaultdict(lambda: defaultdict(int))  # Nested dictionary to store word probabilities for each class

  def preprocessing(self, text):
    """
    Preprocess text data by converting to lowercase, removing stopwords, and removing non-alphanumeric characters.

    Args:
      text: The text to preprocess.

    Returns:
      The preprocessed text.
    """
    text = text.lower() # Convert each letter into lowercase
    stop_words = stopwords.words('english')  # Load stopwords from nltk corpus
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters (punctuations)
    return text

  def fit(self, X, y):
    """
    Train the Naive Bayes classifier on the given data.

    Args:
      X: A pandas Series containing the training data (text).
      y: A pandas Series containing the class labels (positive or negative).
    """
    # Calculate class probabilities
    classes, counts = np.unique(y, return_counts=True)  # Get unique classes and their counts
    total_samples = len(y)
    for c, count in zip(classes, counts):
      self.class_prob[c] = count / total_samples  # Calculate probability of each class

    # Calculate word probabilities
    for i in range(len(X)):
      text = self.preprocessing(X.iloc[i])  # Preprocess text
      words = text.split()
      for word in words: # Count word occurrences for each class
        self.word_prob[word][y.iloc[i]] += 1  

    # Convert counts to probabilities with Laplace Smoothing
    for word in self.word_prob:
      total_count = sum(self.word_prob[word].values())
      for c in self.word_prob[word]:
        self.word_prob[word][c] /= (total_count + 1)  # Apply Laplace Smoothing to avoid zero probabilities

  def predict(self, X):
    """
    Predict the class labels for new data points.

    Args:
      X: A list of text data points.

    Returns:
      A list of predicted class labels.
    """
    predictions = []
    for text in X:
      text = self.preprocessing(text)  # Preprocess text
      words = text.split()
      class_scores = {c: np.log(self.class_prob[c]) for c in self.class_prob}  # Initialize class scores with log probabilities

      for word in words:
        if word in self.word_prob:  # Check if word exists in vocabulary
          for c in self.class_prob:
            class_scores[c] += np.log(self.word_prob[word][c] + 1)  # Apply Laplace Smoothing and add log probabilities

      predicted_class = max(class_scores, key=class_scores.get)  # Get the class with the highest score
      predictions.append(predicted_class)
    return predictions

"""

# --------------------------------------------------------------------- # 
#                       Loading and Dividing Dataset                    #  
# --------------------------------------------------------------------- #

df = pd.read_csv('IMDBDataset.csv')
X = df['review']
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# --------------------------------------------------------------------- #
#                 Training and Testing Model on Dataset                 #
# --------------------------------------------------------------------- #

NBClf = NaiveBayesBinary()
NBClf.fit(X_train, y_train)

pred = NBClf.predict(X_test)

accuracy = accuracy_score(y_test, pred) * 100
precisionPos = precision_score(y_test, pred, pos_label='positive') * 100
precisionNeg = precision_score(y_test, pred, pos_label='negative') * 100
recallPos = recall_score(y_test, pred, pos_label='positive') * 100
recallNeg = recall_score(y_test, pred, pos_label='negative') * 100

print('Accuracy Score: ', accuracy)
print('Precision +ve: ', precisionPos)
print('Precision -ve: ', precisionNeg)
print('Recall +ve: ', recallPos)
print('Recall -ve: ', recallNeg)

# --------------------------------------------------------------------- # 
#                     Testing Model on Custom Text                      #
# --------------------------------------------------------------------- #

text = "I didn't enjoy this film at all. It was boring and predictable."
predictions = NBClf.predict([text])
print(predictions)

"""