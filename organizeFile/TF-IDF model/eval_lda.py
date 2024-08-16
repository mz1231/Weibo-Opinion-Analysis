import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataset_utils import load_and_split_dataset, load_and_upsample_dataset
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model and vectorizer
model_filename = 'trained_lda_model.pkl'
vectorizer_filename = 'lda_vectorizer.pkl'

lda = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

json_file_path = 'datasets/sequence-dataset-q2.json'

# Load and split the dataset
dataset, test_df, train_df = load_and_upsample_dataset(json_file_path)

X_test = test_df['text']
y_test = test_df['label']

X_test_tfidf = vectorizer.transform(X_test)

# Example new test data
new_test_texts = ['干脆别有信访办啊，反正大家都在微博举身份证', '#山东高院工作报告#', '#北京交通管制#北京交通管制真是没有下限了，一个多小时了还不解除。发声平台这么多以后能不能提前通知？不能通知的话能不能让领导快走两步？', '我不喜欢总统', '中国的GDP下降得太低了。我们需要更多的投资', '干脆别有信访办啊，反正大家都在微博举身份证']

# Vectorize the new test text data using the loaded vectorizer
new_test_tfidf = vectorizer.transform(new_test_texts)

# Make predictions on the new test data
new_test_predictions = lda.predict(X_test_tfidf.toarray())
new_test_probabilities = lda.predict_proba(X_test_tfidf.toarray())

# Output the predictions
for text, prediction, probabilities in zip(X_test, new_test_predictions, new_test_probabilities):
    print(f'Text: {text}\nPrediction: {prediction}\nProbabilities: {probabilities}\n')

# Calculate and print the accuracy
test_accuracy = accuracy_score(y_test, new_test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Print classification report
print("Test Classification Report:")
print(classification_report(y_test, new_test_predictions))
