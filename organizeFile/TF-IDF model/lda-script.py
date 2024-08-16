import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from dataset_utils import load_and_split_dataset, load_and_upsample_dataset, upsample_dataset_no_val
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Function to train and evaluate a Linear Discriminant Analysis model
def train_and_evaluate_model_no_val(train_df, test_df):
    # Extracting text and labels
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    # Vectorizing the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Training the Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_tfidf.toarray(), y_train)

    # Evaluating the model on the test set
    test_predictions = lda.predict(X_test_tfidf.toarray())
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Classification Report:")
    print(classification_report(y_test, test_predictions))
    
    # Confusion Matrix for Test set
    test_conf_matrix = confusion_matrix(y_test, test_predictions)
    print("Test Confusion Matrix:")
    print(test_conf_matrix)
    sns.heatmap(test_conf_matrix, annot=True, fmt='d')
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Print misclassified examples
    print("Misclassified Examples:")
    for text, prediction, true_label in zip(X_test, test_predictions, y_test):
        if prediction != true_label:
            print(f"Text: {text}\nPredicted Label: {prediction}\nTrue Label: {true_label}\n")

    # Save the trained model and vectorizer
    model_filename = 'trained_lda_model.pkl'
    vectorizer_filename = 'lda_vectorizer.pkl'
    joblib.dump(lda, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

# Main script
if __name__ == "__main__":
    json_file_path = 'datasets/big-sequence-dataset-q2.json'  # Replace with your JSON file path

    # Load and split the dataset
    dataset, test_df, train_df = upsample_dataset_no_val(json_file_path)

    # Train and evaluate the model
    train_and_evaluate_model_no_val(train_df, test_df)

    # Save the trained model and vectorizer
    model_filename = 'trained_lda_model.pkl'
    vectorizer_filename = 'lda_vectorizer.pkl'

    # Load the trained model and vectorizer
    loaded_model = joblib.load(model_filename)
    loaded_vectorizer = joblib.load(vectorizer_filename)

