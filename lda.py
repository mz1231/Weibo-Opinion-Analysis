import jieba
import re
import gensim
import pandas as pd
from gensim import corpora
from sklearn.utils import resample
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from TCSP import read_stopwords_list
from dataset_utils import load_and_split_dataset, load_and_upsample_dataset, upsample_dataset_no_val

stopwords_list = read_stopwords_list()
json_file_path = 'datasets/big-sequence-dataset-q2.json'  # Replace with your JSON file path

# Load and split the dataset
dataset, test_df, train_df = upsample_dataset_no_val(json_file_path)

# Extract the text data from the training dataset
train_texts = train_df['text'].tolist()  # Assuming 'text' column contains the text data

# Chinese stopwords list
stopwords = set(stopwords_list)
print(stopwords)

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation and non-Chinese characters
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    # Tokenize
    tokens = jieba.lcut(text)
    # Remove stopwords and short tokens
    tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
    return tokens

# Apply preprocessing
processed_data = [preprocess_text(text) for text in train_texts]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(processed_data)

# Convert documents to Bag-of-Words format
corpus = [dictionary.doc2bow(text) for text in processed_data]

# Set number of topics (you can adjust this based on your data)
num_topics = 4

# Train the LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")

# Manually assign labels to topics based on the top words
topic_labels = {
    0: 'government',
    1: 'economy',
    2: 'personal_wellbeing',
    3: 'none'
}

# Function to classify a new document
def classify_document(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Convert to Bag-of-Words format
    bow = dictionary.doc2bow(processed_text)
    # Get topic distribution
    topics = lda_model.get_document_topics(bow)
    # Find the most dominant topic
    dominant_topic = max(topics, key=lambda item: item[1])[0]
    # Return the label of the dominant topic
    return topic_labels[dominant_topic]

# Classify each document in your dataset
classified_results = [classify_document(text) for text in train_texts]
print(classified_results)

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report
# from dataset_utils import load_and_split_dataset, load_and_upsample_dataset, upsample_dataset_no_val

# def train_lda(train_df, test_df):
#     # Extracting text and labels
#     X_train = train_df['text']
#     y_train = train_df['label']
#     X_test = test_df['text']
#     y_test = test_df['label']

#     # Create a pipeline
#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000)),  # Convert text to TF-IDF features
#         ('lda', LatentDirichletAllocation(n_components=20, random_state=42)),  # LDA with 10 topics
#         ('clf', LogisticRegression())  # Logistic Regression as the classifier
#     ])

#     # Train the model
#     pipeline.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = pipeline.predict(X_test)

#     # Evaluate the model
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))

# # Main script
# if __name__ == "__main__":
#     json_file_path = 'datasets/big-sequence-dataset-q2.json'  # Replace with your JSON file path

#     # Load and split the dataset
#     dataset, test_df, train_df = upsample_dataset_no_val(json_file_path)

#     # Train and evaluate the model
#     train_lda(train_df, test_df)