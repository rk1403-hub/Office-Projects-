import requests
from bs4 import BeautifulSoup
import nltk
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Define a class for training the version detection model
class VersionDetectorTrainer:
    def __init__(self, training_data_path):
        self.training_data_path = training_data_path
        self.model = None
        self.tfidf_vectorizer = None

    def load_data(self):
        # Load your training data from a CSV file with a specific encoding
        training_data = pd.read_csv(self.training_data_path, encoding="ISO-8859-1")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            training_data["Sentence"],
            training_data["label"],
            test_size=0.2,
            random_state=42,
        )

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.load_data()

        # Initialize a TF-IDF vectorizer with additional preprocessing
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)

        # Transform the sentences into TF-IDF features
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

        # Initialize and train the Logistic Regression model with tuned hyperparameters
        model = LogisticRegression(C=1.0)
        model.fit(X_train_tfidf, y_train)

        self.model = model
        self.tfidf_vectorizer = tfidf_vectorizer

    def save_model(self, model_filename="version_detection_model.pkl", vectorizer_filename="tfidf_vectorizer.pkl"):
        if self.model and self.tfidf_vectorizer:
            joblib.dump(self.model, model_filename)
            joblib.dump(self.tfidf_vectorizer, vectorizer_filename)
        else:
            print("Model or vectorizer not trained yet.")

# Define a class for predicting version numbers
class VersionNumberPredictor:
    def __init__(self, model_filename="version_detection_model.pkl", vectorizer_filename="tfidf_vectorizer.pkl"):
        self.model = joblib.load(model_filename)
        self.tfidf_vectorizer = joblib.load(vectorizer_filename)

    def predict_versions(self, visible_text):
        # Tokenize the visible text into sentences
        sentences = nltk.sent_tokenize(visible_text)

        # Transform the visible text sentences into TF-IDF features
        visible_text_tfidf = self.tfidf_vectorizer.transform(sentences)

        # Use the trained model to predict sentences in the visible text
        predictions = self.model.predict(visible_text_tfidf)

        # Return the sentences with version numbers based on predictions
        version_sentences = [sentence for sentence, prediction in zip(sentences, predictions) if prediction]

        return version_sentences

    @staticmethod
    def extract_version_numbers(sentence):
        words = sentence.split()
        version_candidates = []

        for word in words:
            if all(char.isdigit() or char == "." for char in word):
                version_candidates.append(word)

        return version_candidates

    @staticmethod
    def get_unique_sorted_versions(version_sentences):
        version_numbers_set = set()

        for sentence in version_sentences:
            version_numbers = VersionNumberPredictor.extract_version_numbers(sentence)
            cleaned_versions = [v.strip(".").strip() for v in version_numbers]  # Normalize and clean version numbers
            version_numbers_set.update(cleaned_versions)

        # Convert the set back to a sorted list
        sorted_versions = sorted(version_numbers_set, reverse=True)

        return sorted_versions

# Example usage:
if __name__ == "__main__":
    # Define the URL of the webpage you want to scrape
    url = "https://www.ruby-lang.org/en/downloads/#:~:text=The%20current%20stable%20version%20is%203.2.2."  # Replace with your URL

    # Send a GET request to the webpage
    response = requests.get(url)

    # Create a BeautifulSoup object with the webpage content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the <body> tag
    body = soup.find("body")

    # Define a list of tags that typically contain visible text
    visible_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]

    # Filter out tags based on their attributes to only keep visible text
    visible_text = [tag.get_text() for tag in body.find_all(visible_tags) if tag.get("style") != "display:none"]

    # Join the extracted text into a single string
    visible_text = " ".join(visible_text)

    # Initialize and train the model
    trainer = VersionDetectorTrainer("version_data.csv")
    trainer.train_model()
    trainer.save_model()

    # Use the trained model for prediction
    predictor = VersionNumberPredictor()
    version_sentences = predictor.predict_versions(visible_text)
    sorted_versions = predictor.get_unique_sorted_versions(version_sentences)

    # Print the sentences with version numbers (predicted)
    print("Sentences with version numbers (predicted):")
    for i, sentence in enumerate(version_sentences):
        print(f"{i+1}. {sentence}")

    # Print the unique and sorted version numbers
    print("Unique and sorted version numbers:", sorted_versions)
