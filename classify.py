import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        return pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: The Parquet file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while loading the Parquet file: {e}")
    return None

def preprocess_data(data):
    if 'question' in data.columns and 'responses_are_same' in data.columns:
        X = data['question']
        y = data['responses_are_same']

        # Check class distribution before splitting
        print("Class distribution before split:")
        print(y.value_counts())

        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Balance the dataset
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train.to_frame(), y_train)

        # Check class distribution after resampling
        print("\nClass distribution after resampling:")
        print(y_resampled.value_counts())

        return X_resampled['question'], X_test, y_resampled, y_test

    print("Error: Required columns not found in the data.")
    return None, None, None, None

def train_classifier(X_train, y_train):
    try:
        tfidf = TfidfVectorizer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_train_tfidf, y_train)
        return tfidf, classifier
    except Exception as e:
        print(f"An error occurred during classifier training: {e}")
    return None, None

def evaluate_classifier(X_test, y_test, tfidf, classifier):
    try:
        X_test_tfidf = tfidf.transform(X_test)
        predictions = classifier.predict(X_test_tfidf)
        print("Classification Report:")
        print(classification_report(y_test, predictions))
    except Exception as e:
        print(f"An error occurred during classifier evaluation: {e}")

def classify_and_generate_response(question, tfidf, classifier, data):
    try:
        question_tfidf = tfidf.transform([question])
        response_index = classifier.predict(question_tfidf)[0]
        ai_responses = data.iloc[response_index]
        return ai_responses[['response_j_x', 'response_k_x', 'response_j_y', 'response_k_y']].tolist()
    except Exception as e:
        print(f"An error occurred during classification or response generation: {e}")
    return None

def main():
    file_path = '.\data\part-00000-694db9fd-774c-4205-b938-3729b352d322-c000.snappy.parquet'
    data = load_data(file_path)
    if data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        if X_train is not None:
            tfidf, classifier = train_classifier(X_train, y_train)
            if tfidf and classifier:
                evaluate_classifier(X_test, y_test, tfidf, classifier)
                question = "I have been feeling lonely."
                responses = classify_and_generate_response(question, tfidf, classifier, data)
                if responses:
                    print("\nUser Question:", question)
                    print("AI Response (Response_j_x):", responses[0])
                    print("AI Response (Response_k_x):", responses[1])
                    print("AI Response (Response_j_y):", responses[2])
                    print("AI Response (Response_k_y):", responses[3])
                else:
                    print("No AI responses found for the given input.")
            else:
                print("Script cannot continue without a trained classifier.")
        else:
            print("Script cannot continue without preprocessed data.")
    else:
        print("Script cannot continue without valid data.")

if __name__ == "__main__":
    main()
