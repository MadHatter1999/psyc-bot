import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4
import logging
from datetime import datetime
import os

#Config Configging
log_directory = 'Logs'
os.makedirs(log_directory, exist_ok=True)
CURRENT_DATETIME = datetime.now()
log_filename = f'{CURRENT_DATETIME.strftime("%B-%d-%Y-%H-%M-%S")}_log_file.log'
log_filepath = os.path.join(log_directory, log_filename)
logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    print(message)
    logging.info(message)

def load_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
        log_info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        log_info(f"Error loading data from {file_path}: {e}")
        return None

def clean_and_analyze_responses(merged_df):
    try:
        log_info("Cleaning and analyzing responses...")
        # Cleaning Text Data
        merged_df['response_j_x'] = merged_df['response_j_x'].str.lower().str.strip()
        merged_df['response_j_y'] = merged_df['response_j_y'].str.lower().str.strip()

        # Add a new column indicating if the responses are the same
        merged_df['responses_are_same'] = merged_df['response_j_x'] == merged_df['response_j_y']

        # Sentiment Analysis
        merged_df['sentiment_j_x'] = merged_df['response_j_x'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        merged_df['sentiment_j_y'] = merged_df['response_j_y'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

        log_info("Responses cleaned and analyzed successfully.")
        return merged_df
    except Exception as e:
        log_info(f"Error cleaning and analyzing responses: {e}")
        return None

def visualize_responses(merged_df):
    try:
        #I am a sucker for graphs
        log_info("Visualizing responses...")
        plt.figure(figsize=(10, 5))

        plt.title('Sentiment Distribution of Responses J and K')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.show()
        log_info("Responses visualized successfully.")
    except Exception as e:
        log_info(f"Error visualizing responses: {e}")

def save_data(merged_df, file_path):
    try:
        log_info(f"Saving data to {file_path}...")
        merged_df.to_parquet(file_path)
        log_info(f"Data saved successfully to {file_path}")
        return file_path
    except Exception as e:
        log_info(f"Error saving data to {file_path}: {e}")
        return None

def combine_data():
    log_info("Starting the analysis...")
    file_path1 = './data/part-00000-694db9fd-774c-4205-b938-3729b352d322-c000.snappy.parquet'
    file_path2 = './data/part-00001-694db9fd-774c-4205-b938-3729b352d322-c000.snappy.parquet'
    uuid = uuid4()
    save_path = f'./output/psychiDoc-{uuid}.parquet'

    df1 = load_parquet(file_path1)
    df2 = load_parquet(file_path2)

    if df1 is not None and df2 is not None:
        merged_df = pd.merge(df1, df2, on='question', how='outer')
        cleaned_and_analyzed_df = clean_and_analyze_responses(merged_df)
        if cleaned_and_analyzed_df is not None:
            visualize_responses(cleaned_and_analyzed_df)
            post_res = save_data(cleaned_and_analyzed_df, save_path)
            if (post_res):
                return post_res
            return None
        else:
            log_info("Analysis and visualization were not performed due to a previous error.")

if __name__ == "__main__":
    combine_data()