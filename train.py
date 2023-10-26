import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import logging
from datetime import datetime
import os

# Config Logging
log_directory = 'Logs'
os.makedirs(log_directory, exist_ok=True)
CURRENT_DATETIME = datetime.now()
log_filename = f'{CURRENT_DATETIME.strftime("%B-%d-%Y-%H-%M-%S")}_log_file.log'
log_filepath = os.path.join(log_directory, log_filename)
logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    print(message)
    logging.info(message)

def load_data(file_paths):
    try:
        data_frames = [pd.read_parquet(file_path) for file_path in file_paths]
        data = pd.concat(data_frames, ignore_index=True)
        log_info("Data loaded successfully.")
        return data
    except Exception as e:
        log_info(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    try:
        # Concatenate 'question' and 'response_j' for training
        data['text'] = data['question'] + "\n" + data['response_j']
        data = data['text'].dropna().reset_index(drop=True)
        log_info("Data preprocessed successfully.")
        return data
    except Exception as e:
        log_info(f"Error preprocessing data: {e}")
        return None

def train_chatbot_model(data):
    try:
        # Split the data
        train_data, val_data = train_test_split(data, test_size=0.1)

        # Save the data to text files
        train_data.to_csv('train_data.txt', index=False, header=False)
        val_data.to_csv('val_data.txt', index=False, header=False)

        # Tokenize and prepare the data
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        train_dataset = TextDataset(tokenizer=tokenizer, file_path='train_data.txt', block_size=128)
        val_dataset = TextDataset(tokenizer=tokenizer, file_path='val_data.txt', block_size=128)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Define the model
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
        )

        # Create the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model('trained_chatbot_model')
        log_info("Chatbot model trained and saved successfully.")
    except Exception as e:
        log_info(f"Error training chatbot model: {e}")

def main():
    file_paths = [
        'data\part-00000-694db9fd-774c-4205-b938-3729b352d322-c000.snappy.parquet',
        'data\part-00001-694db9fd-774c-4205-b938-3729b352d322-c000.snappy.parquet'
    ]
    
    data = load_data(file_paths)
    if data is not None:
        preprocessed_data = preprocess_data(data)
        if preprocessed_data is not None:
            train_chatbot_model(preprocessed_data)

if __name__ == "__main__":
    main()
