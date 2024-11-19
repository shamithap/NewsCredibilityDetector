#fakenewsdetector.py
import os
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import AdamWeightDecay

# Ensure 'model' directory exists
os.makedirs('./model', exist_ok=True)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Memory growth enabled for {device}")
else:
    print("No GPU detected. Using CPU.")

# Load the model and tokenizer
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
print("Model and tokenizer loaded successfully.")

def load_data():
    # Load the Excel file with only the necessary columns
    df = pd.read_excel("WELFake_Dataset.xlsx", usecols=['text', 'label'], dtype=str)

    # Drop rows with missing values in 'text' or 'label'
    df = df.dropna(subset=['text', 'label'])

    # Filter to keep only rows where 'label' contains numeric values
    df = df[df['label'].str.isnumeric()]

    # Convert labels to integers
    df['label'] = df['label'].astype(int)

    # Filter out invalid labels (only keep 0 and 1)
    df = df[df['label'].isin([0, 1])]

    # Limit the dataset to 10,000 samples
    df = df.sample(n=10000, random_state=42)  # Adjust sample size as needed

    print("Subset of dataset loaded and cleaned successfully.")
    return df

print("Model and data loaded successfully. Beginning training.")

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Check if dataframe loaded correctly
    print(f"DataFrame shape: {df.shape}")
    
    # Handle missing values and convert text to string
    x = df['text'].fillna('').astype(str).tolist()
    print(f"Processed {len(x)} text samples.")
    
    # Extract labels
    y = list(df['label'].astype(int))
    print(f"Processed {len(y)} labels.")
    
    # Tokenization
    try:
        encodings = tokenizer(x, max_length=100, truncation=True, padding=True, return_tensors='tf')
        print("Tokenization completed successfully.")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        raise
    
    return encodings, y

def train_model():
    df = load_data()  # Load the entire dataset
    encodings, y = preprocess_data(df)
    
    print("Training the model...")
    y = tf.convert_to_tensor(y, dtype=tf.int32)  # Ensure consistent data type
    tfdataset = tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    tfdataset = tfdataset.shuffle(len(df))  # Shuffle the dataset
    train_size = int(len(df) * 0.7)  # Set the train size to 70% of the data
    tfdataset_train = tfdataset.take(train_size).batch(1)  # Smaller batch size for quicker training
    tfdataset_test = tfdataset.skip(train_size).batch(1)  # Test dataset

    print(f"Train dataset size: {train_size}")
    print(f"Test dataset size: {len(df) - train_size}")

    # Correct optimizer setup using AdamWeightDecay for compatibility
    optimizer = AdamWeightDecay(learning_rate=1e-5, weight_decay_rate=0.01)

    # Model compilation
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        tfdataset_train,
        validation_data=tfdataset_test,
        epochs=1,  # Start with 1 epoch for testing
        verbose=3  # Use verbose output to track progress
    )

    # Final evaluation on the test dataset
    benchmarks = model.evaluate(tfdataset_test, return_dict=True)
    print(f"Final test accuracy: {benchmarks['accuracy']}")
    print(f"Final test loss: {benchmarks['loss']}")

    # Save model weights manually after training
    model.save_weights('./model/final_model.weights.h5')
    print("Model weights saved successfully.")

def detect_fake_news(text):
    # Preprocess the input text
    encodings = tokenizer([text], return_tensors='tf', truncation=True, padding=True, max_length=100)
    
    # Make predictions using the model
    preds = model(encodings)[0]
    
    # Get the prediction (0 for real news, 1 for fake news)
    prediction = tf.argmax(preds, axis=1).numpy()
    
    return bool(prediction[0])  # Return True if fake news, False otherwise

if __name__ == "__main__":
    train_model()
