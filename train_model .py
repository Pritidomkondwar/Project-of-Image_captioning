import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import os

# Function to create a custom CNN model for feature extraction
def create_custom_cnn_model():
    print("Creating custom CNN model for feature extraction...")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))  # First convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
    model.add(Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
    model.add(Flatten())  # Flatten the output for fully connected layer
    model.add(Dense(128, activation='relu'))  # Fully connected layer
    print("Custom CNN model created.")
    return model

# Function to save a model to a specific path
def save_model(model, model_path):
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

# Preprocess image for the custom CNN
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to (224, 224) for CNN
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Load Captions from Text Files
def load_captions_from_directory(captions_directory):
    captions_dict = {}
    try:
        for filename in os.listdir(captions_directory):
            file_path = os.path.join(captions_directory, filename)
            if filename.endswith('.txt'):
                with open(file_path, 'r') as file:
                    captions = file.readlines()
                    captions_dict[filename] = [caption.strip() for caption in captions]
        print(f"Loaded {len(captions_dict)} caption files.")
        return captions_dict
    except Exception as e:
        print(f"Failed to load captions from directory: {e}")
        return {}

# Prepare Tokenizer for caption generation
def prepare_tokenizer(captions_directory):
    captions_dict = load_captions_from_directory(captions_directory)
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    print("Tokenizer prepared.")
    return tokenizer

# Create a simple LSTM-based Captioning model
def create_captioning_model(vocab_size, max_sequence_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256, input_length=max_sequence_length))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("Captioning model created.")
    return model

# Train the captioning model (this is just a simplified function)
def train_captioning_model(feature_extractor_model, caption_model, tokenizer, max_sequence_length, images, captions):
    # Assuming 'images' is a list of image paths and 'captions' is a list of corresponding captions
    print("Training model...")
    
    # Placeholder: Create features for images (feature extraction using CNN model)
    features = np.array([feature_extractor_model.predict(preprocess_image(img_path)) for img_path in images])
    
    # Tokenizing and padding captions
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    
    # Training the model (simplified)
    caption_model.fit([features, padded_sequences], np.array(sequences), epochs=10, batch_size=32)
    print("Model training complete.")
    return caption_model

# Main function to train and save models
def train_and_save_models():
    # Set file paths
    captions_directory = 'D://image_Caption//Document//P5 Image Captioning//Flickr8k_text'
    images_directory = 'D://image_Caption//Document//P5 Image Captioning//Flicker8k_Dataset'
    
    # Load the tokenizer
    tokenizer = prepare_tokenizer(captions_directory)
    
    # Create CNN model for feature extraction
    feature_extractor_model = create_custom_cnn_model()
    
    # Create the captioning model
    vocab_size = len(tokenizer.word_index) + 1  # Plus one for padding token
    max_sequence_length = 34  # Assuming 34 is the max sequence length for captions
    caption_model = create_captioning_model(vocab_size, max_sequence_length)
    
    # Example image and caption paths (this should be replaced by your actual data)
    images = [os.path.join(images_directory, fname) for fname in os.listdir(images_directory) if fname.endswith('.jpg')]
    captions = ['This is an example caption for the image.'] * len(images)  # Replace with actual captions
    
    # Train the captioning model
    trained_caption_model = train_captioning_model(feature_extractor_model, caption_model, tokenizer, max_sequence_length, images, captions)
    
    # Save the trained models
    save_model(feature_extractor_model, "D://image_Caption//Document//P5 Image Captioning//Saved_CNN_Model.h5")
    save_model(trained_caption_model, "D://image_Caption//Document//P5 Image Captioning//Saved_Caption_Model.h5")

# Run the training and saving process
train_and_save_models()

