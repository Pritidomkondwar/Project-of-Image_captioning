import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Embedding, Dense, Add
from tensorflow.keras.models import Model
import numpy as np
import os
from tensorflow.keras.optimizers import Adam

# Function to create a custom CNN model for feature extraction
def create_custom_cnn_model():
    print("Creating custom CNN model for feature extraction...")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))  # First convolutional layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
    model.add(tf.keras.layers.Flatten())  # Flatten the output for fully connected layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # Fully connected layer
    print("Custom CNN model created.")
    return model

# Function to create the LSTM captioning model
# Function to create the LSTM captioning model
def create_captioning_model(vocab_size, max_sequence_length):
    print("Creating captioning LSTM model...")
    input_features = tf.keras.layers.Input(shape=(128,))  # Feature vector from CNN
    input_sequence = tf.keras.layers.Input(shape=(max_sequence_length,))  # Sequence of previous words
    
    # Add a dense layer to match the shape of the LSTM output
    feature_dense = tf.keras.layers.Dense(256, activation='relu')(input_features)  # Match the LSTM output shape
    
    # Embedding layer to transform word indices into word vectors
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(input_sequence)
    
    # LSTM for generating the next word in the caption
    lstm = tf.keras.layers.LSTM(256)(embedding)
    
    # Combine features from CNN (transformed to match LSTM output) and LSTM
    decoder_input = tf.keras.layers.Add()([feature_dense, lstm])  # Merge both layers
    output = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder_input)
    
    model = tf.keras.models.Model(inputs=[input_features, input_sequence], outputs=output)
    print("Captioning LSTM model created.")
    return model


# Function to load and preprocess image
def preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    img = load_img(image_path, target_size=(224, 224))  # Resize to (224, 224) for CNN
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to prepare the tokenizer using captions
def prepare_tokenizer(captions_directory):
    captions_dict = {}
    for filename in os.listdir(captions_directory):
        file_path = os.path.join(captions_directory, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                captions = file.readlines()
                captions_dict[filename] = [caption.strip() for caption in captions]
    
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# Function to train the model
def train_model(captions_directory, image_directory, model_save_path, vocab_size, max_sequence_length):
    print("Training the model...")
    
    feature_extractor_model = create_custom_cnn_model()
    
    # Create the captioning LSTM model
    caption_model = create_captioning_model(vocab_size, max_sequence_length)
    
    # Compile the model
    caption_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load your data and prepare for training (this would need to be implemented based on your dataset)
    tokenizer = prepare_tokenizer(captions_directory)
    # Generate features and prepare the inputs and targets for training
    # For simplicity, we skip actual training data preparation code here

    # Model training: replace with actual training process
    # caption_model.fit(...)

    print(f"Training completed. Saving model to {model_save_path}...")
    caption_model.save(model_save_path)
    print("Model saved.")

# Example usage
captions_directory = 'D://image_Caption//Document//P5 Image Captioning//Flickr8k_text'
image_directory = 'D://image_Caption//Document//P5 Image Captioning//Flicker8k_Dataset'
model_save_path = 'D://image_Caption//Document//P5 Image Captioning//Flicker8k_Dataset_model.h5'

# Example training parameters (adjust based on your data)
vocab_size = 5000  # Adjust according to your tokenizer
max_sequence_length = 34  # Adjust based on your caption length

# Train the model
train_model(captions_directory, image_directory, model_save_path, vocab_size, max_sequence_length)
