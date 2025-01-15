import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk


# Load the trained model
model = load_model('D://image_Caption//Document//P5 Image Captioning//Flicker8k_Dataset_model.h5')


# Load the tokenizer and captions
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
    return tokenizer, captions_dict


# Function to extract features using MobileNetV2
def extract_image_features(image_path):
    # Load MobileNetV2 model pre-trained on ImageNet without the top classification layers
    mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    # Preprocess image for MobileNetV2 model
    img = load_img(image_path, target_size=(224, 224))  # MobileNetV2 requires 224x224 images
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocessing for MobileNetV2
    
    # Extract features using the pre-trained MobileNetV2 model
    features = mobilenet_model.predict(img_array)
    return features


# Function to find caption for the image from text files
def find_caption(image_filename, captions_dict):
    image_id = image_filename.split('.')[0]  # Extract image ID without extension (assuming filenames match)
    for captions in captions_dict.values():
        for caption in captions:
            if image_id in caption:  # Check if the caption contains the image ID
                return caption
    return None  # Return None if no caption is found


# Generate the caption for an image if not found in text files
def generate_caption(image_path, model, tokenizer, max_sequence_length=34):
    # Extract image features using the MobileNetV2 model
    image_features = extract_image_features(image_path)
    
    # Initialize the caption sequence with 'startseq'
    sequence = tokenizer.texts_to_sequences(['startseq'])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
    
    # Predict words until 'endseq' is predicted
    caption = []
    for i in range(max_sequence_length):
        y_pred = model.predict([image_features, padded_sequence])
        predicted_word_idx = np.argmax(y_pred)
        
        # Decode the word index into the word
        word = tokenizer.index_word.get(predicted_word_idx, '')
        if word == 'endseq':
            break
        caption.append(word)
        
        # Update the sequence with the new word
        padded_sequence = pad_sequences([tokenizer.texts_to_sequences([['startseq'] + caption])[0]], maxlen=max_sequence_length, padding='post')
    
    return ' '.join(caption)


# Function to handle the upload button
def upload_image():
    # Open a file dialog to choose an image
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")])
    
    if not image_path:
        messagebox.showwarning("No file selected", "Please select an image file.")
        return
    
    # Load tokenizer and captions
    captions_directory = 'D://image_Caption//Document//P5 Image Captioning//Flickr8k_text'
    tokenizer, captions_dict = prepare_tokenizer(captions_directory)
    
    # Get the image filename
    image_filename = os.path.basename(image_path)
    
    # Check if caption is already present in the text files
    existing_caption = find_caption(image_filename, captions_dict)
    
    if existing_caption:
        # If caption is found in the text files, show it
        print(f"The caption is: {existing_caption}")
        caption_label.config(text=f"Caption: {existing_caption}")
    else:
        # If no caption is found, generate a new one
        print("Generating new caption...")
        caption = generate_caption(image_path, model, tokenizer)
        print(f"Generated caption: {caption}")
        caption_label.config(text=f"Generated Caption: {caption}")
    
    # Display the selected image
    img = Image.open(image_path)
    img = img.resize((500, 500))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


# Set up the Tkinter window
root = tk.Tk()
root.title("Image Captioning")

# Set the window size
root.geometry("600x600")

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Create a label to display the caption
caption_label = tk.Label(root, text="Upload an image to generate or retrieve a caption.", wraplength=500, justify="center", font=("Arial", 14))
caption_label.pack(pady=20)

# Create a button to upload the image
upload_button = tk.Button(root, text="Upload Image", command=upload_image, height=2, width=20)
upload_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
