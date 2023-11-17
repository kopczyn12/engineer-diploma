# Import necessary libraries and functions
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
import base64
import io
from PIL import Image, UnidentifiedImageError
import numpy as np
import os
import uuid
import cv2
from django.conf import settings
from torchvision import transforms
from transformers import ConvNextForImageClassification
from torch import nn
import torch

# Define a view for the homepage
def home(request):
    # Render the 'index.html' template when accessing the homepage
    return render(request, 'index.html')

# Define a view for the FAQ page
def faq(request):
    # Render the 'faq.html' template when accessing the FAQ page
    return render(request, 'faq.html')

# Set up the device for model computation based on availability of CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained ConvNeXt model and modify the classifier for our use case
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 5)  # Assuming 5 classes for classification

# Load the model onto the device and set it to evaluation mode
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'emotion_app/model/convnext.pth')
model = model.to(device)
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Function to preprocess a single image for the ConvNeXt model
def preprocess_single_image(image):
    """Preprocesses the input image for the ConvNeXt model."""
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    return image_tensor

# Function to predict the emotion from an image
def predict_emotion_from_image(image):
    """Predict emotion on image"""
    # Preprocess the image
    image_tensor = preprocess_single_image(image)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs.logits, 1)  # Get the index of the max log-probability
    
    # Define the possible labels for the classification
    labels = ["Anger", "Disgust", "Happy", "Sad", "Surprised"]
    predicted_label = labels[preds.item()]  # Map prediction to the corresponding label
    
    return predicted_label

# Define a view for the photo emotion detection feature
def photo_emotion_detection(request):
    """Emotion detection"""
    # Initialize variables to hold the emotion and image URL
    emotion = None
    image_url = None

    # Handle file upload via POST request
    if request.method == 'POST' and request.FILES.get('photo'):
        uploaded_photo = request.FILES['photo']
        image_stream = io.BytesIO(uploaded_photo.read())

        try:
            # Attempt to open and verify the uploaded image
            photo = Image.open(image_stream)
            photo.verify()  # Verify that it is, in fact, an image
            image_stream.seek(0)  # Seek to the beginning of the file
            photo = Image.open(image_stream)  # Reopen the image file

            # Create a temporary directory for saving the uploaded photo if it doesn't exist
            if not os.path.exists(os.path.join(settings.MEDIA_ROOT, 'temp')):
                os.makedirs(os.path.join(settings.MEDIA_ROOT, 'temp'))

            # Generate a unique filename for the uploaded photo
            unique_filename = str(uuid.uuid4()) + '.' + uploaded_photo.name.split('.')[-1]
            image_path = os.path.join(settings.MEDIA_ROOT, 'temp', unique_filename)
            photo.save(image_path)  # Save the photo to the filesystem
            image_url = settings.MEDIA_URL + 'temp/' + unique_filename  # Construct the URL for the saved photo

            # Detect faces in the photo
            face = detect_face(photo)
            if face is None:
                emotion = "No face detected in the uploaded image."
            else:
                # Predict the emotion from the detected face
                emotion = predict_emotion_from_image(face)

        except UnidentifiedImageError:
            # Handle exceptions if the image is corrupted or invalid
            emotion = "Invalid or corrupted image uploaded."

    # Render the photo emotion detector template with the emotion and image URL
    return render(request, 'photo_emotion_detector.html', {'emotion': emotion, 'image_url': image_url})

# Function to detect a face in the image
def detect_face(image):
    """Detect face on the frame"""
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image from RGB to BGR, as OpenCV expects images in BGR format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(opencv_image, 1.1, 4)
    
    # If faces are detected, return the first face found
    for (x, y, w, h) in faces:
        face = opencv_image[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
    # If no faces are detected, return None
    return None

# Define a view to handle emotion detection via JSON requests
@csrf_exempt
def detect_emotion(request):
    """Handles emotion detection from a JSON payload containing the image."""
    if request.method == 'POST':
        # Parse the JSON data from the request
        data = json.loads(request.body)
        # Decode the image from base64
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Detect a face in the image
        face = detect_face(image)
        if face is None:
            # If no face is detected, return an error in JSON format
            return JsonResponse({'error': 'No face detected in the image'})

        # Predict the emotion from the detected face
        emotion = predict_emotion_from_image(face)
        # Return the predicted emotion in JSON format
        return JsonResponse({'emotion': emotion})

    # If the request is not a POST, render the emotion detector HTML template
    return render(request, 'emotion_detector.html', {})
