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

def home(request):
    return render(request, 'index.html')

def faq(request):
    return render(request, 'faq.html')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 5)  
model.load_state_dict(torch.load('./engineer-diploma/emotion_detector_project/emotion_app/model/convnext.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_single_image(image):
    """Preprocesses the input image for the ConvNeXt model."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_emotion_from_image(image):
    image_tensor = preprocess_single_image(image)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs.logits, 1)
    
    labels = ["Anger", "Disgust", "Happy", "Sad", "Surprised"]
    predicted_label = labels[preds.item()]
    
    return predicted_label

def photo_emotion_detection(request):
    emotion = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('photo'):
        uploaded_photo = request.FILES['photo']
        image_stream = io.BytesIO(uploaded_photo.read())

        try:
            photo = Image.open(image_stream)
            photo.verify()
            image_stream.seek(0)
            photo = Image.open(image_stream)

            if not os.path.exists(os.path.join(settings.MEDIA_ROOT, 'temp')):
                os.makedirs(os.path.join(settings.MEDIA_ROOT, 'temp'))

            unique_filename = str(uuid.uuid4()) + '.' + uploaded_photo.name.split('.')[-1]
            image_path = os.path.join(settings.MEDIA_ROOT, 'temp', unique_filename)
            photo.save(image_path)
            image_url = settings.MEDIA_URL + 'temp/' + unique_filename

            face = detect_face(photo)
            if face is None:
                emotion = "No face detected in the uploaded image."
            else:
                emotion = predict_emotion_from_image(face)

        except UnidentifiedImageError:
            emotion = "Invalid or corrupted image uploaded."

    return render(request, 'photo_emotion_detector.html', {'emotion': emotion, 'image_url': image_url})

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(opencv_image, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = opencv_image[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
    return None

@csrf_exempt
def detect_emotion(request):
    """Detekcja emocji z obrazu przesłanego w formacie JSON."""
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        
        face = detect_face(image)
        if face is None:
            return JsonResponse({'error': 'No face detected in the image'})

        emotion = predict_emotion_from_image(face)
        return JsonResponse({'emotion': emotion})

    return render(request, 'emotion_detector.html', {})
