# Import the function to define URL paths and the views module from the current package
from django.urls import path
from . import views

# List of URL patterns where each path is associated with a view
urlpatterns = [
    # Map the 'detect_emotion/' URL path to the detect_emotion view with a name for reverse URL matching
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),

    # Map the root URL path ('') to the home view, which is the index or the main page
    path('', views.home, name='home'),

    # Map the 'photo_emotion_detection/' URL path to the photo_emotion_detection view with a name for reverse URL matching
    path('photo_emotion_detection/', views.photo_emotion_detection, name='photo_emotion_detection'),

    # Map the 'faq/' URL path to the faq view, which likely displays frequently asked questions
    path('faq/', views.faq, name='faq')
]
