from django.urls import path
from . import views

urlpatterns = [
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),
    path('', views.home, name='home'),
    path('photo_emotion_detection/', views.photo_emotion_detection, name='photo_emotion_detection'),
    path('faq/', views.faq, name='faq')
]

