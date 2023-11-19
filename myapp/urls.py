from django.urls import path
from .views import LicensePlateRecognition

urlpatterns = [
    path('process_image/', LicensePlateRecognition.as_view(), name='process_image'),
]