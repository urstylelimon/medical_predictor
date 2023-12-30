# yourapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='predict'),  # Add this line to handle the root URL
]
