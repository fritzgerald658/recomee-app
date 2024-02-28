from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_started, name= 'get_started'),
    path('predict_position/', views.predict_position, name='predict_position'),
]