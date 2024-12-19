from django.urls import path

from apps.main.ai.chatbot import get_islamic_chatbot
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path(
        'get-chatbot-response/',
        views.get_chatbox_response,
        name='get_chatbox_response')
]

get_islamic_chatbot()
