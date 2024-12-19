from django.http import JsonResponse
from django.shortcuts import render

from apps.main.ai.chatbot import get_islamic_chatbot
import json


def index(request):
    """Renders the index.html template."""
    return render(request, 'index.html')


def get_chatbox_response(request):
    data = json.loads(request.body)
    user_message = data.get('user_message')
    islamic_chatbot = get_islamic_chatbot()
    response = islamic_chatbot.query(user_message)
    return JsonResponse({'response': response})
