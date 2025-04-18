from django.urls import path
from .views import pdf_specific_chatbot_api, pdf_library_page, get_pdf_chat_history

urlpatterns = [
    path('pdf_chat/', pdf_specific_chatbot_api, name='pdf_specific_chatbot'),
    path('pdf_library/', pdf_library_page, name='pdf_library'),
    path('chat_history/<int:pdf_id>/', get_pdf_chat_history, name='pdf_chat_history'),
]
