"""
URL configuration for pdf_summarizer project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
from accounts.views import welcome
# Import the views directly
from accounts.views import (
    welcome, register_page, login_page, about_app,
    upload_pdf, chatbot_page, history_page, user_profile
)
from chatbot.views import pdf_library_page

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', welcome, name='welcome'),

    # API endpoints
    path('api/auth/', include('accounts.urls')),
    path('api/chatbot/', include('chatbot.urls')),

    # Direct page URLs
    path('register_page/', register_page, name='register_page'),
    path('login_page/', login_page, name='login_page'),
    path('about/', about_app, name='about'),
    path('upload_pdf/', upload_pdf, name='upload_pdf'),
    path('chatbot_page/', chatbot_page, name='chatbot_page'),
    path('history_page/', history_page, name='history_page'),
    path('profile_page/', user_profile, name='profile_page'),
    path('pdf_library/', pdf_library_page, name='pdf_library'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)