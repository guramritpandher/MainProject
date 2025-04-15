# PDF Summarizer and Chatbot

A Django application that allows users to upload PDF files, summarize them, and chat with them using an AI-powered chatbot.

## Features

- User authentication with JWT
- PDF upload and management
- PDF summarization
- AI-powered chatbot for asking questions about PDF content
- Responsive web interface

## Deployment Instructions

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Apply migrations: `python manage.py migrate`
4. Create a superuser: `python manage.py createsuperuser`
5. Run the development server: `python manage.py runserver`

### Production Deployment (PythonAnywhere)

1. Create a PythonAnywhere account
2. Upload the project files
3. Create a virtual environment and install dependencies
4. Configure the WSGI file
5. Set up static files
6. Apply migrations and create a superuser
7. Configure environment variables

## Environment Variables

- `DJANGO_SECRET_KEY`: Secret key for Django
- `DJANGO_DEBUG`: Set to 'False' in production
