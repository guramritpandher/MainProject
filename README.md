# PDF Summarizer

A web application that allows users to upload PDF documents, generate summaries, and chat with their documents using advanced NLP techniques.

## Features

- **PDF Upload**: Upload PDF documents for summarization
- **Automatic Summarization**: Generate concise summaries of uploaded PDFs
- **Interactive Chatbot**: Ask questions about your PDF documents
- **PDF Library**: Access all your previously uploaded PDFs
- **User History**: View and manage your upload history
- **User Profiles**: Customize your profile with a username and profile picture
- **Translation**: Translate PDF summaries into different languages
- **Responsive Design**: Beautiful UI that works on desktop and mobile devices

## Technology Stack

- **Backend**: Django, Django REST Framework
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Authentication**: JWT (JSON Web Tokens)
- **NLP**: Transformers, spaCy, NLTK
- **PDF Processing**: PyMuPDF
- **Vector Database**: FAISS
- **Machine Learning**: Hugging Face Transformers, Sentence Transformers

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pdf_summarizer
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Apply migrations:
   ```
   python manage.py migrate
   ```

5. Run the development server:
   ```
   python manage.py runserver
   ```

6. Access the application at http://127.0.0.1:8000/

## Usage

1. Register a new account or log in
2. Upload a PDF document
3. View the generated summary
4. Ask questions about your document using the chatbot
5. Access your PDF library to view all your documents
6. View your upload history and manage your documents

## Project Structure

- `accounts/`: User authentication and profile management
- `chatbot/`: PDF processing, summarization, and chatbot functionality
- `pdf_summarizer/`: Main project settings and configuration
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and other static files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for their amazing Transformers library
- Django and Django REST Framework for the web framework
- Bootstrap for the frontend components