from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .serializers import CustomUserSerializer, LoginSerializer, UploadedPDFSerializer
from rest_framework import generics, permissions
from . models import CustomUser, UploadedPDF
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated
import fitz, os
from transformers import pipeline
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import spacy
from rest_framework.decorators import api_view,permission_classes
from chatbot.utils1 import EnhancedPDFChatbot
from chatbot.conversation_utils import get_basic_response
from .translation_utils import translator  # Import the translator
from django.shortcuts import render
from django.middleware.csrf import get_token
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.parsers import MultiPartParser, FormParser

def welcome(request):
    return render(request, 'welcome.html')

class RegisterView(generics.CreateAPIView):
    queryset=CustomUser.objects.all()
    serializer_class=CustomUserSerializer
    permission_classes = [AllowAny]

@method_decorator(csrf_exempt, name='dispatch')
class LoginView(generics.GenericAPIView):
    serializer_class = LoginSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]

        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "message": "Login successful"
            },
            status=status.HTTP_200_OK,
        )

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": "Invalid refresh token"}, status=status.HTTP_400_BAD_REQUEST)

#This view will return the logged-in user's details
class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = CustomUserSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]  # Add this line

    def get_object(self):
        return self.request.user

# Download nltk data
nltk.download("punkt")

# Load the Hugging Face Summarization Model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class UploadPDFView(APIView):
    permission_classes = [IsAuthenticated]  # Ensures only logged-in users can access

    def post(self, request):
        if 'pdf_file' not in request.FILES:
            return Response({"error": "No file uploaded."}, status=400)

        pdf_file = request.FILES["pdf_file"]

        # Create the PDF object
        uploaded_pdf = UploadedPDF.objects.create(
            user=request.user,
            pdf_file=pdf_file
        )
        uploaded_pdf.save()

        pdf_path = uploaded_pdf.pdf_file.path
        if os.path.exists(pdf_path):
            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text() for page in doc])

            # Clean text (remove unwanted symbols and extra spaces)
            text = self.clean_text(text)

            # Generate a summary using Hugging Face
            summary = self.generate_summary(text)

            uploaded_pdf.summary = summary
            uploaded_pdf.save()

            return Response({
                "summary": summary,
                "id": uploaded_pdf.id
            })

        return Response({"error": "File was not saved correctly."}, status=500)

    def clean_text(self, text):
        """Cleans extracted text by removing unwanted characters and symbols."""
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        text = re.sub(r"[^a-zA-Z0-9.,!?; ]", "", text)  # Keep only useful characters
        sentences = sent_tokenize(text)  # Tokenize into sentences
        return " ".join(sentences[:30])  # Keep only the first 20 sentences


    def clean_summary(self, summary_text):
        """Ensures summary ends with a complete sentence and removes broken endings."""
        # Split into sentences
        sentences = sent_tokenize(summary_text)

        # Filter out any obviously broken or incomplete sentences
        clean_sentences = []
        for sentence in sentences:
            if len(sentence.split()) < 4:
                continue  # Skip very short, likely broken sentences
            if re.search(r"\b(com|data\.com|\.com)\b", sentence.lower()):
                continue  # Skip malformed web-like endings
            clean_sentences.append(sentence)

        # Join and ensure the final one ends properly
        if clean_sentences and not clean_sentences[-1].endswith(('.', '!', '?')):
            clean_sentences[-1] += '.'

        return " ".join(clean_sentences)

    def generate_summary(self, text):
        """Splits long text into chunks and summarizes each part."""
        chunks = self.chunk_text(text, max_tokens=900)
        all_summaries = []

        for chunk in chunks:
            if len(chunk) < 100:
                all_summaries.append(chunk)
            else:
                summary = summarizer(chunk, max_length=1000, min_length=500, do_sample=False)
                all_summaries.append(summary[0]['summary_text'])

        final_summary = " ".join(all_summaries)

        # Ensure the final summary ends with a full sentence
        if not final_summary.endswith((".", "!", "?")):
            final_summary += "."

        return final_summary

    def chunk_text(self, text, max_tokens=900):
        """Splits text into chunks under the token limit for summarization models."""
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            token_count = len(sentence.split())
            if current_length + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = token_count
            else:
                current_chunk.append(sentence)
                current_length += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

def upload_pdf(request):
    return render(request, "upload.html")

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")
print("SpaCy model loaded successfully!")

@api_view(["POST"])
@permission_classes([IsAuthenticated])  # Ensure only logged-in users can access
def chatbot_api(request):
    """API endpoint to interact with the chatbot for the user's uploaded PDF."""
    query = request.data.get("query", "").strip()

    if not query:
        return JsonResponse({"error": "Query is required"}, status=400)

    # Check if this is a basic conversational query
    # Get the username for personalized responses
    username = request.user.username if hasattr(request.user, 'username') else None
    basic_response = get_basic_response(query, username)
    if basic_response:
        return JsonResponse({"query": query, "answer": basic_response})

    # If not a basic query, proceed with PDF-based response
    # Retrieve the latest uploaded PDF for the authenticated user
    uploaded_pdf = UploadedPDF.objects.filter(user=request.user).order_by("-uploaded_at").first()

    if not uploaded_pdf:
        return JsonResponse({"error": "No uploaded PDF found. Please upload a document first."}, status=400)

    pdf_path = uploaded_pdf.pdf_file.path

    # Initialize the chatbot with the user's uploaded PDF
    chatbot = EnhancedPDFChatbot(pdf_path=pdf_path)

    # Generate an answer for the query
    answer = chatbot.generate_answer(query)

    return JsonResponse({"query": query, "answer": answer})

@login_required
def chatbot_page(request):
    return render(request, "chatbot.html")

class UserHistoryListView(generics.ListAPIView):
    """
    API to fetch all PDFs uploaded by the authenticated user
    """
    serializer_class = UploadedPDFSerializer
    permission_classes = [IsAuthenticated]  # Ensure only logged-in users access this

    def get_queryset(self):
        # Ensure the user is authenticated
        if self.request.user.is_authenticated:
            return UploadedPDF.objects.filter(user=self.request.user).order_by("-uploaded_at")
        return UploadedPDF.objects.none()  # Return empty queryset if not authenticated

@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_pdf(request, pdf_id):
    """
    API endpoint to delete a PDF from user's history
    """
    try:
        # Find the PDF and ensure it belongs to the current user
        pdf = UploadedPDF.objects.get(id=pdf_id, user=request.user)

        # Get the file path to delete the actual file
        file_path = pdf.pdf_file.path

        # Delete any associated chat messages
        from chatbot.models import ChatMessage
        chat_messages_count = ChatMessage.objects.filter(pdf=pdf).count()
        ChatMessage.objects.filter(pdf=pdf).delete()

        # Delete the PDF record from the database
        pdf_name = pdf.pdf_file.name.split('/')[-1] if '/' in pdf.pdf_file.name else pdf.pdf_file.name
        pdf.delete()

        # Try to delete the actual file from storage
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            # Log the error but continue - we've already deleted the database record
            print(f"Error deleting file {file_path}: {str(e)}")

        return JsonResponse({
            "message": f"PDF '{pdf_name}' successfully deleted along with {chat_messages_count} chat messages.",
            "deleted_id": pdf_id
        })

    except UploadedPDF.DoesNotExist:
        return JsonResponse({"error": "PDF not found or you don't have permission to delete it."}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

def history_page(request):
    return render(request, "history.html")

def register_page(request):
    return render(request,"register.html")

def login_page(request):
    return render(request,"login.html")

def get_csrf_token(request):
    return JsonResponse({"csrfToken": get_token(request)})

def about_app(request):
    return render(request, 'about.html')

def user_profile(request):
    return render(request, "profile.html")

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def translate_summary(request):
    """
    API endpoint to translate a PDF summary to a different language.

    Expected request data:
    - pdf_id: ID of the PDF to translate (optional, uses latest if not provided)
    - target_language: Language to translate to (e.g., 'hindi', 'spanish')

    Returns:
    - translated_summary: The translated summary
    - target_language: The language translated to
    """
    pdf_id = request.data.get("pdf_id")
    target_language = request.data.get("target_language", "hindi").lower()

    try:
        # Get the PDF
        if pdf_id:
            # Get specific PDF and ensure it belongs to the current user
            pdf = UploadedPDF.objects.get(id=pdf_id, user=request.user)
        else:
            # Get the latest PDF for the user
            pdf = UploadedPDF.objects.filter(user=request.user).order_by("-uploaded_at").first()

        if not pdf:
            return JsonResponse({"error": "No PDF found. Please upload a document first."}, status=404)

        # Get the summary
        original_summary = pdf.summary

        if not original_summary:
            return JsonResponse({"error": "No summary found for this PDF"}, status=400)

        # Translate the summary
        translated_summary = translator.translate(original_summary, target_language)

        return JsonResponse({
            "translated_summary": translated_summary,
            "target_language": target_language
        })

    except UploadedPDF.DoesNotExist:
        return JsonResponse({"error": "PDF not found or you don't have permission to access it"}, status=404)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

@api_view(["GET"])
def available_languages(request):
    """
    API endpoint to get a list of available languages for translation.
    """
    languages = translator.get_available_languages()
    return JsonResponse({"languages": languages})