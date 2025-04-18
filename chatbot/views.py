from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse
from .utils1 import EnhancedPDFChatbot
from accounts.models import UploadedPDF
from django.contrib.auth.decorators import login_required
from .models import ChatMessage
from .conversation_utils import get_basic_response

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def pdf_specific_chatbot_api(request):
    """API endpoint to interact with a specific PDF from user's history."""
    query = request.data.get("query", "").strip()
    pdf_id = request.data.get("pdf_id")

    if not query:
        return JsonResponse({"error": "Query is required"}, status=400)

    if not pdf_id:
        return JsonResponse({"error": "PDF ID is required"}, status=400)

    # Check if this is a basic conversational query
    # Get the username for personalized responses
    username = request.user.username if hasattr(request.user, 'username') else None
    basic_response = get_basic_response(query, username)
    if basic_response:
        # Save the basic conversation to chat history
        try:
            uploaded_pdf = UploadedPDF.objects.get(id=pdf_id, user=request.user)
            # Save the chat message to the database
            ChatMessage.objects.create(
                user=request.user,
                pdf=uploaded_pdf,
                query=query,
                response=basic_response
            )
        except UploadedPDF.DoesNotExist:
            # If PDF doesn't exist, just return the response without saving
            pass

        return JsonResponse({
            "query": query,
            "answer": basic_response,
            "pdf_id": pdf_id
        })

    try:
        # Retrieve the specific PDF by ID, ensuring it belongs to the current user
        uploaded_pdf = UploadedPDF.objects.get(id=pdf_id, user=request.user)
        pdf_path = uploaded_pdf.pdf_file.path

        # Initialize the chatbot with the selected PDF
        chatbot = EnhancedPDFChatbot(pdf_path=pdf_path)

        # Generate an answer for the query
        answer = chatbot.generate_answer(query)

        # Save the chat message to the database
        ChatMessage.objects.create(
            user=request.user,
            pdf=uploaded_pdf,
            query=query,
            response=answer
        )

        # Get a meaningful PDF name
        pdf_name = uploaded_pdf.pdf_file.name

        # If the name is just the path or empty, try to extract a better name
        if not pdf_name or pdf_name == 'pdfs/' or '/' not in pdf_name:
            # Try to extract from the file path
            file_path = uploaded_pdf.pdf_file.path
            if file_path and '/' in file_path:
                pdf_name = file_path.split('/')[-1]
            else:
                # Use a default name with the ID
                pdf_name = f"PDF #{uploaded_pdf.id}"
        else:
            # Extract just the filename from the path
            pdf_name = pdf_name.split('/')[-1]

        return JsonResponse({
            "query": query,
            "answer": answer,
            "pdf_name": pdf_name,
            "pdf_id": uploaded_pdf.id,
            "uploaded_at": uploaded_pdf.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")
        })

    except UploadedPDF.DoesNotExist:
        return JsonResponse({"error": "PDF not found or you don't have permission to access it."}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def get_pdf_chat_history(request, pdf_id):
    """API endpoint to retrieve chat history for a specific PDF."""
    try:
        # Verify the PDF exists and belongs to the user
        pdf = UploadedPDF.objects.get(id=pdf_id, user=request.user)

        if request.method == "DELETE":
            # Delete all chat messages for this PDF
            deleted_count, _ = ChatMessage.objects.filter(pdf=pdf, user=request.user).delete()
            return JsonResponse({
                'message': f'Successfully deleted {deleted_count} chat messages.',
                'pdf_id': pdf_id
            })
        else:  # GET request
            # Get all chat messages for this PDF
            chat_messages = ChatMessage.objects.filter(pdf=pdf, user=request.user).order_by('created_at')

            # Format the messages for the response
            messages = [{
                'id': msg.id,
                'query': msg.query,
                'response': msg.response,
                'created_at': msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
            } for msg in chat_messages]

            return JsonResponse({
                'pdf_id': pdf_id,
                'pdf_name': pdf.pdf_file.name.split('/')[-1] if '/' in pdf.pdf_file.name else pdf.pdf_file.name,
                'messages': messages
            })

    except UploadedPDF.DoesNotExist:
        return JsonResponse({"error": "PDF not found or you don't have permission to access it."}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

@login_required
def pdf_library_page(request):
    """Render the PDF library page where users can chat with their previously uploaded PDFs."""
    return render(request, "pdf_library.html")
