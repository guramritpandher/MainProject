from django.db import models
from accounts.models import CustomUser, UploadedPDF

class ChatMessage(models.Model):
    """Model to store chat messages between users and PDFs"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='chat_messages')
    pdf = models.ForeignKey(UploadedPDF, on_delete=models.CASCADE, related_name='chat_messages')
    query = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.user.email} - {self.pdf.pdf_file.name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
