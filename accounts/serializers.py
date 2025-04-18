from rest_framework import serializers
from .models import CustomUser
from django.contrib.auth import authenticate
from .models import UploadedPDF


class CustomUserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=False)
    profile_picture = serializers.ImageField(required=False, use_url=True)

    class Meta:
        model = CustomUser
        fields = ['id', 'email', 'username', 'password', 'profile_picture']

    def create(self, validated_data):
        user = CustomUser.objects.create_user(
            email=validated_data['email'],
            username=validated_data['username'],
            password=validated_data['password'],
            profile_picture=validated_data.get('profile_picture', None)
        )
        return user

    def update(self, instance, validated_data):
        instance.username = validated_data.get('username', instance.username)
        instance.email = validated_data.get('email', instance.email)
        profile_picture = validated_data.get('profile_picture', None)
        if profile_picture:
            instance.profile_picture = profile_picture
        instance.save()
        return instance


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        email = data.get("email")
        password = data.get("password")

        if email and password:
            user = authenticate(email=email, password=password)
            if user is None:
                raise serializers.ValidationError("Invalid email or password")
            data["user"] = user
        else:
            raise serializers.ValidationError("Both fields are required")

        return data

class UploadedPDFSerializer(serializers.ModelSerializer):
    pdf_name = serializers.SerializerMethodField()

    class Meta:
        model = UploadedPDF
        fields = ["id", "pdf_file", "pdf_name", "summary", "uploaded_at"]
        read_only_fields = ["id", "summary", "uploaded_at", "pdf_name"]

    def get_pdf_name(self, obj):
        """Extract a meaningful name from the PDF file"""
        # If no file, return a default name
        if not obj.pdf_file:
            return f"PDF #{obj.id}"

        # Get the filename from the path
        file_path = obj.pdf_file.path if hasattr(obj.pdf_file, 'path') else str(obj.pdf_file)
        filename = file_path.split('/')[-1] if '/' in file_path else file_path

        # Remove file extension if present
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]

        # Replace underscores and hyphens with spaces for better readability
        filename = filename.replace('_', ' ').replace('-', ' ')

        # Capitalize words for better presentation
        filename = ' '.join(word.capitalize() for word in filename.split())

        # If filename is still empty after processing, use a default name
        if not filename or filename.strip() == '':
            return f"PDF #{obj.id}"

        return filename



