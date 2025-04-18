"""
Custom middleware for JWT authentication
"""
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.contrib.auth.models import AnonymousUser
from django.utils.functional import SimpleLazyObject

def get_user_jwt(request):
    """
    Attempt to authenticate the user using JWT from various sources
    """
    user = None
    jwt_auth = JWTAuthentication()

    # Try to get the token from various sources
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    else:
        # Try to get from cookie
        token = request.COOKIES.get('access_token', '')

    # If we have a token, try to authenticate
    if token:
        try:
            validated_token = jwt_auth.get_validated_token(token)
            user = jwt_auth.get_user(validated_token)
        except Exception as e:
            print(f"JWT Authentication failed: {str(e)}")

    return user or AnonymousUser()

class JWTAuthenticationMiddleware:
    """
    Middleware that authenticates users via JWT
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Don't auto-authenticate on the welcome page
        if request.path == '/':
            # Let the welcome page be accessible without authentication
            response = self.get_response(request)
            return response

        # If user is already authenticated via session, don't override
        if not hasattr(request, 'user') or request.user.is_anonymous:
            request.user = SimpleLazyObject(lambda: get_user_jwt(request))

        response = self.get_response(request)
        return response
