from django.urls import path
from .views import welcome, history_page, user_profile, get_csrf_token, about_app, upload_pdf, chatbot_page, login_page, RegisterView, register_page, LoginView, LogoutView, UserProfileView, UploadPDFView, UserHistoryListView, delete_pdf
from django.conf.urls.static import static
from django.conf import settings
from .views import chatbot_api, translate_summary, available_languages


urlpatterns = [

    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("profile/", UserProfileView.as_view(), name="profile"),  # Protected Route
    path("upload/", UploadPDFView.as_view(), name="upload_pdf"),
    path("chatbot/", chatbot_api, name="chatbot"),
    path('history/', UserHistoryListView.as_view(), name='user_history'),
    path('delete_pdf/<int:pdf_id>/', delete_pdf, name='delete_pdf'),



    path("register_page/",register_page , name="register_page"),
    path("login_page/",login_page,name="login_page"),
    path('about/', about_app, name='about'),
    path('upload_pdf/',upload_pdf, name="upload_pdf" ),
    path("chatbot_page/", chatbot_page, name="chatbot-page"),
    path("history_page/", history_page, name="history_page"),
    path('profile_page',user_profile, name="profile_page"),


    path("csrf/", get_csrf_token, name="csrf_token"),

    # Translation endpoints
    path("translate-summary/", translate_summary, name="translate_summary"),
    path("available-languages/", available_languages, name="available_languages"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)