from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from .views.register_login import RegisterView, CustomTokenObtainPairView
from .views.profile import UserDetailView, ChangePasswordView, UploadImageView
from .views.robot import TempUploadView, AnalyzeView, ChatHistoryView
from .views.realtime_audio import RealTimeAudioAnalysisView
from .views.realtime_video import RealTimeVideoAnalysisView

urlpatterns = [
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/me/', UserDetailView.as_view(), name='user-detail'),
    path('auth/change-password/', ChangePasswordView.as_view(), name='change-password'),
    path('auth/upload/', UploadImageView.as_view(), name='file-upload'),
    path('robot/upload-temp/', TempUploadView.as_view(), name='robot-upload-temp'),
    path('robot/analyze/', AnalyzeView.as_view(), name='robot-analyze'),
    path('robot/chat-history/', ChatHistoryView.as_view(), name='chat-history'),
    path('robot/real-time-analysis/audio/', RealTimeAudioAnalysisView.as_view(), name='real_time_audio_analysis'),
    path('robot/real-time-analysis/video/', RealTimeVideoAnalysisView.as_view(), name='real_time_video_analysis'),
]
