from django.conf import settings
from django.db import models
from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    head_image = models.ImageField(upload_to='user_head_images/', blank=True, null=True)
    nick_name = models.CharField(max_length=30, blank=True)
    signature = models.TextField(blank=True)
    sex = models.IntegerField(choices=[(0, 'Male'), (1, 'Female')], default=0)


class ChatMessage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    from_user = models.CharField(max_length=50)
    message_type = models.CharField(max_length=50)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.content[:20]}"


class RealTimeAnalysis(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    content = models.TextField()

    def __str__(self):
        return f"Analysis at {self.timestamp}"


# 用于实时分析
class RealtimeAnalysisHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    result = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    screenshot = models.ImageField(upload_to='screenshots/')

    def __str__(self):
        return f"{self.user.username} - {self.content[:20]}"