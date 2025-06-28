from django.urls import re_path
from users import consumers

websocket_urlpatterns = [
    re_path(r'ws/video/$', consumers.VideoConsumer.as_asgi()),
]
