from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from django.conf import settings
import os
from ..serializers import UserSerializer, ChangePasswordSerializer

User = get_user_model()

class UserDetailView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        return Response(serializer.data)

class ChangePasswordView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def put(self, request, *args, **kwargs):
        serializer = ChangePasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = request.user

        if not user.check_password(serializer.validated_data['old_password']):
            return Response({"old_password": ["原密码错误。"]}, status=400)

        user.set_password(serializer.validated_data['new_password'])
        user.save()
        return Response({"detail": "密码已修改。"})

class UploadImageView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        file = request.FILES['file']
        user_id = request.user.id
        user_folder = f'user_{user_id}'

        print(settings.MEDIA_ROOT)
        print(user_folder)
        # 确保用户目录存在
        user_directory = os.path.join(settings.MEDIA_ROOT, user_folder)
        if not os.path.exists(user_directory):
            os.makedirs(user_directory)

        # 删除旧的头像文件
        if request.user.head_image:
            # 确保路径正确
            old_file_path = os.path.join(settings.MEDIA_ROOT, request.user.head_image.name.replace("../", ""))
            print("Old file path:", old_file_path)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)

        # 保存新文件
        file_name = os.path.join(user_folder, file.name)
        file_path = default_storage.save(file_name, ContentFile(file.read()))

        # 获取相对路径并生成正确的URL
        relative_file_path = os.path.join(user_folder, os.path.basename(file_path)).replace("\\", "/")
        print(relative_file_path)
        # 确保路径正确拼接
        file_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, relative_file_path).replace("\\", "/"))
        print(file_url)

        # 更新用户的头像信息
        request.user.head_image = relative_file_path
        request.user.save()

        return Response({
            "originUrl": file_url,
            "thumbUrl": file_url
        }, status=status.HTTP_201_CREATED)