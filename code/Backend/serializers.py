from django.conf import settings
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.utils.translation import gettext_lazy as _
from .models import ChatMessage

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    username = serializers.CharField(
        required=True,
        error_messages={
            'required': _('用户名是必须的'),
            'blank': _('用户名不能为空')
        }
    )
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        error_messages={
            'required': _('密码是必须的'),
            'blank': _('密码不能为空')
        }
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        error_messages={
            'required': _('确认密码是必须的'),
            'blank': _('确认密码不能为空')
        }
    )

    class Meta:
        model = User
        fields = ('username', 'password', 'password2')

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password2": _("两次输入的密码不一致")})
        return attrs

    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user


class UserSerializer(serializers.ModelSerializer):
    nickName = serializers.CharField(source='nick_name', required=False)
    headImage = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'username', 'nickName', 'headImage', 'sex', 'signature']

    def get_headImage(self, obj):
        request = self.context.get('request')
        if obj.head_image and hasattr(obj.head_image, 'url'):
            return request.build_absolute_uri(settings.MEDIA_URL + obj.head_image.name)
        return None


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)
    confirm_password = serializers.CharField(required=True)

    def validate(self, attrs):
        if attrs['new_password'] != attrs['confirm_password']:
            raise serializers.ValidationError({"password": "新密码和确认密码不一致。"})
        return attrs


class ChatMessageSerializer(serializers.ModelSerializer):
    from_user = serializers.ReadOnlyField()
    timestamp = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S")

    class Meta:
        model = ChatMessage
        fields = ['from_user', 'message_type', 'content', 'timestamp']
