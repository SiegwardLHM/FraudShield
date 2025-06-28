import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        # 将接收到的数据转换为图像
        np_arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 进行图像处理，例如转换为灰度
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.jpg', processed_img)

        # 发送处理后的图像回客户端
        await self.send(bytes_data=buffer.tobytes())
