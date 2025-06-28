<template>
  <el-container class="robot-page">
    <el-header class="header-title">
      <h2>检测机器人</h2>
    </el-header>
    <el-main>
      <el-scrollbar class="chat-messages">
        <div v-for="(message, index) in messages" :key="index" :class="{'message-container': true, 'left-container': message.from === 'robot', 'right-container': message.from === 'user'}">
          <div v-if="message.from === 'robot'" class="avatar-container">
            <img src="@/static/icon/robot.png" class="avatar" alt="机器人头像">
          </div>
          <div v-if="message.from === 'user'" class="avatar-container user-avatar-container">
            <img :src="userInfo.headImage" class="avatar" alt="用户头像">
          </div>
          <div class="message-wrapper">
            <el-card shadow="never" :class="{'robot-card': message.from === 'robot', 'user-card': message.from === 'user', 'left': message.from === 'robot', 'right': message.from === 'user'}">
              <p v-if="message.from === 'user'">{{ message.fileName }}</p>
              <div v-if="message.from === 'robot'">
                <p v-html="message.text"></p>
              </div>
            </el-card>
            <small class="time-stamp" :class="{'user-time': message.from === 'user', 'robot-time': message.from === 'robot'}">{{ message.time }}</small>
          </div>
        </div>
      </el-scrollbar>
    </el-main>
    <div v-if="showIcon" class="scroll-to-bottom-icon" @click="scrollToBottom">
      <el-icon><Bottom /></el-icon>
    </div>
    <el-footer class="upload-footer" @click="triggerFileInput">
      <div class="upload-left">
        <input type="file" ref="fileInput" accept=".txt,.mp4,.wav,.png" @change="onFileChange" style="display: none;">
        <img src="@/static/icon/upload.png" class="upload-icon" alt="上传文件">
        <div v-if="fileName" class="selected-file">{{ fileName }}</div>
      </div>
      <div class="upload-hint">选择文件发送...</div>
      <div class="upload-right">
        <el-button @click.stop="sendFile" type="success" class="send-button">发送</el-button>
      </div>
    </el-footer>
  </el-container>
</template>

<script>
import axios from '../../axios'; // 导入配置好的axios实例
import { ElLoading } from 'element-plus';
import { Bottom } from '@element-plus/icons-vue';
import * as echarts from 'echarts';

export default {
  data() {
    return {
      showIcon: false,
      fileList: [],
      fileUrl: '',
      fileName: '',
      messages: [],
      activeStep: 0,
      userInfo: {
        headImage: ''
      },
      uploadHeaders: {
        'Authorization': `Bearer ${localStorage.getItem('access')}`
      },
      steps: [
        { text: '选择文件' },
        { text: '上传文件' },
        { text: '分析文件' }
      ]
    }
  },
  components: {
    Bottom
  },
  methods: {
    scrollToBottom() {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: 'smooth'
      });
    },
    handleScroll() {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
      const documentHeight = document.documentElement.scrollHeight;

      // 隐藏图标当接近底部时
      this.showIcon = scrollTop + viewportHeight < documentHeight - 50;
    },
    triggerFileInput() {
      this.$refs.fileInput.click();
    },
    async onFileChange(event) {
      const file = event.target.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await axios.post('/robot/upload-temp/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            ...this.uploadHeaders
          }
        });
        this.fileUrl = response.data.fileUrl || `https://114.132.166.176/media/${response.data.filePath}`;
        this.fileName = file.name;
        console.log('File uploaded successfully:', this.fileUrl);
      } catch (error) {
        console.error('Error uploading file:', error);
        this.$message.error('文件上传失败，请重试');
      }
    },
    async sendFile() {
      if (!this.fileUrl) {
        this.$message({
          message: '请先上传文件',
          type: 'warning'
        });
        return;
      }

      const loadingInstance = ElLoading.service({ fullscreen: true, text: '分析中，请稍候...' });

      try {
        const userMessageTime = new Date();
        const userTimeString = this.formatDate(userMessageTime);

        const userMessage = {
          from: 'user',
          type: 'file',
          fileName: this.fileName,
          url: this.fileUrl,
          time: userTimeString
        };

        this.messages.push(userMessage);

        const response = await axios.post('/robot/analyze/', { fileUrl: this.fileUrl }, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access')}`,
            'Content-Type': 'application/json'
          }
        });

        const robotMessageTime = new Date();
        const robotTimeString = this.formatDate(robotMessageTime);

        const robotMessage = {
          from: 'robot',
          type: 'text',
          text: `分析结果: ${response.data.result.replace(/\n/g, '<br>')}`, // 替换换行符为HTML的<br>标签
          showChart: this.isAudioOrVideoFile(this.fileName),
          chartData: exampleChartData,
          time: robotTimeString
        };

        this.messages.push(robotMessage);

        await this.saveChatHistory();

        this.fileUrl = '';
        this.fileName = '';
        this.fileList = [];
        this.activeStep = 0;
      } catch (error) {
        console.error('Error analyzing file:', error);
        this.$message.error('文件分析请求失败，请重试');
      } finally {
        loadingInstance.close();
      }
    },
    isAudioOrVideoFile(fileName) {
      const fileExtension = fileName.split('.').pop().toLowerCase();
      return ['mp4', 'wav'].includes(fileExtension);
    },
    async saveChatHistory() {
      try {
        await axios.post('/robot/chat-history/', this.messages, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access')}`,
            'Content-Type': 'application/json'
          }
        });
        console.log('Chat history saved successfully');
      } catch (error) {
        console.error('Error saving chat history:', error);
      }
    },
    async loadChatHistory() {
      try {
        const response = await axios.get('/robot/chat-history/', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access')}`
          }
        });
        if (Array.isArray(response.data)) {
          this.messages = response.data.map(msg => ({
            from: msg.from_user,
            type: msg.message_type,
            text: msg.message_type === 'text' ? msg.content : '',
            fileName: msg.message_type !== 'text' ? msg.content : '',
            time: msg.timestamp,
          }));
          console.log('Chat history loaded successfully');
        } else {
          this.messages = [];
          console.error('Unexpected data format:', response.data);
          this.$message.error('加载聊天记录失败，返回数据格式不正确');
        }
      } catch (error) {
        console.error('Error loading chat history:', error);
        this.$message.error('加载聊天记录失败，请重试');
      }
      
      if (this.messages.length === 0) {
        const now = new Date();
        const timeString = this.formatDate(now);
        const welcomeMessage = {
          from: 'robot',
          type: 'text',
          text: '欢迎使用检测机器人，上传文件即可得到分析结果！',
          time: timeString
        };
        this.messages.push(welcomeMessage);
        await this.saveChatHistory();
      }
    },
    formatDate(date) {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      const hours = String(date.getHours()).padStart(2, '0');
      const minutes = String(date.getMinutes()).padStart(2, '0');
      const seconds = String(date.getSeconds()).padStart(2, '0');
      return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
    },
    async fetchUserInfo() {
      try {
        const response = await axios.get('/auth/me/');
        this.userInfo = response.data;
        localStorage.setItem('userInfo', JSON.stringify(this.userInfo));
      } catch (error) {
        console.error('Error fetching user info:', error);
      }
    },
  },
  created() {
    this.loadChatHistory();
    this.fetchUserInfo();
    window.addEventListener('scroll', this.handleScroll);
  },
  beforeDestroy() {
    // 移除滚动事件监听器
    window.removeEventListener('scroll', this.handleScroll);
  }
};
</script>

<style scoped>
html, body, #app {
  height: 100%;
  margin: 0;
}
.robot-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: #f5f5f5;
  border: 1px solid gray;
}
.header-title {
  width: 60%;
  margin-left: 13%;
  text-align:center;
  font-size: 24px;
  font-weight: bold;
}
.el-header, .el-footer {
  flex-shrink: 0;
}
.el-main {
  flex-grow: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  margin-top: 20%;
}
.chat-messages {
  flex-grow: 1;
  padding: 20px 0 20px 0;
  background-color: #ffffff;
  margin-bottom: 45%;
}
.message-container {
  display: flex;
  align-items: flex-start;
  margin-bottom: 20px;
}
.left-container {
  flex-direction: row;
  align-items: flex-start;
}
.right-container {
  flex-direction: row-reverse;
  align-items: flex-start;
}
.message-wrapper {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}
.left-container .message-wrapper {
  align-items: flex-start;
}
.right-container .message-wrapper {
  align-items: flex-end;
}
.left {
  background-color: #e0f7fa;
}
.right {
  background-color: #009688;
  color: #ffffff;
}
.avatar-container {
  display: flex;
  align-items: flex-start;
  margin-left: 10px;
}
.right-container .avatar-container {
  margin-left: 0;
  margin-right: 10px;
}
.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
}
.time-stamp {
  font-size: 12px;
  color: #888;
  margin-top: 5px;
}
.user-time {
  align-self: flex-end;
}
.robot-time {
  align-self: flex-start;
}
.robot-card {
  margin: 0 10px;
  padding: 0 10px;
  border-radius: 15px;
}
.user-card {
  margin-right: 10px;
  padding: 0 10px;
  border-radius: 15px;
}
.robot-card /deep/ .el-card__body, .user-card /deep/ .el-card__body {
  padding: 0 !important;
}
.upload-footer {
  position: fixed;
  bottom: 0;
  width: 100%;
  background: #F8F7F7;
  padding-bottom: 30%;
  padding-top: 10%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
}
.upload-left {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.upload-icon {
  width: 50px;
  height: 50px;
}
.upload-hint {
  font-size: 18px;
  color: #888;
  text-align: center;
}
.selected-file {
  font-size: 14px;
  color: #333;
  margin-top: 10px;
}
.send-button {
  width: 130px;
  height: 50px;
  font-size: 20px;
  background-color: #2A395A;
  border-color: #fff;
  cursor: pointer;
  border-radius: 10px;
}
.send-button:hover {
  background-color: #2A395A;
  border-color: #fff;
}
.scroll-to-bottom-icon {
  position: fixed;
  bottom: 20%;
  right: 40%;
  font-size: 20px;
  cursor: pointer;
  background-color:#2A395A;  ;
  border-radius: 48%;
  padding: 10px;
  color: rgb(255, 255, 255);
}

.scroll-to-bottom-icon:hover {
  background-color: rgb(253, 253, 253);
  color: #111;
}
</style>