<template>
  <el-container class="robot-page">
      <div class="header-title">
          <h2>诈骗检测</h2>
      </div>
      <div class="main">
          <div class="video-section">
              <!-- First box: Video Upload -->
              <div class="upload-video-box" @click="triggerVideoInput">
                <p v-if="!uploadedTextContent && !uploadedVideoUrl">点击上传</p>
                <div v-else-if="uploadedTextContent" class="text-content">{{ uploadedTextContent }}</div>
                <video v-else :src="uploadedVideoUrl" controls class="showVideo"></video>
                <input type="file" ref="videoInput"  @change="onVideoChange" style="display: none;">
              </div>
              <!-- Button to start monitoring -->
              <div class="monitor-button">
                  <div class="monitor-icon">
                      <img v-if="!showJsonAnimation" src="@/static/pointer.png" alt="">
                      <video v-else src="@/static/loading5.mp4" autoplay muted loop > </video>
                  </div>
                  <el-button @click="handleStartMonitoring" type="danger" round>开始检测</el-button>
                  <!-- <el-button type="primary" round>Primary</el-button> -->
              </div>

              <!-- Second box: Display monitored video -->
              <div class="display-video-box">
                   <p v-if="!videoFilePath">待检测</p> 
                   <video v-else :src="videoFilePath" controls class="showVideo"></video> 

              </div>
              <ExpandablePanel :displayContent="displayContent" :isReportButtonDisabled="isReportButtonDisabled"></ExpandablePanel>
          </div>
      </div>
  </el-container>
</template>

<script>
import axios from '../../axios';
import videoPath from "@/assets/video/video.mp4";
import ExpandablePanel from '../../components/ExpandablePanel.vue';

export default {
  components: {
    ExpandablePanel, // 注册子组件
  },
  data() {
      return {
          uploadedTextContent: '',
          fileList: [],
          fileUrl: '',
          fileName: '',
          messages: [],
          isReportButtonDisabled: false,
          displayContent:{
            Confidence:{
              fraud:"-1",
              carrier:"0.4",
              video:"-1",
              sound:"-1",
              action:"0.3",
              emotion:"-1",
              soundEmotion:"-1",
              content:"0.5"
            },
            videoText: "待检测",
            video:[
            ],
            sound:[
              "sound1",
              "sound2",
              "sound3"
            ],
            action:[
              "https://www.baidu.com/img/flexible/logo/pc/result.png",
              "action2_url",
              "action3_url"
            ],
            emotion:[
              "emotion1",
              "emotion2",
              "emotion3"
            ],
            soundEmotion:[
              "soundEmotion1",
              "soundEmotion2",
              "soundEmotion3"
            ],
            text:"text1",
            transfer: "transfer"
          },
          uploadedVideoUrl: '', // Stores the URL of the uploaded video for local display
          videoFilePath: '', // Stores the path of the video from the monitored folder
          uploadHeaders: {
              'Authorization': `Bearer ${localStorage.getItem('access')}`
          },
          monitoringInterval: null, // Store the interval for monitoring
          showJsonAnimation: false, // Track if JSON animation is shown
      }
  },
  methods: {
      // Trigger file input click for video upload
      triggerVideoInput() {
          this.$refs.videoInput.click();
      },
      // Handle video file selection and upload
      async onVideoChange(event) {
          const file = event.target.files[0];
          if (!file) return;

          console.log('File selected:', file);

          const fileType = file.type;
          console.log('File type:', fileType);
          if (fileType.startsWith('video/') || fileType.startsWith('audio/')) {
              // Clear any previous text content and display video or audio
              this.uploadedTextContent = '';
              this.uploadedVideoUrl = URL.createObjectURL(file);

              // Prepare the form data for uploading the media file
              const formData = new FormData();
              formData.append('file', file);

              try {
                  const response = await axios.post('/robot/upload-temp/', formData, {
                      headers: {
                          'Content-Type': 'multipart/form-data',
                          ...this.uploadHeaders
                      }
                  });
                  this.fileUrl = response.data.fileUrl || `http://localhost:8000/media/${response.data.filePath}`;
                  this.fileName = file.name;
                  console.log('Media file uploaded successfully:', this.fileUrl);
              } catch (error) {
                  console.error('Media file upload failed:', error);
              }
          } else if (fileType === 'text/plain') {
              // Clear any previous media URL and display text snippet
              this.uploadedVideoUrl = '';

              const reader = new FileReader();
              reader.onload = (e) => {
                  this.uploadedTextContent = e.target.result.slice(0, 1000) ; // Display first 100 characters
              };
              reader.readAsText(file);

              // Prepare the form data for uploading the text file
              const formData = new FormData();
              formData.append('file', file);

              try {
                  const response = await axios.post('/robot/upload-temp/', formData, {
                      headers: {
                          'Content-Type': 'multipart/form-data',
                          ...this.uploadHeaders
                      }
                  });
                  this.fileUrl = response.data.fileUrl || `http://localhost:8000/media/${response.data.filePath}`;
                  this.fileName = file.name;
                  console.log('Text file uploaded successfully:', this.fileUrl);
              } catch (error) {
                  console.error('Text file upload failed:', error);
              }
          } else {
              alert('只支持视频、音频文件或.txt文件');
          }
      },

      // Start monitoring folder for new videos with a 7-second delay
      async handleStartMonitoring() {
            if (!this.fileUrl) {
          this.$message({
            message: '请先上传文件',
            type: 'warning'
          });
            return;
          }
          this.showJsonAnimation = true; // Show JSON animation
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
          this.displayContent.Confidence = response.data.Confidence;
          this.displayContent.video = response.data.video;
          this.displayContent.action = response.data.action;
          this.displayContent.soundEmotion = response.data.soundEmotion;
          this.displayContent.emotion = response.data.emotion;
          this.displayContent.sound = response.data.sound;
          this.displayContent.text = response.data.text;
          this.videoFilePath = response.data.videoFilePath;
          this.displayContent.videoText = response.data.videoText;
          this.displayContent.transfer = response.data.transfer;

          console.log('Monitoring started:', response.data.videoFilePath);
          this.showJsonAnimation = false;
          this.isReportButtonDisabled = false;
          // setTimeout(() => {
          //     this.showJsonAnimation = false; // Hide JSON animation after 7 seconds
          //     this.videoFilePath = videoPath; // Set local video
          // }, 7000);
      },
      formatDate(date) {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      const hours = String(date.getHours()).padStart(2, '0');
      const minutes = String(date.getMinutes()).padStart(2, '0');
      const seconds = String(date.getSeconds()).padStart(2, '0');
      return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
    }
  },
  beforeDestroy() {
      if (this.monitoringInterval) {
          clearInterval(this.monitoringInterval); // Stop monitoring when component is destroyed
      }
  }
};
</script>

<style scoped>
.robot-page {
display: flex;
flex-direction: column;
height: 100%;
}
.header-title {
width: 60%;
height: 14%;
margin-left: 13%;
font-size: 4vh;
font-weight: bold;
border-bottom: 1px dashed gray;
text-align: center;
display: flex;
align-items: center;
justify-content: center;
}
.main {
flex-grow: 1;
height: 80%;
}
.video-section {
display: flex;
margin-top: 5%;
align-items: center;
height: 80%;
flex-grow: 1;
}
.upload-video-box {
float: left;
margin-left: 5%;
width: 30%;
height: 100%;
text-align: center;
display: flex;
align-items: center;
justify-content: center;
cursor: pointer;
border: 3px dashed 	#DDDDDD;
border-radius: 5%;
overflow: hidden;
}

.upload-video-box:hover{
border: 3px dashed 	#FF5511;
box-shadow: rgba(0, 0, 0, 0.07) 0px 1px 2px, rgba(0, 0, 0, 0.07) 0px 2px 4px, rgba(0, 0, 0, 0.07) 0px 4px 8px, rgba(0, 0, 0, 0.07) 0px 8px 16px, rgba(0, 0, 0, 0.07) 0px 16px 32px, rgba(0, 0, 0, 0.07) 0px 32px 64px;
}

.display-video-box {
float: left;
margin-left: 4%;
width: 30%;
height: 100%;
text-align: center;
display: flex;
align-items: center;
justify-content: center;
border: 3px dashed 	#DDDDDD;
border-radius: 5%;
overflow: hidden;
}
.display-video-box:hover{
border: 3px dashed 	#FF5511;
}

.showVideo {
width: 100%;
height: 100%;
object-fit: cover; /* Maintain aspect ratio and avoid overflow */
}


.monitor-icon video{

  width: 100%;
  height: 100%;
  object-fit:cover;
  border: none;
  background-color: white;
  
  object-fit: contain;
  outline: none !important;

  mix-blend-mode: darken;

}

.monitor-button{
height: 20vh;
float: left;
margin-left: 4%;
}
.monitor-icon{
height: 10vh;
text-align: center;
display: flex;
align-items: center;
justify-content: center;
margin-bottom: 50%;
}
.monitor-icon img{
max-width: 100%;
max-height: 100%;
object-fit: contain;
}

</style>
