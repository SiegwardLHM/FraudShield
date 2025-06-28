<template>
  <el-container class="robot-page">
    <div class="content-container">
      <div class="left-content">
        <div class="title-bar">
          <div class="title-container">
            <span>检测材料</span>
          </div>
        </div>
        <div class="upload-dom">
          <div class="upload-video-box" @click="triggerVideoInput">
            <p v-if="!uploadedTextContent && !uploadedVideoUrl && !videoAudioUrl">点击上传</p>
            <div v-else-if="uploadedTextContent" class="text-content">{{ uploadedTextContent }}</div>
            <video v-else :src="uploadedVideoUrl" controls class="showVideo"></video>
            <input type="file" ref="videoInput" @change="onVideoChange" style="display: none;">
          </div>
        </div>
        <div class="upload-tip">
          请上传视频、音频文件或.txt文件
        </div>
        <div class="divider"></div>
        <div class="feature-selection">
          <h3>选择检测功能</h3>
          <div class="feature-item">
            <div class="feature-name">伪造检测</div>
            <div class="item-container">
              <el-checkbox v-if="uploadedVideoUrl" v-model="faceForgeryDetection">人脸伪造检测</el-checkbox>
              <el-checkbox v-if="uploadedVideoUrl || videoAudioUrl" v-model="audioForgeryDetection">音频伪造检测</el-checkbox>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-name">文本检测</div>
            <div class="item-container">
              <el-checkbox v-if="uploadedVideoUrl || videoAudioUrl || uploadedTextContent"
                v-model="fraudScriptDetection">诈骗话术检测</el-checkbox>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-name">情绪检测</div>
            <div class="item-container">
              <el-checkbox v-if="uploadedVideoUrl" v-model="facialEmotionRecognition">面部情绪识别</el-checkbox>
              <el-checkbox v-if="uploadedVideoUrl || videoAudioUrl"
                v-model="voiceEmotionRecognition">语音情绪识别</el-checkbox>
              <el-checkbox v-if="uploadedVideoUrl" v-model="actionEmotionRecognition">动作情绪识别</el-checkbox>
              <el-checkbox v-if="uploadedVideoUrl || videoAudioUrl || uploadedTextContent"
                v-model="textEmotionRecognition">文本情绪识别</el-checkbox>
            </div>
          </div>
        </div>
        <el-button @click="handleStartMonitoring" class="submit-btn">开始检测</el-button>
        <el-progress :percentage="progress" :status="progressStatus" :text-inside="true" :stroke-width="20"
          class="progress-bar"></el-progress>
      </div>
      <div class="right-content">
        <div class="title-bar">
          <div class="title-container">
            <span>检测结果</span>
          </div>
        </div>
        <div v-if="progressStatus == 'success'" class="result-container">
          <ExpandablePanel :displayContent="displayContent" :isReportButtonDisabled="isReportButtonDisabled">
          </ExpandablePanel>
          <div class="divider"></div>
          <div ref="chart" class="chart"></div>
          <div class="divider"></div>
          <el-button class="submit-btn">下载报告</el-button>
        </div>
      </div>
    </div>
  </el-container>
</template>

<script>
import axios from '../../axios';
import videoPath from "@/assets/video/video.mp4";
import ExpandablePanel from '../../components/ExpandablePanel.vue';
import NERTC from 'nertc-web-sdk';
import * as echarts from 'echarts';

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
      progress: 0, // Track the progress percentage
      isProgressVisible: false, // Track if the progress bar is visible
      progressStatus: 'false', // Track the status of the progress bar
      displayContent: {
        Confidence: {
          fraud: "-1",
          carrier: "0.4",
          video: "-1",
          sound: "-1",
          action: "0.3",
          emotion: "-1",
          soundEmotion: "-1",
          content: "0.5"
        },
        videoText: "待检测",
        video: [
        ],
        sound: [
          "sound1",
          "sound2",
          "sound3"
        ],
        action: [
          "https://www.baidu.com/img/flexible/logo/pc/result.png",
          "action2_url",
          "action3_url"
        ],
        emotion: [
          "emotion1",
          "emotion2",
          "emotion3"
        ],
        soundEmotion: [
          "soundEmotion1",
          "soundEmotion2",
          "soundEmotion3"
        ],
        text: "text1",
        transfer: "transfer"
      },
      uploadedVideoUrl: '', // Stores the URL of the uploaded video for local display
      videoFilePath: '', // Stores the path of the video from the monitored folder
      videoAudioUrl: '',
      uploadHeaders: {
        'Authorization': `Bearer ${localStorage.getItem('access')}`
      },
      monitoringInterval: null, // Store the interval for monitoring
      showJsonAnimation: false, // Track if JSON animation is shown
      chartInstance: null,
      chartData: [
        { time: '00:00', confidence: 0.21, faceForgery: '正常', textDetection: '正常', emotionDetection: '正常' },
        { time: '01:00', confidence: 0.18, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        { time: '02:00', confidence: 0.36, faceForgery: '正常', textDetection: '可疑', emotionDetection: '正常' },
        { time: '03:00', confidence: 0.52, faceForgery: '正常', textDetection: '正常', emotionDetection: '可疑' },
        { time: '04:00', confidence: 0.89, faceForgery: "16%", textDetection: '85%', emotionDetection: '79%' },
        { time: '05:00', confidence: 0.82, faceForgery: '可疑', textDetection: '可疑', emotionDetection: '可疑' },
        { time: '06:00', confidence: 0.71, faceForgery: '可疑', textDetection: '可疑', emotionDetection: '可疑' },
        { time: '07:00', confidence: 0.53, faceForgery: '可疑', textDetection: '可疑', emotionDetection: '可疑' },
        { time: '08:00', confidence: 0.65, faceForgery: '可疑', textDetection: '可疑', emotionDetection: '可疑' },
        { time: '09:00', confidence: 0.32, faceForgery: '可疑', textDetection: '可疑', emotionDetection: '可疑' },
      ],
    }
  },
  mounted() {
    if (!NERTC.checkSystemRequirements()) {
      this.isSupport = false;
    }
    // this.initChart();
    // this.$nextTick(() => {
    //   this.initChart();
    // });
    // this.initChart();
  },
  watch: {
    progressStatus(newVal, oldVal) {
      if (newVal === 'success') {
        this.$nextTick(() => {
          this.initChart();
        });
      }
    },
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

        // // Prepare the form data for uploading the media file
        // const formData = new FormData();
        // formData.append('file', file);

        // try {
        //     const response = await axios.post('/robot/upload-temp/', formData, {
        //         headers: {
        //             'Content-Type': 'multipart/form-data',
        //             ...this.uploadHeaders
        //         }
        //     });
        //     this.fileUrl = response.data.fileUrl || `http://localhost:8000/media/${response.data.filePath}`;
        //     this.fileName = file.name;
        //     console.log('Media file uploaded successfully:', this.fileUrl);
        // } catch (error) {
        //     console.error('Media file upload failed:', error);
        // }
      } else if (fileType === 'text/plain' || file.name.endsWith('.txt')) {
        // Clear any previous media URL and display text snippet
        this.uploadedVideoUrl = '';


        const reader = new FileReader();
        reader.onload = (e) => {
          this.uploadedTextContent = e.target.result.slice(0, 1000); // Display first 100 characters
          console.log(this.uploadedTextContent);
        };
        reader.readAsText(file);

        // // Prepare the form data for uploading the text file
        // const formData = new FormData();
        // formData.append('file', file);

        // try {
        //     const response = await axios.post('/robot/upload-temp/', formData, {
        //         headers: {
        //             'Content-Type': 'multipart/form-data',
        //             ...this.uploadHeaders
        //         }
        //     });
        //     this.fileUrl = response.data.fileUrl || `http://localhost:8000/media/${response.data.filePath}`;
        //     this.fileName = file.name;
        //     console.log('Text file uploaded successfully:', this.fileUrl);
        // } catch (error) {
        //     console.error('Text file upload failed:', error);
        // }
      } else {
        alert('只支持视频、音频文件或.txt文件');
      }
    },

    // Start monitoring folder for new videos with a 7-second delay
    async handleStartMonitoring() {
      if (!this.uploadedVideoUrl && !this.uploadedTextContent) {
        this.$message({
          message: '请先上传文件',
          type: 'warning'
        });
        console.log("错误");
        return;
      }

      console.log("你好");

      this.showJsonAnimation = true; // Show JSON animation
      this.isProgressVisible = true; // Show progress bar
      this.progress = 0; // Reset progress to 0
      this.progressStatus = 'false'; // Reset progress status

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
      // Simulate progress update
      const interval = setInterval(() => {
        if (this.progress >= 100) {
          clearInterval(interval);
          this.progressStatus = 'success'; // Set progress status to success
        } else {
          this.progress += 10; // Increase progress by 10%
        }
      }, 1000); // Update every 1000ms (1 second)

      // try {
      //   const response = await axios.post('/robot/analyze/', { fileUrl: this.fileUrl }, {
      //     headers: {
      //       'Authorization': `Bearer ${localStorage.getItem('access')}`,
      //       'Content-Type': 'application/json'
      //     }
      //   });
      //   this.displayContent.Confidence = response.data.Confidence;
      //   this.displayContent.video = response.data.video;
      //   this.displayContent.action = response.data.action;
      //   this.displayContent.soundEmotion = response.data.soundEmotion;
      //   this.displayContent.emotion = response.data.emotion;
      //   this.displayContent.sound = response.data.sound;
      //   this.displayContent.text = response.data.text;
      //   this.videoFilePath = response.data.videoFilePath;
      //   this.displayContent.videoText = response.data.videoText;
      //   this.displayContent.transfer = response.data.transfer;

      //   console.log('Monitoring started:', response.data.videoFilePath);
      // } catch (error) {
      //   console.error('Analysis failed:', error);
      //   this.progressStatus = 'exception'; // Set progress status to exception
      // }
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
    initChart() {
      const chart = echarts.init(this.$refs.chart);
      const option = {
        title: {
          text: '检测结果流程图',
          left: 'center',
        },
        tooltip: {
          trigger: 'axis',
          formatter: (params) => {
            const param = params[0];
            return `
              <div>
                <p>时间: ${param.name}</p>
                <p>置信度: ${param.value}</p>
                <p>人脸伪造检测: ${param.data.faceForgery}</p>
                <p>文本检测: ${param.data.textDetection}</p>
                <p>情绪检测: ${param.data.emotionDetection}</p>
              </div>
            `;
          },
          backgroundColor: 'rgba(0, 0, 0, 0.5)', // 设置背景颜色为半透明白色
          borderColor: '#ccc', // 设置边框颜色
          borderWidth: 1, // 设置边框宽度
          textStyle: {
            color: '#ffffff', // 设置文本颜色
            fontSize: 15, // 设置文本大小
          },
          padding: 10, // 设置内边距
          extraCssText: 'z-index: 1000;', // 设置 z-index 确保 tooltip 不被遮挡
        },
        xAxis: {
          type: 'category',
          data: this.chartData.map(item => item.time),
          boundaryGap: false,
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 1,
          interval: 1,
          splitNumber: 1,
        },
        series: [
          {
            data: this.chartData.map(item => ({
              value: item.confidence,
              faceForgery: item.faceForgery,
              textDetection: item.textDetection,
              emotionDetection: item.emotionDetection,
            })),
            type: 'line',
            smooth: true,
            symbol: 'circle',
            symbolSize: 10,
            lineStyle: {
              color: "#116FCD",
              width: 2,
            },
            itemStyle: {
              color: "#116FCD",
            },
            areaStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  {
                    offset: 0,
                    color: "rgba(17,111,205,0.3)",
                  },
                  {
                    offset: 1,
                    color: "rgba(17,111,205,0)",
                  },
                ],
              },
            },
          },
        ],
        grid: {
          left: '4%',
          right: '4%',
          bottom: '0',
          containLabel: true,
        },
      };
      chart.setOption(option, true);
      this.chartInstance = chart;
    },
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
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.content-container {
  width: 80%;
  height: 80%;
  padding: 40px 20px;
  margin-left: 4%;
  display: flex;
  justify-content: flex;
  /* align-items: center; */
}

.upload-dom {
  width: 100%;
  height: 25%;
  display: flex;
  /* 使用 Flexbox 布局 */
  justify-content: center;
  /* 水平居中对齐 */
  align-items: center;
  /* 竖直居中对齐 */
}

.upload-video-box {
  /* float: left; */
  margin-top: 3%;
  margin-bottom: 3%;
  width: 80%;
  height: 70%;
  text-align: center;
  font-size: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  border: 3px dashed #DDDDDD;
  border-radius: 5%;
  overflow: hidden;
}

.upload-video-box:hover {
  border: 3px dashed #116FCD;
  /* box-shadow: rgba(0, 0, 0, 0.07) 0px 1px 2px, rgba(0, 0, 0, 0.07) 0px 2px 4px, rgba(0, 0, 0, 0.07) 0px 4px 8px, rgba(0, 0, 0, 0.07) 0px 8px 16px, rgba(0, 0, 0, 0.07) 0px 16px 32px, rgba(0, 0, 0, 0.07) 0px 32px 64px; */
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
  border: 3px dashed #DDDDDD;
  border-radius: 5%;
  overflow: hidden;
}

.display-video-box:hover {
  border: 3px dashed #116FCD;
}

.showVideo {
  width: 100%;
  height: 100%;
  object-fit: cover;
  /* Maintain aspect ratio and avoid overflow */
}

.left-content {
  width: 57%;
  text-align: center;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.1);
  margin-right: 60px;
  border-radius: 8px;
}

.right-content {
  width: 45%;
  text-align: center;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.1);
  border-radius: 8px;
}

.title-bar {
  background-color: #116FCD;
  color: #fff;
  height: 55px;
  font-size: 25px;
  letter-spacing: 1.5px;
  font-weight: bold;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  margin-bottom: 10px;
  display: flex;
  /* 使用 flex 布局 */
  align-items: center;
  /* 竖直居中 */
  padding-left: 1vw;
  /* 横向距离 */
  padding-right: 0.5vw;
  /* 横向距离 */
}

.title-container {
  display: flex;
  /* 使用 flex 布局 */
  width: 60%;
  font-size: min(25px, 3vw);
}

.upload-tip {
  margin-left: 2%;
  font-size: 18px;
  margin-top: 1%;
  margin-right: 2%;
}

/* 新增的样式 */
.divider {
  margin-top: 2vw;
  margin-bottom: 2vw;
  border-top: 1px solid #ccc;
}

.feature-selection {
  text-align: left;
  margin-top: 20px;
}

.feature-selection h3 {
  letter-spacing: 2px;
  font-size: 30px;
  margin-top: 1vw;
  margin-bottom: 1vw;
  font-weight: bold;
  text-align: center;
  /* margin-bottom: 10px; */
}

.feature-item {
  margin-bottom: 30px;
  display: flex;
  align-items: center;
  padding-left: 15px;
}

.feature-name {
  width: 30%;
  font-size: 25px;
  letter-spacing: 1px;
  font-weight: bold;
  /* margin-bottom: 5px; */
  /* margin-bottom: 2px; */
}

.item-container {
  width: 70%;
}

.feature-item .el-checkbox {
  margin-left: 10px;
  margin-right: 40px;
  margin-top: 5px;
  margin-bottom: 5px;
  width: 35%;
}

.feature-item .el-checkbox ::v-deep .el-checkbox__label {
  font-size: 17px;
}

.feature-item .el-checkbox ::v-deep .el-checkbox__inner {
  height: 20px;
  width: 20px;
}

.feature-item .el-checkbox ::v-deep .el-checkbox__inner:after {
  height: 13px;
  width: 7px;
  left: 6px;
}

.submit-btn {
  width: 60%;
  height: max(3vw, 50px);
  border-radius: 10px;
  margin-top: 10px;
  background-color: #116FCD;
  border-color: #116FCD;
  font-size: 20px;
  font-weight: bolder;
  letter-spacing: 1px;
  color: #fff;
}

.submit-btn:hover {
  background-color: #2a82db;
  border-color: #2a82db;
  color: #fff;
}

.progress-bar {
  margin-top: 30px;
  width: 80%;
  margin-left: auto;
  margin-right: auto;
}

.chart {
  width: 100% !important;
  height: 220px;
  margin-top: 20px;
  margin-bottom: 30px;
}
</style>
