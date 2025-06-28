<template>
  <div>
    <!--  -->

    <el-button :disabled="isReportButtonDisabled" type="info" class="icon-button" @click="togglePanel"
      round>查看检测报告</el-button>
    <!-- v-if="isPanelVisible" -->
    <div class="overlay" v-if="isPanelVisible">
      <div class="panel">
        <div class="close-button" @click="togglePanel">×</div>
        <!-- Banner start -->
        <div class="top-bar">
          <div class="bar-title">诈骗风险摘要报告</div>
        </div>
        <div class="container">
          <div class="all-content">
            <!-- Main content -->
            <div class="content-wrapper">
              <div class="left-content">
                <!-- <div class="data-title">
                      <span class="data-title-text" style="font-size: 3vh;transform: translateY(60px);">诈骗概率{{ displayContent.Confidence.fraud*100 }}</span>
                    </div> -->
                <div class="title-bar">
                  诈骗数据
                </div>
                <div class="data-fruad"
                  style="justify-content: center;align-items: center;display: flex;color: white; ">
                  <!-- <span class="data-title-text">诈骗概率{{ displayContent.Confidence.fraud * 100 }}</span> -->
                  <span class="data-title-text">诈骗概率{{ 0.8892 * 100 }}</span>
                </div>
                <div class="project-overview" style="text-align: center;">
                  <!-- Each pie chart display -->
                  <div class="chart-display" v-show="displayContent.Confidence.carrier * 100 > 0">
                    <span class="overview-title">通话载体</span>
                    <canvas id="pie1" width="130" height="130"></canvas>
                  </div>

                  <div class="chart-display" v-show="displayContent.Confidence.action * 100 > 0">
                    <span class="overview-title">通话行为</span>
                    <canvas id="pie2" width="130" height="130"></canvas>
                  </div>

                  <div class="chart-display" v-show="displayContent.Confidence.content * 100 > 0">
                    <span class="overview-title">{{ displayContent.Confidence.action > 0 ? '通话语义' : '内容语义' }}</span>
                    <canvas id="pie3" width="130" height="130"></canvas>
                  </div>
                </div>
              </div>

              <!-- Center display -->
              <div class="mid-content">
                <div class="title-bar">
                  诈骗语义分析
                </div>
                <p class="info-text" style="white-space:pre-line;overflow-y: auto;height: 100%;padding-bottom: 20px;">
                  <!-- {{ displayContent.text }} -->
                  
                </p>
              </div>

              <!-- Right display -->
              <div class="right-content">
                <div class="title-bar">
                  诈骗载体 / 行为分析
                </div>

                <div class="inspection-status">
                  <div v-if="displayContent.video.length > 0">
                    <!-- <div class="video-block">
                      <div v-for="(url, index) in displayContent.video" :key="index" class="video-display">
                        <div class="text-content">{{ imgLabel[index] }}</div>
                        <img :src="url" :alt="url" />
                      </div>
                    </div>
                    <div class="video-block">
                      <div v-for="(url, index) in displayContent.action" :key="index" class="video-display">
                        <div class="text-content">{{ imgLabel[index + 3] }}</div>
                        <img :src="url" :alt="url" />
                      </div>
                    </div> -->
                  </div>
                  <!-- <p v-else class="info-text" style="white-space:pre-line;">{{ displayContent.transfer }}</p> -->
                </div>
              </div>
            </div>

            <div class="chart-content">
              <div class="timeline-chart" ref="timelineChart"></div>
            </div>
          </div>

          <!-- 新增的 robot-chat -->
          <div class="robot-chat">
            <div class="chat-title">智能诈骗风险摘要助手</div>
            <div class="chat-box">
                <div class="message bot">您好！我是通话分析助手，有什么可以帮助您的吗？</div>
                <div class="message user">我想知道本次通话的风险点，请详细具体地解释一下。</div>
                <div class="message bot">
                  您好 😊 ，根据对本次通话的深入分析，我发现了一些需要引起您重视的异常情况，以下是详细的分析结果：🔎 
                  <br>
                  <br>
                  1. 通话载体存在伪造迹象
                  在整段通话过程中，我检测到对方的语音和人脸存在伪造的嫌疑。通过音频和视频特征比对，系统识别出对方的声音可能经过音频合成或调制，且面部表情在多次动态比对中存在不一致性，这可能是因为深度伪造（DeepFake）或类似技术所导致。这种伪造贯穿了整个通话过程，提示该通话可能涉及高度欺诈行为。
                  <br>
                  <br>
                  2. 行为和情绪存在异常
                  在4到8秒这一时间段内，对方的行为出现了多次不自然的表现。通过面部微表情和行为动态分析，系统检测到对方存在以下异常：
                  <br>
                  <br>
                  频繁触摸鼻子和眼神飘忽不定，这些行为通常与说谎或紧张心理相关。
                  在上述时间段内，系统识别出对方的面部肌肉活动存在非自然收缩，尤其是在嘴角和眼部区域，提示存在微表情不协调现象。
                  对方的语气在这一时段内出现了明显波动，表现为不稳定和紧张情绪，结合面部表情和动作特征，系统判定对方在此时段存在较高的欺诈或隐瞒信息的可能性。
                  经过对通话内容和异常行为的联合分析，系统认为在这一时间段内，对方的行为与其言语内容存在明显的不一致，进一步增加了诈骗行为的可疑度。
                  <br>
                  <br>
                  3. 通话内容存在异常和潜在的诈骗特征
                  在通话过程中，对方自称是抖音平台的客服人员，但系统通过语义分析发现了以下异常点：
                  抖音平台的官方服务渠道通常不会通过私人电话或非官方应用与用户联系。
                  对方提到与财务或续费相关的内容，这在官方渠道中极为罕见，且未能提供权威身份验证。
                  最值得警惕的是，对方在通话中要求您下载一款定位类应用。根据安全协议，正规客服在与用户交互时，不会主动提出此类涉及隐私和位置信息的要求。这种行为通常与位置追踪、信息窃取或后续的恶意操作相关。
                  <br>
                  <br>
                  综上所述
                  系统在结合行为特征、语音和面部分析、情绪波动以及通话内容的上下文匹配后，判定本次通话存在较高的诈骗可能性。系统已为您标记出异常时间段（第4到8秒），并根据跨模态分析（语音 + 行为 +
                  内容）进一步确认了对方存在深度伪造和不一致行为。建议您保持警惕，避免下载或运行对方提供的任何第三方应用，以防止潜在的信息泄露或财务损失。
                  <br>
                  <br>
                  如果需要进一步的安全提示或帮助，我随时为您提供支持。</div>
            </div>
            <div class="chat-input">
              <button>
                <img src="/src/assets/img/icon/audio.png" class="audio-button" alt="audio icon">
              </button>
              <textarea v-model="userInput" placeholder="请输入消息..."></textarea>
              <button @click="sendMessage">发送</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Chart from '@/assets/styles/acc';
import * as echarts from 'echarts';
import axios from '../axios';

export default {
  components: {
    Chart
  },
  data() {
    return {
      isPanelVisible: false,
      panelWidth: 1200,
      panelHeight: 800,
      imgLabel: ['人脸伪造', '表情心虚', '摸鼻子', '摸脖子', '斜视', '摸下巴'],
      timelineChartData: [
        { time: '00:00', confidence: 0.12, faceForgery: '正常', textDetection: '正常', emotionDetection: '正常' },
        { time: '01:00', confidence: 0.21, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        { time: '02:00', confidence: 0.32, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        { time: '03:00', confidence: 0.15, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        { time: '04:00', confidence: 0.46, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        { time: '05:00', confidence: 0.78, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        { time: '06:00', confidence: 0.69, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' },
        // 其他数据...
      ],
      timelineChartInstance: null,
      userInput: '', // 用户输入的消息
    };
  },
  props: {
    displayContent: {

    },
    isReportButtonDisabled: {
      type: Boolean,
      default: true
    }
  },
  methods: {
    togglePanel() {
      this.isPanelVisible = !this.isPanelVisible;
      console.log(this.isPanelVisible);
    },
    initTimelineChart() {
      const chart = echarts.init(this.$refs.timelineChart);
      const option = {
        title: {
          text: '检测时间轴图',
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
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          borderColor: '#ccc',
          borderWidth: 1,
          textStyle: {
            color: '#ffffff',
            fontSize: 15,
          },
          padding: 10,
          extraCssText: 'z-index: 1000;',
        },
        xAxis: {
          type: 'category',
          data: this.timelineChartData.map(item => item.time),
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
            data: this.timelineChartData.map(item => ({
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
              color: '#116FCD',
              width: 2,
            },
            itemStyle: {
              color: '#116FCD',
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
                    color: 'rgba(17, 111, 205, 0.3)',
                  },
                  {
                    offset: 1,
                    color: 'rgba(17, 111, 205, 0)',
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
      this.timelineChartInstance = chart;
    },
  },
  // mounted() {
  //   // 其他挂载逻辑...
  //   this.initTimelineChart();
  // },
  sendMessage() {
    if (this.userInput.trim()) {
      // 模拟用户消息
      const userMessage = document.createElement('div');
      userMessage.className = 'message user';
      userMessage.textContent = this.userInput;

      // 模拟机器人回复
      const botMessage = document.createElement('div');
      botMessage.className = 'message bot';
      botMessage.textContent = '这是机器人的回复：' + this.userInput;

      // 将消息添加到聊天框
      const chatBox = document.querySelector('.chat-box');
      chatBox.appendChild(userMessage);
      chatBox.appendChild(botMessage);

      // 清空输入框
      this.userInput = '';

      // 滚动到底部
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  },
  adjustHeight(event) {
    const textarea = event.target;
    textarea.style.height = 'auto'; // 重置高度
    textarea.style.height = `${textarea.scrollHeight}px`; // 根据内容调整高度
  },
  watch: {
    isPanelVisible(newVal) {
      if (newVal) {
        this.$nextTick(() => {
          // new Chart("pie1").ratePie(this.displayContent.Confidence.carrier * 100);
          // new Chart("pie2").ratePie(this.displayContent.Confidence.action * 100);
          // new Chart("pie3").ratePie(this.displayContent.Confidence.content * 100);
          new Chart("pie1").ratePie(0.96 * 100);
          new Chart("pie2").ratePie(0.71 * 100);
          new Chart("pie3").ratePie(0.86 * 100);
          this.initTimelineChart();
        });
      }
    }
  }
};
</script>

<style scoped>
/* 样式保持不变 */
.icon-button {
  width: 60%;
  height: max(3vw, 50px);
  border-radius: 3px;
  margin-top: 3vw;
  margin-bottom: 1vw;
  background-color: #116FCD;
  border-color: #116FCD;
  font-size: 20px;
  font-weight: bolder;
  letter-spacing: 1px;
  color: #fff;
}

.icon-button:hover {
  background-color: #2a82db;
  border-color: #2a82db;
  color: #fff;
}

.overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  /* 固定overlay的大小 */
  max-width: 100vw;
  max-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  z-index: 1000;
  overflow: hidden;
  /* 防止内容溢出 */
}

.panel {
  /* background-color: #fff; */
  background-image: url('@/assets/img/background.jpg');
  /* background-image: url('@/assets/img/bg-fade.png'); */
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  /* opacity: 0.8; */
  /* border-radius: 8px; */
  /* box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); */
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  height: 100%;
}

.close-button {
  position: absolute;
  right: 30px;
  cursor: pointer;
  font-size: 40px;
  color: white;
}

* {
  padding: 0;
  margin: 0;
  font-family: "微软雅黑";

  font-style: normal;
}

a {
  text-decoration: none;
}

img {
  border: 0;
}

ul,
li {
  padding: 0;
  margin: 0;
  list-style: none;
}

.float-left {
  float: left;
}

.float-right {
  float: right;
}

body {
  margin: 0;
  padding: 0;
  color: #000;
  background-color: #000c3b;
}

.header-title {
  font-size: 4vh;
  font-family: "黑体";
  color: #fff;
  max-width: 900px;
  margin: 10px auto;
  height: 48px;
}

.main-title {
  line-height: 48px;
  display: inline-block;
  width: 276px;
  text-align: center;
  font-size: 28px;
}

.data-box {
  border: 2px solid #032d60;
  box-shadow: inset 0 0 30px #07417a;
  position: relative;
  text-align: center;
}

.data-title {
  text-align: center;
  background-color: #000c3b;
  margin: -18px 0 0;
  display: inline-block;
  color: #83c7e3;
  font-size: 20px;

  transform: translateY(0.6vh);
}

.data-fruad {
  width: 100%;
  height: 4%;
}

.data-title-text {
  text-align: center;
  font-size: 20px;
  color: #333;
  display: center;
  justify-content: center;
}

.project-overview {
  width: 100%;
  color: #fff;
  padding: 0 0;
  font-size: 3vh;
}

.overview-title {
  height: 100%;
  padding: 7px;
  display: block;
  border-top: 1px #ccc solid;
  color: #333;
  font-size: 2vh;

}

.video-block {

  width: 100%;
  margin-top: 2vh;
  height: 38vh;
  display: flex;
}

.video-display {
  height: 100%;
  width: 30%;
  display: flex;
  margin-left: 3%;
  float: left;

  /* flex-direction: column;  */
}

.video-display img {
  width: 100%;
  height: auto;
  min-height: 90%;
  margin-top: 10%;
  object-fit: contain;

}

.info-text {
  font-size: 20px;
  color: black;
  white-space: pre-line;
  overflow-y: auto;
}

/* 新增的 top-bar 样式 */
.top-bar {
  background-color: #116FCD;
  height: 6%;
  /* 页面高度的 10% */
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #fff;
  font-family: "黑体";
  box-shadow: 0px 10px 9px 4px rgba(47, 56, 111, 0.5);
  /* margin-bottom: 1%; */
}

/* bar-title 样式 */
.bar-title {
  color: #116FCD;
  font-size: 35px;
  font-weight: bold;
  letter-spacing: 2px;
  font-family: "黑体";
  width: 25%;
  background-color: #fff;
  height: 100%;
  align-items: center;
  justify-content: center;
  display: flex;
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

.all-content {
  width: 70%;
  height: 100%;
  flex-direction: column;
  align-items: center;
  display: flex;
  /* justify-content: center; */
  /* justify-content: space-around; */
}

.content-wrapper {
  width: 90%;
  height: 72%;
  padding-left: 10px;
  padding-right: 10px;
  overflow: hidden;
  padding-top: 30px;
  justify-content: space-around;
  display: flex;
}

.left-content {
  width: 20%;
  height: 95%;
  text-align: center;
  align-items: flex;
  justify-content: flex;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.3);
  margin-right: 20px;
  margin-left: 20px;
  border-radius: 8px;
}

.mid-content {
  width: 40%;
  height: 95%;
  text-align: center;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.3);
  margin-right: 20px;
  margin-left: 20px;
  border-radius: 8px;
}

.right-content {
  width: 40%;
  height: 95%;
  text-align: center;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.3);
  margin-right: 20px;
  margin-left: 20px;
  border-radius: 8px;
}

.timeline-chart {
  width: 90%;
  height: 90%;
}

.chart-content {
  width: 87%;
  height: 20%;
  text-align: center;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.3);
  justify-content: space-around;
  padding-top: 10px;
  display: flex;
  overflow: hidden;
  border-radius: 8px;
}

.container {
  width: 100%;
  height: 100%;
  display: flex;
  /* 使用 Flexbox 布局 */
  flex-direction: row;
  /* 子元素水平排列 */
  justify-content: space-between;
  /* 子元素之间有间距 */
}

.robot-chat {
  width: 30%;
  /* 可根据需要调整宽度 */
  height: 100%;
  max-height:900px;
  /* 高度占据整个页面 */
  background-color: #fff;
  /* 背景颜色 */
  /* border-left: 7px solid #1567bb; 添加边框，颜色为 #1567bb */
  /* border-radius: 8px; 圆角 */
  box-shadow: -4px 4px 8px rgba(0, 0, 0, 0.2);
  /* 阴影效果 */
  /* overflow-y: auto; */
  /* 内容超出时显示滚动条 */
  padding: 20px;
  /* 内边距 */
}

/* 标题样式 */
.robot-chat .chat-title {
  font-size: 24px;
  font-weight: bold;
  color: #333;
  margin-bottom: 20px;
  text-align: center;
  letter-spacing: 1px;
}


/* 模拟与大语言模型交流的界面 */
.robot-chat .chat-box {
  width: 100%;
  height: 83%;
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  /* 每个消息之间的间距 */
}

.robot-chat .message {
  padding: 10px;
  border-radius: 8px;
  max-width: 85%;
  /* 消息的最大宽度 */
  white-space: pre-wrap;
  word-wrap: normal;  
  text-align: left;
}

.robot-chat .message.user {
  background-color: #e0e0e0;
  /* 用户消息的颜色 */
  align-self: flex-end;
  /* 用户消息靠右对齐 */
}

.robot-chat .message.bot {
  background-color: #d0e6ff;
  /* 机器人消息的颜色 */
  align-self: flex-start;
  word-wrap: normal;  
  /* 机器人消息靠左对齐 */
}

/* 输入框和发送按钮样式 */
.robot-chat .chat-input {
  display: flex;
  margin-top: 20px;
}

.robot-chat .chat-input textarea {
  flex: 1;
  padding: 10px;
  border: 2px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
  margin-right: 10px;
  resize: none;
  /* 禁止手动调整大小 */
  min-height: 50px;
  /* 最小高度 */
  max-height: 150px;
  /* 最大高度 */
  overflow-y: auto;
  /* 超出最大高度时显示滚动条 */
  white-space: pre-wrap;
  /* 自动换行 */
  word-wrap: break-word;
  /* 长单词自动换行 */
}

.robot-chat .chat-input button {
  padding: 10px 20px;
  margin-right: 10px;
  background-color: #1567bb;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}

.robot-chat .chat-input button:hover {
  background-color: #2a82db;
}

.audio-button{
  width: 35px;
  height: 35px;
}
</style>
