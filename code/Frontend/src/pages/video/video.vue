<template>
  <div class="join">
    <div class="content-container">
      <!-- 左边模块框 -->
      <div class="left-content">
        <div class="title-bar">
          <div class="title-container">
            <span v-if="isVideo">视频通话</span>
            <span v-else>语音通话</span>
          </div>
            <div class="switch-container">
              <span v-if="isVideo">切换至<br>语音通话</span>
              <span v-else>切换至<br>视频通话</span>
              <el-switch
              v-model="isVideo"
              style="margin-left: 0px;
                    --el-switch-on-color: #c43d60; 
                    --el-switch-off-color: #414656;
                    zoom:1.5;"                    
              size="large"
              />
            </div>
        </div>
        <el-input
          v-model.trim="channelName"
          placeholder="请输入房间号"
          class="input"
        >
          <template #prefix>
            <el-icon>
              <House />
            </el-icon>
          </template>
        </el-input>
        <el-select
          v-model="channelName"
          placeholder="请选择房间号"
          class="select-channel"
        >
          <el-option
            v-for="channel in recentChannels"
            :key="channel"
            :label="channel"
            :value="channel"
          />
        </el-select>
        <div class="add">
          <el-button
            :disabled="!isSupport"
            class="submit-btn"
            type="primary"
            @click="handleSubmit"
            >加入房间</el-button>
        </div>
        <div class="errorMsg" v-show="!isSupport">
          当前浏览器不支持体验，建议下载安装最新chrome浏览器
        </div>
        <div class="divider"></div>
        <div class="feature-selection">
          <h3>选择功能</h3>
          <div class="feature-item">
            <div class="feature-name">伪造检测</div>
            <div class="item-container">
              <el-checkbox v-if="isVideo" v-model="faceForgeryDetection">人脸伪造检测</el-checkbox>
              <el-checkbox v-model="audioForgeryDetection">音频伪造检测</el-checkbox>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-name">文本检测</div>
            <div class="item-container">
              <el-checkbox v-model="fraudScriptDetection">诈骗话术检测</el-checkbox>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-name">情绪检测</div>
            <div class="item-container">
              <el-checkbox v-if="isVideo" v-model="facialEmotionRecognition">面部情绪识别</el-checkbox>
              <el-checkbox v-model="voiceEmotionRecognition">语音情绪识别</el-checkbox>
              <el-checkbox v-if="isVideo" v-model="actionEmotionRecognition">动作情绪识别</el-checkbox>
              <el-checkbox v-model="textEmotionRecognition">文本情绪识别</el-checkbox>
            </div>
          </div>
        </div>
        <div class="advice">
            <el-button
              class="submit-btn"
              type="primary"
              @click="selectAllFeatures"
            >勾选推荐选项</el-button>
        </div>
      </div>

      <!-- 右边模块框 -->
      <div class="right-content">
        <div class="notes notes-first">
          <div class="title-bar">往期检测结果</div>
          <div v-if="isVideo" class="note1-text">
            上次视频通话时间: 2025年2月3日12:08
          </div>
          <div v-else class="note1-text">
            上次音频通话时间: 2025年2月3日12:08
          </div>
          <div id="main">
            <div ref="chart" class="chart"></div>
          </div>
        </div>
        <div class="notes notes-second">
          <div class="title-bar">防诈热点</div>
          <div class="news-container">
            <div class="left-news">
              <div v-if="currentImageIndex == 0" class="image-container">
                <img src="@/assets/img/news_1.png" alt="News Image" class="news-image" />
                <a href="https://www.chinacourt.org/article/detail/2021/04/id/5967868.shtml" target="_blank" class="image-link"><span>坚持以人民为中心 坚决打击电信网络诈骗</span></a>
              </div>
              <div v-else-if="currentImageIndex == 1" class="image-container">
                <img src="@/assets/img/news_2.jpg" alt="News Image" class="news-image" />
                <a href="https://www.msn.cn/zh-cn/money/%E9%80%9A%E7%94%A8/%E8%AD%A6%E6%96%B9%E7%B4%A7%E6%80%A5%E6%8F%90%E9%86%92-%E4%B8%AD%E9%93%B6%E4%BC%9A%E8%AE%AE-%E9%93%B6%E8%81%94%E4%BC%9A%E8%AE%AE-%E9%83%BD%E6%98%AF%E8%AF%88%E9%AA%97%E8%BD%AF%E4%BB%B6/ar-AA1A2u0E?ocid=BingNewsSerp" target="_blank" class="image-link"><span>警方紧急提醒：“中银会议”“银联会议”都是诈骗软件</span></a>
              </div>
              <div v-else-if="currentImageIndex == 2" class="image-container">
                <img src="@/assets/img/news_3.jpg" alt="News Image" class="news-image" />
                <a href="https://www.163.com/dy/article/JPJLEH1H0525KHJ9.html?f=post2020_dy_recommends" target="_blank" class="image-link"><span>警惕披着赌博外衣的网络诈骗，仁怀已有人中招</span></a>
              </div>
            </div>
            <div class="right-news">
              <span>诈骗高敏感词</span>
              <ul class="sensitive-word-list">
                <li>1. (权威机构)推荐</li>
                <li>2. 老字号</li>
                <li>3. 提高智商</li>
                <li>4. 国际品质</li>
                <li>5. 延年益寿</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import NERTC from 'nertc-web-sdk';
import * as echarts from 'echarts';

export default {
  name: 'VideoChatJoin',
  data() {
    return {
      channelName: '',
      isSupport: true,
      isVideo: true,
      videoColor: '#414656',
      videoHoverColor: '#353a4a',
      recentChannels: [],
      faceForgeryDetection: false,
      audioForgeryDetection: false,
      fraudScriptDetection: false,
      facialEmotionRecognition: false,
      voiceEmotionRecognition: false,
      actionEmotionRecognition: false,
      textEmotionRecognition: false,
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
      currentImageIndex: 0,
    };
  },
  watch: {
    isVideo(newVal) {
      if (newVal) {
        this.videoColor = '#414656';
        this.videoHoverColor = '#6c748f';
      } else {
        this.videoColor = '#c43d60';
        this.videoHoverColor = '#e6557c';
      }
      this.initChart(); // 重新初始化图表以应用新的颜色
    },
  },
  mounted() {
    if (!NERTC.checkSystemRequirements()) {
      this.isSupport = false;
    }
    this.initChart();

    // 添加定时器来切换图片
    setInterval(() => {
      this.currentImageIndex = (this.currentImageIndex + 1) % 3;
    }, 5000); // 每隔3秒切换一次图片
  },
  methods: {
    handleSubmit() {
        const { channelName } = this;

        if (!channelName) {
          this.$message({
            message: '请输入房间号',
            type: 'warning',
          });
          return;
        } else if (!/^[0-9]{1,12}$/.test(channelName)) {
          this.$message({
            message: '房间号为12位以内的数字',
            type: 'warning',
          });
          return;
        }

        // 将当前输入的房间号添加到最近房间号列表中
        if (!this.recentChannels.includes(channelName)) {
          this.recentChannels.unshift(channelName);
          if (this.recentChannels.length > 5) {
            this.recentChannels.pop();
          }
        }
        if(this.isVideo)
          this.$router.push(`/video-chat?channelName=${channelName}`);
        else
          this.$router.push(`/audio-chat?channelName=${channelName}`);
    },
    selectAllFeatures() {
      this.faceForgeryDetection = true;
      this.audioForgeryDetection = true;
      this.fraudScriptDetection = true;
      this.facialEmotionRecognition = true;
      this.voiceEmotionRecognition = true;
      this.actionEmotionRecognition = true;
      this.textEmotionRecognition = true;
    },
    hexToRgba(hex, alpha) {
      // Remove the hash at the start if it's there
      hex = hex.replace(/^#/, '');

      // Parse the r, g, b values
      let r = parseInt(hex.substring(0, 2), 16);
      let g = parseInt(hex.substring(2, 4), 16);
      let b = parseInt(hex.substring(4, 6), 16);

      // Return the rgba string
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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
              color: this.videoColor,
              width: 2,
            },
            itemStyle: {
              color: this.videoColor,
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
                    color: this.hexToRgba(this.videoColor, 0.3),
                  },
                  {
                    offset: 1,
                    color: this.hexToRgba(this.videoColor, 0),
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
};
</script>

<style scoped>
.join {
  height: 100%;
  display: flex;
  justify-content: center;
  flex-direction: column;
}

.content-container {
  width: 80%; 
  height: 80%;
  padding: 40px 20px;
  margin-left: 4%;
  display: flex;
  justify-content: flex;
}

.left-content {
  width: 57%;
  text-align: center;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.2);
  margin-right: 60px;
  border-radius: 8px;
}

.right-content {
  width: 45%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.header {
  margin-bottom: 20px;
}

.header h2 {
  margin: 0;
  font-size: 24px;
  font-weight: bold;
  color: #333;
}

.logo {
  height: 50px;
  margin-bottom: 40px;
}

.input {
  height: 50px;
  margin-top: 2vw;
  margin-bottom: 2vw;
  width: 60%;
  font-size:18px;
}

.add {
  text-align: center;
}

.submit-btn {
  width: 60%;
  height: max(3vw, 50px);
  border-radius: 10px;
  margin-top: 1vw;
  background-color: v-bind('videoColor');
  border-color: v-bind('videoColor');
  font-size: 20px;
  font-weight: bolder;
  letter-spacing:1px;
}

.submit-btn:hover {
  background-color: v-bind('videoHoverColor');
  border-color: v-bind('videoHoverColor');
}

.errorMsg {
  font-size: 14px;
  color: red;
}

.notes {
  width: 105%;
  background: #fff;
  box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.2);
  border-radius: 8px;
  height: 47%;
}

.notes h3 {
  margin-bottom: 10px;
  font-size: 18px;
  font-weight: bold;
  text-align: center;
}

.notes ul {
  list-style: none;
  padding: 0;
}

.notes li {
  margin-bottom: 10px;
  font-size: 14px;
  color: #333;
}

.title-bar {
  background-color: v-bind('videoColor'); 
  color: #fff;
  height: 55px;
  font-size: 25px;
  letter-spacing: 1.5px;
  font-weight: bold;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  margin-bottom: 10px;
  display: flex; /* 使用 flex 布局 */
  align-items: center; /* 竖直居中 */
  padding-left: 1vw; /* 横向距离 */
  padding-right: 0.5vw; /* 横向距离 */
}

/* 隐藏原来的 h3 标题 */
.notes h3 {
  display: none;
}

.switch-container {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  letter-spacing: 0;
  width: 40%;
}

.switch-container span {
  font-size: 15px;
  letter-spacing: 1px;
  color: #ffffff;
  margin-right: max(2px,0.5vw);
  white-space: nowrap; /* 确保文本不换行 */
}

/* 添加媒体查询 */
@media screen and (max-width: 1000px) {
  .switch-container span {
    display: none; /* 隐藏文字 */
  }
}

.title-container {
  display: flex; /* 使用 flex 布局 */
  width: 60%;
  font-size: min(25px, 3vw);
}

.select-channel {
  width: 30%;
  margin-left: 10px;
}

.select-channel ::v-deep .el-select__wrapper {
  height: 50px;
}

/* 新增的样式 */
.divider {
  margin-top:2vw;
  margin-bottom:2vw;
  border-top: 1px solid #ccc;
}

.feature-selection {
  text-align: left;
  margin-top: 20px;
}

.feature-selection h3 {
  letter-spacing: 2px;
  font-size: 30px;
  margin-top:2vw;
  margin-bottom:2vw;
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
  width:30%;
  font-size:25px;
  letter-spacing: 1px;
  font-weight: bold;
  /* margin-bottom: 5px; */
  /* margin-bottom: 2px; */
}

.item-container{
  width: 70%;
}

.feature-item .el-checkbox {
  margin-left: 10px;
  margin-right: 40px;
  margin-top:5px;
  margin-bottom:5px;
  width:35%;
}

.feature-item .el-checkbox ::v-deep .el-checkbox__label{
  font-size:17px;
}

.feature-item .el-checkbox ::v-deep .el-checkbox__inner{
  height: 20px;
  width: 20px;
}

.feature-item .el-checkbox ::v-deep .el-checkbox__inner:after{
  height:13px;
  width: 7px;
  left: 6px;
}

.advice{
  text-align: center;
}

.note1-text{
  font-size: min(2vw, 20px);
  padding-left: 15px;
  letter-spacing: 3px;
  line-height: 1.5;
  color:#333;
}

.chart {
  width: 100%;
  height: 220px;
  margin-top: 10px;
}

.news-container {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  width: 100%;
  height: 80%;
}

.left-news{
  width:65%;
  height:90%;
}

.right-news{
  width:30%;
  height:90%;
}

.right-news span{
  font-size: 20px;
  letter-spacing: 2px;
  display: block; /* 确保标题独占一行 */
  margin-bottom: 10px; /* 添加底部间距 */
}

.sensitive-word-list {
  list-style-type: decimal; /* 使用数字编号 */
  padding-left: 0px; /* 缩进列表 */
  margin: 0; /* 移除默认的外边距 */
}

.sensitive-word-list li {
  font-size: 18px; /* 设置列表项的字体大小 */
  line-height: 2; /* 设置行高 */
  color: #333; /* 设置文本颜色 */
}

.image-container {
  position: relative;
  width: 100%;
  height: 100%;
  animation: fadeInAnimation ease 1s; 
  animation-iteration-count: 1; 
  animation-fill-mode: forwards; 
}

.news-image {
  width: 100%;
  height: 100%;
  object-fit: fill;
}

.image-link {
  position: absolute;
  height:10%;
  width:100%;
  bottom: 0px;
  left: 0px;
  color: #fff;
  background-color: rgba(0, 0, 0, 0.5);
  text-decoration: none;
  font-size: 15px;
  display: flex;
  align-items: center;
  /* justify-content: center; */
}

.image-link span{
  padding-left: 10px;
}

.image-link:hover{
  color: #62a8ef;
}

@keyframes fadeInAnimation { 
    0% { 
        opacity: 0; 
    } 
    100% { 
        opacity: 1; 
    } 
}

</style>