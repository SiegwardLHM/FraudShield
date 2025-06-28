<template>
  <div class="wrapper">
    <div class="top-bar">
      <h2 class="header-title">语音通话</h2>
    </div>
    <div class="content">
      <div class="window-wrapper">
        <div v-if="localStream" class="window" ref="local">
          <div class="user-info">本地用户</div>
        </div>
        <template v-if="remoteStreams.length">
          <div
            v-for="item in remoteStreams"
            :key="item.getId()"
            class="window"
            ref="remote"
            :data-uid="item.getId()"
          >
            <div class="user-info">用户ID: {{ item.getId() }}</div>
          </div>
        </template>
        <div v-else class="window" ref="remote">
          <span class="loading-text">等待对方加入...</span>
        </div>
      </div>
    </div>
    <div class="toggle-analysis" @click="toggleAnalysisVisibility">
      <el-button type="primary">{{ isAnalysisVisible ? '隐藏分析结果' : '显示分析结果' }}</el-button>
    </div>
    <div v-if="isAnalysisVisible">
      <!-- <VueDragResize
        :isActive="true"
        :isResizable="true"
        :w="400"
        :h="300"
        @resizing="resize"
        class="analysis-result"
      >
        <div v-for="(analysis, index) in analysisHistory" :key="index">
          <p>时间区间: {{ analysis.start }} - {{ analysis.end }}</p>
          <p v-html="analysis.result"></p>
          <div class="actions">
            <el-button @click="downloadAnalysis(analysis)">下载分析报告</el-button>
            <el-button @click="copyAnalysis(analysis)">复制分析结果</el-button>
          </div>
        </div>
      </VueDragResize> -->
    </div>
    <ul class="tab-bar">
      <li :class="{ silence: true, isSilence }" @click="setOrRelieveSilence"></li>
      <li :class="{ recording: true, isRecording }" @click="toggleRecording"></li>
      <li class="over" @click="confirmHandleOver"></li>
      <li :class="{ realtime: true, isRealTime }" @click="confirmToggleRealTimeAnalysis"></li>
      <li class="timer">{{ formatTime(timer) }}</li>
    </ul>
    <el-message v-if="showReminder" type="warning" show-close>
      注意下载分析结果
    </el-message>
  </div>
</template>

<script>
import NERTC from 'nertc-web-sdk';
import config from '../../config';
import { getToken } from '../../common';
import WavEncoder from 'wav-encoder';
import VueDragResize from 'vue-drag-resize/src';
import lamejs from 'lamejs';
import * as echarts from 'echarts';
import ExpandablePanel from '../../components/ExpandablePanel.vue';

export default {
  name: 'multiple',
  components: {
    VueDragResize,
    ExpandablePanel,
  },
  data() {
    return {
      isSilence: false,
      isRealTime: false,
      isRecording: false,
      showReminder: false,
      analysisHistory: [],
      isAnalysisVisible: false,
      client: null,
      localUid: Math.ceil(Math.random() * 1e5),
      localStream: null,
      remoteStreams: [],
      max: 20,
      mediaRecorder: null,
      analysisInterval: null,
      timer: 0,
      timerInterval: null,
      analysisStart: 0,
      analysisEnd: 0,
      audioContext: null,
      mediaStream: null,
      mediaRecorder: null,
      audioChunks: [],
      realTimeMediaRecorder: null,
      recordingMediaRecorder: null,
    };
  },
  mounted() {
    this.client = NERTC.createClient({
      appkey: config.appkey,
      debug: true,
    });
    this.registerEvents();

    this.getToken().then(token => {
      this.joinChannel(token);
    }).catch(e => {
      this.$message.error(e);
      console.error(e);
    });
  },
  beforeDestroy() {
    this.leaveChannel();
    clearInterval(this.timerInterval);
  },
  methods: {
    resize(newRect) {
      this.width = newRect.width;
      this.height = newRect.height;
    },
    toggleAnalysisVisibility() {
      this.isAnalysisVisible = !this.isAnalysisVisible;
      // if (this.isAnalysisVisible) {
      //   this.$nextTick(() => {
      //     this.displayAnalysisChart();
      //   });
      // }
    },
    registerEvents() {
      this.client.on('peer-online', (evt) => {
        console.warn(`${evt.uid} 加入房间`);
      });

      this.client.on('peer-leave', (evt) => {
        console.warn(`${evt.uid} 离开房间`);
        this.remoteStreams = this.remoteStreams.filter(
          (item) => !!item.getId() && item.getId() !== evt.uid
        );
      });

      this.client.on('stream-added', async (evt) => {
        const stream = evt.stream;
        const userId = stream.getId();
        console.warn(`收到 ${userId} 的发布 ${evt.mediaType} 的通知`);
        if (this.remoteStreams.some((item) => item.getId() === userId)) {
          console.warn('收到已订阅的远端发布，需要更新', stream);
          this.remoteStreams = this.remoteStreams.map((item) =>
            item.getId() === userId ? stream : item
          );
          await this.subscribe(stream);
        } else if (this.remoteStreams.length < this.max - 1) {
          console.warn('收到新的远端发布消息', stream);
          this.remoteStreams = this.remoteStreams.concat(stream);
          await this.subscribe(stream);
        } else {
          console.warn('房间人数已满');
        }
      });

      this.client.on('stream-removed', (evt) => {
        const stream = evt.stream;
        const userId = stream.getId();
        console.warn(`收到 ${userId} 的停止发布 ${evt.mediaType} 的通知`);
        stream.stop(evt.mediaType);
        this.remoteStreams = this.remoteStreams.filter(
          (item) => item.getId() !== userId
        );
      });

      this.client.on('stream-subscribed', (evt) => {
        const userId = evt.stream.getId();
        console.warn(`收到订阅 ${userId} 的 ${evt.mediaType} 成功的通知`);
        const remoteStream = evt.stream;
        remoteStream.play().then(() => {
          console.log('播放对端的音频流成功');
        }).catch((err) => {
          console.warn('播放对方音频失败了: ', err);
        });
      });

      this.client.on('uid-duplicate', () => {
        console.warn('==== uid重复，你被踢出');
      });

      this.client.on('error', (type) => {
        console.error('===== 发生错误事件：', type);
        if (type === 'SOCKET_ERROR') {
          console.warn('==== 网络异常，已经退出房间');
        }
      });

      this.client.on('accessDenied', (type) => {
        console.warn(`==== ${type}设备开启的权限被禁止`);
      });

      this.client.on('connection-state-change', (evt) => {
        console.warn(
          `网络状态变更: ${evt.prevState} => ${evt.curState}, 当前是否在重连：${evt.reconnect}`
        );
      });
    },
    async getToken() {
      try {
        const token = await getToken({
          uid: this.localUid,
          appkey: config.appkey,
          appSecret: config.appSecret,
          channelName: this.$route.query.channelName
        });
        return token;
      } catch (e) {
        throw e;
      }
    },
    returnJoin(time = 2000) {
      setTimeout(() => {
        this.$router.push('/audio');
      }, time);
    },
    joinChannel(token) {
      if (!this.client) {
        this.$message.error('内部错误，请重新加入房间');
        return;
      }
      console.info('开始加入房间: ', this.$route.query.channelName);
      this.client
        .join({
          channelName: this.$route.query.channelName,
          uid: this.localUid,
          token,
        })
        .then((data) => {
          console.info('加入房间成功，开始初始化本地音频流');
          this.initLocalStream();
        })
        .catch((error) => {
          console.error('加入房间失败：', error);
          this.$message.error(`${error}: 请检查appkey或者token是否正确`);
          this.returnJoin();
        });
    },
    initLocalStream() {
      this.localStream = NERTC.createStream({
        uid: this.localUid,
        audio: true,
        video: false,
        screen: false,
      });

      this.localStream.setAudioProfile('speech_low_quality');
      this.localStream
        .init()
        .then(() => {
          console.warn('音频开启完成，可以播放了');
          this.$nextTick(() => {
            this.localStream.play().then(() => {
              console.log('播放本地音频流成功');
            }).catch((err) => {
              console.warn('播放本地音频失败: ', err);
            });
          });
          this.publish();
        })
        .catch((err) => {
          console.warn('音频初始化失败: ', err);
          this.$message.error('音频初始化失败');
          this.localStream = null;
        });
    },
    publish() {
      console.warn('开始发布音频流');
      this.client
        .publish(this.localStream)
        .then(() => {
          console.warn('本地 publish 成功');
        })
        .catch((err) => {
          console.error('本地 publish 失败: ', err);
          this.$message.error('本地 publish 失败');
        });
    },
    subscribe(remoteStream) {
      remoteStream.setSubscribeConfig({
        audio: true,
        video: false,
        screen: false
      });
      this.client
        .subscribe(remoteStream)
        .then(() => {
          console.warn('本地 subscribe 成功');
        })
        .catch((err) => {
          console.warn('本地 subscribe 失败: ', err);
          this.$message.error('订阅对方的音频流失败');
        });
    },
    setOrRelieveSilence() {
      const { isSilence } = this;
      this.isSilence = !isSilence;
      if (this.isSilence) {
        console.warn('关闭mic');
        this.localStream
          .close({
            type: 'audio',
          })
          .then(() => {
            console.warn('关闭 mic success');
          })
          .catch((err) => {
            console.warn('关闭 mic 失败: ', err);
            this.$message.error('关闭 mic 失败');
          });
      } else {
        console.warn('打开mic');
        if (!this.localStream) {
          this.$message.error('当前不能打开mic');
          return;
        }
        this.localStream
          .open({
            type: 'audio',
          })
          .then(() => {
            console.warn('打开mic success');
          })
          .catch((err) => {
            console.warn('打开mic失败: ', err);
            this.$message.error('打开mic失败');
          });
      }
    },
    confirmHandleOver() {
      this.$confirm('注意下载分析结果', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }).then(() => {
        this.handleOver();
      }).catch(() => {
        this.$message({
          type: 'info',
          message: '已取消离开房间',
        });
      });
    },
    handleOver() {
      this.stopRealTimeAnalysis();
      this.stopRecording();
      this.leaveChannel();
      this.returnJoin(1);
      if (this.analysisHistory.length > 0) {
        this.showReminder = true;
        setTimeout(() => {
          this.showReminder = false;
        }, 5000);
      }
      this.isAnalysisVisible = false;
    },
    leaveChannel() {
      if (this.localStream) {
        this.localStream.close({
          type: 'audio',
        });
        this.localStream.stop();
        this.localStream = null;
      }
      if (this.client) {
        this.client.leave()
          .then(() => {
            console.warn('成功离开房间');
          })
          .catch((error) => {
            console.error('离开房间失败', error);
          });
      }
    },
    stopRealTimeAnalysis() {
      if (this.isRealTime) {
        this.isRealTime = false;
        if (this.realTimeMediaRecorder) {
          this.realTimeMediaRecorder.stop();
        }
        clearInterval(this.analysisInterval);
        clearInterval(this.timerInterval);
      }
    },
    stopRecording() {
      if (this.isRecording) {
        this.isRecording = false;
        if (this.recordingMediaRecorder) {
          this.recordingMediaRecorder.stop();
        }
      }
    },
    confirmToggleRealTimeAnalysis() {
      if (!this.isRealTime) {
        this.$confirm('开始检测时会同时开启录音，是否继续？', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning',
        }).then(() => {
          this.toggleRealTimeAnalysis();
          if (!this.isRecording) {
            this.toggleRecording();
          }
        }).catch(() => {
          this.$message({
            type: 'info',
            message: '已取消检测',
          });
        });
      } else {
        this.toggleRealTimeAnalysis();
      }
    },
    async toggleRealTimeAnalysis() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        alert('当前设备不支持实时分析');
        return;
      }

      if (this.isRealTime) {
        this.stopRealTimeAnalysis();
      } else {
        this.isRealTime = true;
        this.analysisStart = this.timer;
        this.analysisEnd = 0;
        this.timer = 0;
        this.timerInterval = setInterval(() => {
          this.timer++;
        }, 1000);

        try {
          // 创建 AudioContext
          const audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const audioDest = audioContext.createMediaStreamDestination();

          // 获取屏幕音频流
          const displayStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });

          const source = audioContext.createMediaStreamSource(displayStream);
          source.connect(audioDest);

          // 创建 MediaRecorder
          this.realTimeMediaRecorder = new MediaRecorder(audioDest.stream);
          const audioChunks = [];

          this.realTimeMediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
              audioChunks.push(event.data);
            }
          };

          this.realTimeMediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const buffer = new Uint8Array(arrayBuffer);
            const wavBlob = new Blob([buffer], { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('file', wavBlob, 'audio-recording.wav');
            formData.append('start', this.analysisStart);
            formData.append('end', this.timer);

            const token = localStorage.getItem('access');
            if (!token) {
              console.error('No access token found');
              return;
            }

            fetch('http://localhost:8000/api/robot/real-time-analysis/audio/', {
              method: 'POST',
              body: formData,
              headers: {
                'Authorization': `Bearer ${token}`
              }
            }).then(response => {
              if (!response.ok) {
                throw new Error('Network response was not ok');
              }
              return response.json();
            }).then(data => {
              if (data.result && data.result.includes("疑似诈骗")) {
                this.$message.warning("疑似诈骗...");
              }
              this.analysisHistory.push({
                start: this.formatTime(this.analysisEnd),
                end: this.formatTime(this.timer),
                result: data.result.replace(/\n/g, '<br>'), // 保留换行
              });
              this.analysisEnd = this.timer;
            }).catch(error => {
              console.error('实时分析数据发送失败', error);
            });

            if (this.isRealTime) {
              this.realTimeMediaRecorder.start();
            }
          };
          this.analysisInterval = setInterval(() => {
            this.realTimeMediaRecorder.stop();
          }, 10000);

          this.realTimeMediaRecorder.start();
        } catch (err) {
          console.error('Error accessing media devices.', err);
          alert('无法访问媒体设备，请检查权限。');
          this.isRealTime = false;
        }
      }
    },
    // displayAnalysisChart() {
    //   const chartData = {
    //     timestamps: ['0', '10', '20', '25'],
    //     values: [1, 0, 1, 0],
    //     reasons: ['原因1', '原因2', '原因3', '原因4'],
    //   };
    //   this.$nextTick(() => {
    //     this.renderChart(chartData);
    //   });
    // },
    async toggleRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        alert('当前设备不支持录音');
        return;
      }

      if (this.isRecording) {
        this.isRecording = false;
        if (this.recordingMediaRecorder) {
          this.recordingMediaRecorder.stop();
        }
      } else {
        this.isRecording = true;

        try {
          // 创建 AudioContext
          const audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const audioDest = audioContext.createMediaStreamDestination();
          
          // 获取屏幕音频流
          const mediaStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
          
          const source = audioContext.createMediaStreamSource(mediaStream);
          source.connect(audioDest);

          // 创建 MediaRecorder
          this.recordingMediaRecorder = new MediaRecorder(audioDest.stream);
          const audioChunks = [];

          this.recordingMediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
              audioChunks.push(event.data);
            }
          };

          this.recordingMediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const buffer = new Uint8Array(arrayBuffer);
            const wavBlob = new Blob([buffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(wavBlob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'audio-recording.wav';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
          };

          this.recordingMediaRecorder.start();
        } catch (err) {
          console.error('Error accessing display media.', err);
          alert('无法访问显示媒体，请检查权限。');
          this.isRecording = false;
        }
      }
    },
    downloadAnalysis(analysis) {
      const analysisBlob = new Blob(
        [
          `时间区间: ${analysis.start} - ${analysis.end}\n`,
          `分析结果: ${analysis.result.replace(/<br>/g, '\n')}\n`,
        ],
        { type: 'text/plain' }
      );

      const url = URL.createObjectURL(analysisBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'analysis_report.txt';
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
    },
    copyAnalysis(analysis) {
      const textarea = document.createElement('textarea');
      textarea.value = `时间区间: ${analysis.start} - ${analysis.end}\n分析结果: ${analysis.result.replace(/<br>/g, '\n')}\n`;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      this.$message.success('分析结果已复制到剪贴板');
    },
    formatTime(seconds) {
      const min = Math.floor(seconds / 60);
      const sec = seconds % 60;
      return `${min}:${sec < 10 ? '0' : ''}${sec}`;
    },
    renderChart(data) {
      const chartDom = this.$refs.echart;
      const myChart = echarts.init(chartDom);
      const option = {
        title: {
          text: '时序图'
        },
        tooltip: {
          trigger: 'item',
          formatter: function (params) {
            const reason = data.reasons[params.dataIndex];
            return `时间: ${params.name}<br/>是否为疑似诈骗: ${params.value === 1 ? '是' : '否'}<br/>原因: ${reason}`;
          }
        },
        xAxis: {
          type: 'category',
          data: data.timestamps
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: function (value) {
              return value === 1 ? '是' : '否';
            }
          },
          min: 0,
          max: 1,
        },
        series: [
          {
            type: 'scatter',
            data: data.values,
            itemStyle: {
              normal: {
                color: function (params) {
                  return params.value === 1 ? 'red' : 'green';
                }
              }
            }
          }
        ]
      };
      myChart.setOption(option);
    },
  }
};
</script>

<style scoped lang="less">
.wrapper {
  height: 94vh;
  background-image: linear-gradient(179deg, #141417 0%, #181824 100%);
  display: flex;
  flex-direction: column;

  .top-bar {
    height: 8%;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #292933;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  }

  .header-title {
    color: #ffffff;
    font-size: 24px;
    font-weight: bold;
  }

  .content {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;

    .window-wrapper {
      display: flex;
      flex-wrap: wrap;
      gap: 2%;
      width: 90%;
      height: 80%;
      overflow-y: auto;
    }

    .window {
      position: relative;
      background: #25252d;
      border: 1px solid #ffffff;
      width: 100%;
      height: 30%;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;

      .user-info {
        font-size: 1rem;
        color: rgb(255, 255, 255);
      }

      .loading-text {
        font-size: 1rem;
        color: #fff;
      }
    }
  }

  .toggle-analysis {
    position: absolute;
    right: 10px;
    top: 80px;
    z-index: 1001;
  }

  .tab-bar {
    height: 10%;
    background-image: linear-gradient(180deg, #292933 7%, #212129 100%);
    box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.3);
    list-style: none;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #fff;
    padding: 10px;
    gap: 10px;

    li {
      height: 100%;
      width: 20%;
      cursor: pointer;
      padding: 10px;

      &.silence {
        background: url("@/assets/img/icon/silence.png") no-repeat center;
        background-size: 60px 54px;

        &:hover {
          background: url("@/assets/img/icon/silence-hover.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &:active {
          background: url("@/assets/img/icon/silence-click.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &.isSilence {
          background: url("@/assets/img/icon/relieve-silence.png") no-repeat
            center;
          background-size: 60px 54px;

          &:hover {
            background: url("@/assets/img/icon/relieve-silence-hover.png")
              no-repeat center;
            background-size: 60px 54px;
          }

          &:active {
            background: url("@/assets/img/icon/relieve-silence-click.png")
              no-repeat center;
            background-size: 60px 54px;
          }
        }
      }

      &.over {
        background: url("@/assets/img/icon/over.png") no-repeat center;
        background-size: 68px 36px;

        &:hover {
          background: url("@/assets/img/icon/over-hover.png") no-repeat
            center;
          background-size: 68px 36px;
        }

        &:active {
          background: url("@/assets/img/icon/over-click.png") no-repeat
            center;
          background-size: 68px 36px;
        }
      }

      &.realtime {
        background: url("@/assets/img/icon/realtime.png") no-repeat center;
        background-size: 60px 54px;

        &:hover {
          background: url("@/assets/img/icon/realtime.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &:active {
          background: url("@/assets/img/icon/realtime.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &.isRealTime {
          background: url("@/assets/img/icon/realtime-on.png") no-repeat center;
          background-size: 60px 54px;

          &:hover {
            background: url("@/assets/img/icon/realtime-on.png") no-repeat
              center;
            background-size: 60px 54px;
          }

          &:active {
            background: url("@/assets/img/icon/realtime-on.png") no-repeat
              center;
            background-size: 60px 54px;
          }
        }
      }

      &.recording {
        background: url("@/assets/img/icon/recording-audio.png") no-repeat center;
        background-size: 60px 54px;

        &:hover {
          background: url("@/assets/img/icon/recording-audio.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &:active {
          background: url("@/assets/img/icon/recording-audio.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &.isRecording {
          background: url("@/assets/img/icon/recording-audio-on.png") no-repeat center;
          background-size: 60px 54px;

          &:hover {
            background: url("@/assets/img/icon/recording-audio-on.png") no-repeat
              center;
            background-size: 60px 54px;
          }

          &:active {
            background: url("@/assets/img/icon/recording-audio-on.png") no-repeat
              center;
            background-size: 60px 54px;
          }
        }
      }

      &.timer {
        font-size: 1rem;
        color: #fff;
        display: flex;
        align-items: center;
      }
    }
  }

  .record-procees {
    display: inline-block;
    height: 10px;
    width: 2%;
    margin-top: 12px;
    margin-right: 6px;
    background: red;
    border-radius: 8px;
    animation: blings 1s linear infinite;
  }

  @keyframes blings {
    0% {
      opacity: 0.1;
    }
    100% {
      opacity: 1;
    }
  }

  .analysis-result {
    max-height: 80%;
    overflow-y: auto;
    background: #333;
    color: #fff;
    padding: 20px;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5);
    z-index: 1000;

    .actions {
      margin: 10px 0;

      button {
        margin-right: 10px;
        margin-bottom: 10px;
      }
    }
  }
}
.echart-container {
  width: 100%;
  height: 300px;
  background-color: #ffffff;
  position: relative;
}
</style>
