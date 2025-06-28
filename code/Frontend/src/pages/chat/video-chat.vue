<template>
  <div class="wrapper">
    <div class="top-bar">
      <h2 class="header-title">视频通话</h2>
    </div>
    <div class="content">
      <div class="camera-select-wrapper">
        <select v-model="selectedCameraId" @change="handleCameraChange">
          <option value="">选择摄像头</option>
          <option v-for="device in videoDevices" :key="device.deviceId" :value="device.deviceId">
            {{ device.label || `Camera ${deviceIndex(device.deviceId)}` }}
          </option>
        </select>
        <video ref="video" autoplay playsinline></video>
      </div>
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
    <ExpandablePanel :displayContent="displayContent" :isReportButtonDisabled="isReportButtonDisabled" style="text-align: center;"></ExpandablePanel>

    <ul class="tab-bar">
      <li :class="{ silence: true, isSilence }" @click="setOrRelieveSilence"></li>
      <li :class="{ recording: true, isRecording }" @click="toggleRecording"></li>
      <li class="over" @click="confirmHandleOver"></li>
      <li :class="{ stop: true, isStop }" @click="stopOrOpenVideo"></li>
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
import RecordRTC from 'recordrtc';
import VueDragResize from 'vue-drag-resize/src';
import * as echarts from 'echarts';
import ExpandablePanel from '../../components/ExpandablePanel.vue';
import axios from '../../axios';
import { onMounted } from 'vue'
export default {
  name: 'multiple',
  components: {
    VueDragResize,
    ExpandablePanel,
  },
  data() {
    return {
      panelWidth: 1200,
      panelHeight: 800,
      isSilence: false,
      isStop: false,
      isRealTime: false,
      isRecording: false,
      showReminder: false,
      analysisHistory: [],
      isAnalysisVisible: false,
      isReportButtonDisabled: false,
      client: null,
      localUid: Math.ceil(Math.random() * 1e5),
      localStream: null,
      remoteStreams: [],
      max: 20,
      recorder: null,
      recordingChunks: [],
      mediaRecorder: null,
      analysisInterval: null,
      timer: 0,
      timerInterval: null,
      analysisStart: 0,
      analysisEnd: 0,
      selectedCameraId: '',
      videoDevices: [],
      displayContent:{
            Confidence:{
              fraud:"-1",
              carrier:"-1",
              video:"-1",
              sound:"-1",
              action:"-1",
              emotion:"-1",
              soundEmotion:"-1",
              content:"-1"
            },
            videoText: "待检测",
            video:[
              "https://www.baidu.com/img/flexible/logo/pc/result.png",
              "https://www.baidu.com/img/flexible/logo/pc/result.png",
              "pic3_url"
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
            text:"text1"
          }
    };
  },
  mounted() {
    this.$alert("实时监控中", '注意', {
              confirmButtonText: '关闭',
              type: 'warning'
            });
    this.returnResponse();
    console.log("mounted");
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
    this.setupCamera();
  },
  beforeDestroy() {
    this.leaveChannel();
    clearInterval(this.timerInterval);
  },
  methods: {
    async setupCamera() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        this.videoDevices = devices.filter(device => device.kind === 'videoinput');
      } catch (error) {
        console.error('获取设备列表失败:', error);
      }
    },
    async handleCameraChange() {
      if (this.selectedCameraId) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: this.selectedCameraId } }
          });
          this.$refs.video.srcObject = stream;
          this.localStream = stream;  // Use the selected camera stream
        } catch (error) {
          console.error('获取媒体流失败:', error);
        }
      }
    },
    deviceIndex(deviceId) {
      return this.videoDevices.findIndex(device => device.deviceId === deviceId) + 1;
    },
    resize(newRect) {
      this.width = newRect.width;
      this.height = newRect.height;
    },
    toggleAnalysisVisibility() {
      this.isAnalysisVisible = !this.isAnalysisVisible;
      console.log(this.isAnalysisVisible);
    },
    async returnResponse() {
      console.log('returnResponse');
      const response = await axios.post('/robot/analyze/', 1, {
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
      this.$message.warning("疑似诈骗...");
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
        const div = [...this.$refs.remote].find((item) => {
          return Number(item.dataset.uid) === Number(remoteStream.getId());
        });
        const playOptions = {
          audio: true,
          video: true,
          screen: true
        };
        remoteStream.play(div, playOptions).then(() => {
          console.log('播放对端的流成功: ', playOptions);
          remoteStream.setRemoteRenderMode({
            width: div.clientWidth,
            height: div.clientHeight,
            cut: false,
          });
        }).catch((err) => {
          console.warn('播放对方视频失败了: ', err);
        });

        remoteStream.on('notAllowedError', (err) => {
          const errorCode = err.getCode();
          const id = remoteStream.getId();
          console.log('remoteStream notAllowedError: ', id);
          if (errorCode === 41030) {
            const userGestureUI = document.createElement('div');
            if (userGestureUI && userGestureUI.style) {
              userGestureUI.style.fontSize = '20px';
              userGestureUI.style.position = 'fixed';
              userGestureUI.style.background = 'yellow';
              userGestureUI.style.margin = 'auto';
              userGestureUI.style.width = '100%';
              userGestureUI.style.zIndex = '9999';
              userGestureUI.style.top = '0';
              userGestureUI.onclick = () => {
                if (userGestureUI && userGestureUI.parentNode) {
                  userGestureUI.parentNode.removeChild(userGestureUI);
                }
                remoteStream.resume();
              };
              userGestureUI.style.display = 'block';
              userGestureUI.innerHTML = '自动播放受到浏览器限制，需手势触发。<br/>点击此处手动播放';
              document.body.appendChild(userGestureUI);
            }
          }
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
        this.$router.push('/video');
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
          console.info('加入房间成功，开始初始化本地音视频流');
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
        video: true,
        screen: false,
      });

      this.localStream.setVideoProfile({
        resolution: NERTC.VIDEO_QUALITY_720p,
        frameRate: NERTC.CHAT_VIDEO_FRAME_RATE_15,
      });
      this.localStream.setAudioProfile('speech_low_quality');
      this.localStream
        .init()
        .then(() => {
          console.warn('音视频开启完成，可以播放了');
          this.$nextTick(() => {
            const div = this.$refs.local;
            if (div) {
              this.localStream.play(div);
              this.localStream.setLocalRenderMode({
                width: div.clientWidth,
                height: div.clientHeight,
                cut: false,
              });
            } else {
              console.error('无法找到DOM元素：local');
            }
          });
          this.publish();
        })
        .catch((err) => {
          console.warn('音视频初始化失败: ', err);
          this.$message.error('音视频初始化失败');
          this.localStream = null;
        });
    },
    publish() {
      console.warn('开始发布视频流');
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
        video: true,
        screen: true
      });
      this.client
        .subscribe(remoteStream)
        .then(() => {
          console.warn('本地 subscribe 成功');
        })
        .catch((err) => {
          console.warn('本地 subscribe 失败: ', err);
          this.$message.error('订阅对方的流失败');
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
    stopOrOpenVideo() {
      const { isStop } = this;
      this.isStop = !isStop;
      if (this.isStop) {
        console.warn('关闭摄像头');
        this.localStream
          .close({
            type: 'video',
          })
          .then(() => {
            console.warn('关闭摄像头 success');
          })
          .catch((err) => {
            console.warn('关闭摄像头失败: ', err);
            this.$message.error('关闭摄像头失败');
          });
      } else {
        console.warn('打开摄像头');
        if (!this.localStream) {
          this.$message.error('当前不能打开camera');
          return;
        }
        this.localStream
          .open({
            type: 'video',
          })
          .then(() => {
            console.warn('打开摄像头 success');
            this.$nextTick(() => {
              const div = this.$refs.local;
              if (div) {
                this.localStream.play(div);
                this.localStream.setLocalRenderMode({
                  width: div.clientWidth,
                  height: div.clientHeight,
                  cut: false,
                });
              } else {
                console.error('无法找到DOM元素：local');
              }
            });
          })
          .catch((err) => {
            console.warn('打开摄像头失败: ', err);
            this.$message.error('打开摄像头失败');
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
          type: 'video',
        });
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
      if (this.isRealTime && this.mediaRecorder) {
        this.isRealTime = false;
        clearInterval(this.analysisInterval);
        this.mediaRecorder.stop();
        clearInterval(this.timerInterval);
      }
    },
    stopRecording() {
      if (this.isRecording && this.recorder) {
        this.isRecording = false;
        this.recorder.stopRecording(() => {
          this.downloadRecording(this.recorder.getBlob(), 'screen-recording.mp4');
          this.recorder.destroy();
          this.recorder = null;
        });
      }
    },
    async confirmToggleRealTimeAnalysis() {
      if (!this.isRealTime) {
        this.$confirm('开始检测时会同时开启录屏，是否继续？', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning',
        }).then(() => {
          this.toggleRealTimeAnalysis();
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
        await this.startRecording();
      }
    },
    async startRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        alert('当前设备不支持录屏');
        return;
      }

      this.analysisStart = this.timer;  // 记录当前计时器的值作为分析开始时间
      this.analysisEnd = 0;
      this.timer = 0;  // 重置计时器
      this.timerInterval = setInterval(() => {
        this.timer++;
      }, 1000);

      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true
        });

        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const combinedStream = new MediaStream([
          ...stream.getVideoTracks(),
          ...audioStream.getAudioTracks()
        ]);

        this.mediaRecorder = new MediaRecorder(combinedStream, {
          mimeType: 'video/webm;codecs=vp8,opus'
        });

        this.mediaRecorder.ondataavailable = async (event) => {
          if (event.data.size > 0) {
            const formData = new FormData();
            formData.append('file', event.data, `video-recording-${Date.now()}.webm`);

            const token = localStorage.getItem('access');
            if (!token) {
              console.error('No access token found');
              return;
            }

            try {
              const response = await fetch('http://localhost:8000/api/robot/real-time-analysis/video/', {
                method: 'POST',
                body: formData,
                headers: {
                  'Authorization': `Bearer ${token}`
                }
              });
              if (!response.ok) {
                throw new Error('Network response was not ok');
              }
              const data = await response.json();
              if (data.result && data.result.includes("疑似诈骗")) {
                this.$message.warning("疑似诈骗...");
              }
              this.analysisHistory.push({
                start: this.formatTime(this.analysisEnd),  // 分析开始时间
                end: this.formatTime(this.timer),  // 分析结束时间
                result: data.result.replace(/\n/g, '<br>'),
                screenshot: data.screenshot
              });
              this.analysisEnd = this.timer;
              console.log('更新后的analysisHistory:', this.analysisHistory);
            } catch (error) {
              console.error('实时分析数据发送失败', error);
            }
          }
        };

        this.mediaRecorder.start(20000); // 每10秒触发一次ondataavailable事件

        // // 每10秒请求一次数据
        // this.analysisInterval = setInterval(() => {
        //   if (this.mediaRecorder.state === 'recording') {
        //     this.mediaRecorder.requestData();
        //   }
        // }, 10000);

      } catch (error) {
        console.error('获取媒体流失败:', error);
      }
    },
    async toggleRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        alert('当前设备不支持录屏');
        return;
      }

      if (this.isRecording) {
        this.stopRecording();
      } else {
        await this.startRecording();
      }
    },
    dataURLToBlob(dataURL) {
      const byteString = atob(dataURL.split(',')[1]);
      const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      return new Blob([ab], { type: mimeString });
    },
    downloadRecording(blob, filename) {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
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

      const imgLink = document.createElement('a');
      imgLink.href = analysis.screenshot;
      imgLink.download = 'screenshot.png';
      document.body.appendChild(imgLink);
      imgLink.click();
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
  },
};
</script>

<style scoped lang="less">
.wrapper {
  height: 100%;
  background-image: linear-gradient(179deg, #141417 0%, #181824 100%);
  display: flex;
  flex-direction: column;
  .overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

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
    flex-direction: column;

    .camera-select-wrapper {
      width: 90%;
      height: 50px;
      display: flex;
      flex-direction: column;
      align-items: center;

      select {
        margin-bottom: 20px;
        padding: 5px;
      }

      video {
        width: 100%;
        height: 49%;
        border: 1px solid #ccc;
      }
    }

    .window-wrapper {
      display: flex;
      flex-wrap: wrap;
      gap: 2%;
      width: 90%;
      height: 90%;
      overflow-y: auto;
    }

    .window {
      position: relative;
      background: #25252d;
      border: 1px solid #ffffff;
      width: 100%;
      height: 50%;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;

      .user-info {
        position: absolute;
        bottom: 2%;
        right: 2%;
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

      &.stop {
        background: url("@/assets/img/icon/stop.png") no-repeat center;
        background-size: 60px 54px;

        &:hover {
          background: url("@/assets/img/icon/stop-hover.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &:active {
          background: url("@/assets/img/icon/stop-click.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &.isStop {
          background: url("@/assets/img/icon/open.png") no-repeat center;
          background-size: 60px 54px;

          &:hover {
            background: url("@/assets/img/icon/open-hover.png") no-repeat
              center;
            background-size: 60px 54px;
          }

          &:active {
            background: url("@/assets/img/icon/open-click.png") no-repeat
              center;
            background-size: 60px 54px;
          }
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
        background: url("@/assets/img/icon/recording-video.png") no-repeat center;
        background-size: 60px 54px;

        &:hover {
          background: url("@/assets/img/icon/recording-video.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &:active {
          background: url("@/assets/img/icon/recording-video.png") no-repeat
            center;
          background-size: 60px 54px;
        }

        &.isRecording {
          background: url("@/assets/img/icon/recording-video-on.png") no-repeat center;
          background-size: 60px 54px;

          &:hover {
            background: url("@/assets/img/icon/recording-video-on.png") no-repeat
              center;
            background-size: 60px 54px;
          }

          &:active {
            background: url("@/assets/img/icon/recording-video-on.png") no-repeat
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

    img {
      max-width: 100%;
      height: auto;
    }

    .actions {
      margin: 10px 0;

      button {
        margin-right: 10px;
        margin-bottom: 10px;
      }
    }
  }

  .panel {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: relative;
    overflow: hidden;
  }

  .panel-content {
      height: 100%;
      overflow-y: auto;
      padding-right: 10px;
  }

  .close-button {
      position: absolute;
      top: 10px;
      right: 10px;
      cursor: pointer;
      font-size: 20px;
      color: #333;
  }

  .fraud_analysis {
      height: 200%;
  }

  .inner_title {
      font-size: 5vh;
      font-weight: bold;
      height: 5%;
      text-align: center;
      border-bottom: 1px solid gray;
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: center;
  }

  .inner_doc1 {
      font-size: 3vh;
      font-weight: bold;
      height: 3%;
      text-align: center;
      border-bottom: 1px solid gray;
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: center;
  }

  .inner_doc3 {
      border-bottom: 1px solid gray;
  }

  .inner_doc2 p {
      font-size: 2vh;
      font-weight: bold;
      margin: 0;
  }
  .inner_doc3 p {
      font-size: 2vh;
      font-weight: bold;
      margin: 0;
  }

  .inner_picture {
      width: 100%;
      height: auto;
  }

  .inner_picture img {
      width: 30%;
      height: auto;
      margin-left: 2.5%;
  }
}
</style>
