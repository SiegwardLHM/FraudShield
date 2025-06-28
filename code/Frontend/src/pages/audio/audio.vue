<template>
  <div class="join">
    <div class="content">
      <div class="content-top">
        <div class="header">
          <h2>语音通话</h2>
        </div>
        <img src="@/static/logo.png" class="logo" alt="logo" />
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
      </div>
      <div class="notes">
        <h3>注意事项</h3>
        <ul>
          <li>1. 房间号为12位以内的数字。</li>
          <li>2. 请确保您的麦克风和扬声器正常工作。</li>
          <li>3. 建议在安静的环境下进行语音通话。</li>
          <li>4. 离开房间前请注意下载分析报告！</li>
          <li>5. 为保护用户隐私，系统挂断后会自动清除所有记录！</li>
        </ul>
      </div>
      <div class="add">
        <el-button
          :disabled="!isSupport"
          class="submit-btn"
          type="primary"
          @click="handleSubmit"
          >加入房间</el-button
        >
      </div>
      <div class="errorMsg" v-show="!isSupport">
        当前浏览器不支持体验，建议下载安装最新chrome浏览器
      </div>
    </div>
  </div>
</template>

<script>
import NERTC from 'nertc-web-sdk';

export default {
  name: 'AudioChatJoin',
  data() {
    return {
      channelName: '',
      isSupport: true,
    };
  },
  mounted() {
    if (!NERTC.checkSystemRequirements()) {
      this.isSupport = false;
    }
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

      this.$router.push(`/audio-chat?channelName=${channelName}`);
    },
  },
};
</script>

<style scoped>
.join {
  height: 100%;
  /* background: #f7f8fa; */
  display: flex;
  /* align-items: center; */
  justify-content: center;
  flex-direction: column;
}
.content {
  width: 26%; 
  height: 70vh;
  padding: 40px 20px;
  margin-left: 30%;
  background: #fff;
  box-shadow: 0 4px 10px 0 rgba(47, 56, 111, 0.1);
  border-radius: 8px;
}
.content-top {
  text-align: center;
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
  width: 100%;
  height: 50px;
  margin-bottom: 10px;
}
.add {
  text-align: center;
}
.submit-btn {
  width: 80%;
  height: 60px;
  border-radius: 20px;
  font-size: 20px;
  font-weight: bolder;
}
.errorMsg {
  font-size: 14px;
  color: red;
}
.notes {
  width: 90%;
  max-width: 400px;
  padding: 20px;
  background: #fff;
  box-shadow: 0 4px 10px 0 rgba(47, 56, 111, 0.1);
  border-radius: 8px;
  margin: 30px 0;
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
</style>
