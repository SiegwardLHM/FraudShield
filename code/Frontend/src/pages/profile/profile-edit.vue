<template>
  <div class="page mine-edit">
    <div class="header">
      <span @click="goBack" class="back-button">&lt;</span>
      <h2 class="title">编辑资料</h2>
    </div>
    <div class="content">
      <el-form :model="userInfo" ref="form" label-width="100px">
        <el-form-item label="头像">
          <div class="avatar-uploader" @click="triggerFileInput">
            <input type="file" ref="fileInput" accept="image/*" @change="onFileChange" style="display: none;">
            <img v-if="userInfo.headImage" :src="userInfo.headImage" class="avatar">
            <i v-else class="el-icon-plus avatar-uploader-icon"></i>
          </div>
        </el-form-item>
        <el-form-item label="用户名">
          <el-input v-model="userInfo.username" disabled></el-input>
        </el-form-item>
        <el-form-item label="昵称">
          <el-input v-model="userInfo.nickName"></el-input>
        </el-form-item>
        <el-form-item label="性别">
          <el-radio-group v-model="userInfo.sex">
            <el-radio :value="0">男</el-radio>
            <el-radio :value="1">女</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="签名">
          <el-input type="textarea" v-model="userInfo.signature" placeholder="编辑个性标签,展示我的独特态度"></el-input>
        </el-form-item>
        <div class="btn">
          <el-form-item class="button-group">
            <el-button type="primary" @click="onSubmit" class="submit-button">提交</el-button>
          </el-form-item>
        </div>
      </el-form>
    </div>
  </div>
</template>

<script>
import axios from '../../axios'; // 使用自定义的 axios 实例

export default {
  data() {
    return {
      userInfo: {
        username: '',
        nickName: '',
        headImage: '',
        sex: 0,
        signature: ''
      },
      uploadHeaders: {
        Authorization: `Bearer ${localStorage.getItem('access')}`
      }
    };
  },
  methods: {
    goBack() {
      this.$router.go(-1);
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
        const response = await axios.post('/auth/upload/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            ...this.uploadHeaders
          }
        });
        console.log('Upload response:', response); // 添加日志来查看响应内容
        this.userInfo.headImage = response.data.originUrl || response.data.data.originUrl || response.data.fileUrl || 'default/path/to/image'; // 设置默认值
        console.log(this.userInfo.headImage);
      } catch (error) {
        console.error('Error uploading file:', error);
        this.$message.error('上传失败，请重试');
      }
    },
    async onSubmit() {
      try {
        await axios.put('/auth/me/', this.userInfo);
        localStorage.setItem('userInfo', JSON.stringify(this.userInfo));
        this.$message.success('修改成功');
        this.$router.go(-1);
      } catch (error) {
        console.error('Error updating user info:', error);
        let errorMessage = '更新失败，请重试';
        if (error.response && error.response.data) {
          errorMessage = error.response.data.detail || '更新失败，请重试';
        }
        this.$message.error(errorMessage);
      }
    }
  },
  created() {
    try {
      const storedUserInfo = JSON.parse(localStorage.getItem('userInfo'));
      if (storedUserInfo && typeof storedUserInfo === 'object') {
        this.userInfo = storedUserInfo;
      }
    } catch (e) {
      console.error('Error parsing user info from localStorage:', e);
      // 如果localStorage中存储的不是一个有效的JSON对象，可以考虑清除它
      localStorage.removeItem('userInfo');
    }
  }
};
</script>

<style scoped>
.mine-edit {
  height: 100%;
  background: #f7f8fa;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.header {
  height: 7%;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  background-color: #fff;
  box-shadow: 0 4px 10px 0 rgba(47, 56, 111, 0.1);
  position: relative;
}

.back-button {
  font-size: 24px;
  cursor: pointer;
  /* margin-left: 5%; */
  margin-right: auto;
  padding-left: 10px; /* 适当调整左边距 */
}

.title {
  font-size: 20px;
  font-weight: bold;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.content {
  width: 90%;
  max-width: 600px;
  padding: 0 5%;
  background: #fff;
  box-shadow: 0 4px 10px 0 rgba(47, 56, 111, 0.1);
  border-radius: 8px;
  height: 100%;
}

.avatar-uploader {
  display: block;
  width: 100px;
  height: 100px;
  border: 1px dashed #d9d9d9;
  border-radius: 50%;
  cursor: pointer;
  overflow: hidden;
  position: relative;
}

.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  text-align: center;
  line-height: 100px;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.avatar {
  width: 100%;
  height: 100%;
  display: block;
  border-radius: 50%;
}

.el-form{
  margin-top: 20%;
}
.el-form-item {
  margin-bottom: 8%;
}

.el-form-item label {
  font-size: 16px;
  color: #333;
}

.btn {
  display: flex;
  justify-content: center;
  text-align: center;
}

.submit-button {
  width: 250px;
  height: 60px;
  border-radius: 20px;
  font-size: 20px;
  font-weight: bolder;
}
</style>
