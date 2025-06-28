<template>
  <div class="page mine-password">
    <div class="header">
      <span @click="goBack" class="back-button">&lt;</span>
      <h2 class="title">修改密码</h2>
    </div>
    <div class="content">
      <el-form :model="formData" :rules="rules" ref="form">
        <el-form-item label="原密码" prop="old_password">
          <el-input type="password" v-model="formData.old_password" placeholder="请输入原密码" show-password></el-input>
        </el-form-item>
        <el-form-item label="新密码" prop="new_password">
          <el-input type="password" v-model="formData.new_password" placeholder="请输入新密码" show-password></el-input>
        </el-form-item>
        <el-form-item label="确认密码" prop="confirm_password">
          <el-input type="password" v-model="formData.confirm_password" placeholder="请确认新密码" show-password></el-input>
        </el-form-item>
        <div class="btn">
          <el-form-item class="button-group">
            <el-button type="primary" @click="submit" class="submit-button">提交</el-button>
          </el-form-item>
        </div>
      </el-form>
    </div>
  </div>
</template>

<script>
import axios from '../../axios'; // 使用你的 axios 实例
import { ElMessageBox, ElMessage } from 'element-plus';

export default {
  data() {
    return {
      formData: {
        old_password: '',
        new_password: '',
        confirm_password: ''
      },
      rules: {
        old_password: [
          { required: true, message: '请输入原密码', trigger: 'blur' }
        ],
        new_password: [
          { required: true, message: '请输入新密码', trigger: 'blur' },
          { min: 8, message: '密码长度不能少于8位', trigger: 'blur' },
          { pattern: /^(?!\d+$)(?![a-zA-Z]+$)[0-9A-Za-z]{8,}$/, message: '密码必须包含数字和字母', trigger: 'blur' }
        ],
        confirm_password: [
          { required: true, message: '确认密码不能为空', trigger: 'blur' },
          { validator: (rule, value, callback) => {
            if (value !== this.formData.new_password) {
              callback(new Error('两次输入的密码不一致'));
            } else {
              callback();
            }
          }, trigger: 'blur' }
        ]
      }
    };
  },
  methods: {
    goBack() {
      this.$router.go(-1);
    },
    async submit() {
      this.$refs.form.validate(async (valid) => {
        if (valid) {
          try {
            await axios.put('/auth/change-password/', this.formData);
            ElMessage({
              message: '修改密码成功',
              type: 'success'
            });
            this.$router.go(-1);
          } catch (error) {
            let errorMessage = '网络错误，请稍后重试';
            if (error.response && error.response.data) {
              errorMessage = Object.values(error.response.data).flat().join('\n');
            }
            ElMessageBox.alert(errorMessage, '错误', {
              confirmButtonText: '关闭',
              type: 'error'
            });
          }
        }
      });
    }
  }
};
</script>

<style scoped>
.mine-password {
  height: 100%;
  background: #f7f8fa;
  display: flex;
  flex-direction: column;
  align-items: center;
  /* justify-content: center; */
  /* margin-top: 20%; */
}

.header {
  display: flex;
  display: flex;
  align-items: center;
  width: 100%;
  background-color: #fff;
  box-shadow: 0 4px 10px 0 rgba(47, 56, 111, 0.1);
  position: relative;
}

.back-button {
  font-size: 24px;
  cursor: pointer;
  margin-right: 5%;
}

.title {
  font-size: 20px;
  font-weight: bold;
}

.content {
  width: 90%;
  max-width: 600px;
  padding: 10% 5%;
  background: #fff;
  box-shadow: 0 4px 10px 0 rgba(47, 56, 111, 0.1);
  border-radius: 8px;
}

.el-form-item {
  margin-bottom: 24px;
}

.el-form-item label {
  font-size: 16px;
  color: #333;
}

.btn {
  display: flex;
  justify-content: center;
  /* text-align: center; */
}

.el-form-item {
  margin-bottom: 10%;
}

.submit-button {
  width: 300px;
  height: 60px;
  border-radius: 20px;
  font-size: 20px;
  font-weight: bolder;
}
</style>
