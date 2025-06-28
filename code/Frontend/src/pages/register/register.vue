<template>
  <div class="container">
    <div class="card">
      <div class="content">
        <div class="image-wrapper">
          <img aria-hidden="true" class="image light" src="@/assets/img/backGround4.png" alt="Office" />
          <img aria-hidden="true" class="image dark" src="@/assets/img/backGround4.png" alt="Office" />
        </div>
        <div class="form-wrapper">
          <div class="form-container">
            <h1 class="form-title">欢迎注册</h1>
            <el-form :model="form" :rules="rules" ref="form">
              <el-form-item label="" prop="username">
                <el-input v-model="form.username" placeholder="请输入用户名" class="form-input">
                  <template #prefix>
                    <el-icon><Avatar /></el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item label="" prop="password">
                <el-input v-model="form.password" type="password" placeholder="请输入密码" class="form-input" show-password>
                  <template #prefix>
                    <el-icon><Lock /></el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item label="" prop="password2">
                <el-input v-model="form.password2" type="password" placeholder="请确认密码" class="form-input" show-password>
                  <template #prefix>
                    <el-icon><Lock /></el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <div class="checkbox-wrapper">
                <label class="checkbox-label">
                  <input type="checkbox" class="checkbox" />
                  <span>我已阅读并同意<span class="link">隐私政策</span></span>
                </label>
              </div>
              <el-form-item class="button-group">
                <el-button type="primary" plain @click="submitForm" class="button">注册</el-button>
              </el-form-item>
            </el-form>
            <hr class="divider" />
            <p class="form-footer">
              已有账号？<span @click="goToLogin" class="link">去登录</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from '../../axios';

export default {
  data() {
    return {
      form: {
        username: '',
        password: '',
        password2: ''
      },
      rules: {
        username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 8, message: '密码长度不能少于8位', trigger: 'blur' },
          { pattern: /^(?!\d+$)(?![a-zA-Z]+$)[0-9A-Za-z]{8,}$/, message: '密码必须包含数字和字母', trigger: 'blur' }
        ],
        password2: [
          { required: true, message: '确认密码不能为空', trigger: 'blur' },
          { validator: (rule, value, callback) => {
              if (value !== this.form.password) {
                callback(new Error('两次输入的密码不一致'));
              } else {
                callback();
              }
            },
            trigger: 'blur'
          }
        ]
      }
    };
  },
  methods: {
    async submitForm() {
      this.$refs.form.validate(async (valid) => {
        if (valid) {
          try {
            const response = await axios.post('/auth/register/', this.form);
            const { access, refresh } = response.data;
            localStorage.setItem('access', access);
            localStorage.setItem('refresh', refresh);
            this.$message.success('注册成功');
            this.$router.push('/login');
          } catch (error) {
            let errorMessage = '网络错误，请稍后重试';
            if (error.response && error.response.data) {
              errorMessage = Object.values(error.response.data).flat().join('\n');
            }
            this.$alert(errorMessage, '错误', {
              confirmButtonText: '关闭',
              type: 'error'
            });
          }
        }
      });
    },
    goToLogin() {
      this.$router.push('/login');
    }
  }
};
</script>

<style scoped>
.container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 92vh;
  /* padding: 1.5rem; */
  /* background-color: rgb(249, 250, 251); */
}

.card {
  width: 70%;
  height: 70vh;
  background-color: rgb(255, 255, 255);
  border-radius: 0.5rem;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2), 0 6px 20px rgba(0, 0, 0, 0.19);
  overflow: hidden;
}

.content {
  display: flex;
  width: 100%;
  height: 100%;
  overflow-y: auto;
  overflow: hidden;
}

.image-wrapper {
  flex: 1;
  max-width: 60%;
  border-right: 1px solid rgb(229, 231, 235);
}

.image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.light {
  display: block;
}

.dark {
  display: none;
}

.form-wrapper {
  flex: 1;
  max-width: 40%;
}

.form-container {
  width: 80%;
  height: 100%;
  margin: auto;
}

.form-title {
  text-align: center;
  margin-top: 20%;
  margin-bottom: 15%;
  font-size: 3rem;
  font-weight: 600;
  color: rgb(55, 65, 81);
}

.form-label {
  margin-top: 5%;
  display: block;
  margin-bottom: 1rem;
}

.form-label span {
  display: block;
  margin-bottom: 0.25rem;
  color: rgb(55, 65, 81);
}

.form-input {
  width: 100%;
  box-sizing: border-box;
  padding: 0.5rem;
  /* border: 1px solid rgb(107, 114, 128); */
  background-color: white;
  outline: none;
  border-radius: 0.25rem;
  font-size: 0.8rem;
}

.checkbox-wrapper {
  margin: 1.5rem 0;
  margin-top: 10%;
}

.checkbox-label {
  display: flex;
  align-items: center;
  color: rgb(75, 85, 99);
}

.checkbox {
  margin-right: 0.5rem;
  width: 1rem;
  height: 1rem;
  border: 1px solid rgb(107, 114, 128);
  background-color: white;
}

.link {
  color: #f84667;
  text-decoration: underline;
  cursor: pointer;
}

.button {
  border: none; /* 去除边框 */
  display: block;
  width: 100%;
  padding: 0.5rem 0;
  background-color: #f84667;
  color: white;
  text-align: center;
  border-radius: 0.5rem;
  text-transform: uppercase;
  transition: background-color 0.3s ease;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2), 0 6px 15px rgba(0, 0, 0, 0.19);
}

.button:hover {
  background-color: #ff5675;
}

.divider {
  margin: 1.5rem 0;
  border: none;
  border-top: 1px solid rgb(229, 231, 235);
}

.form-footer {
  text-align: center;
  margin-top: 1rem;
}

.form-footer .link {
  color: #f84667;
}
</style>
