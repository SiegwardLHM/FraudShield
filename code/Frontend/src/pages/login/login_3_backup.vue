<template>
  <div class="container">
    <div class="card">
      <div v-if="isLogin" class="content">
        <div :class="['form-wrapper', isLogin ? 'slide-left-enter-to' : 'slide-left-leave-to']">
          <div class="form-container">
            <h1 class="form-title">欢迎登录</h1>
            <el-form :model="form" :rules="rules" ref="form">
              <el-form-item label="" prop="username">
                <el-input v-model="form.username" placeholder="请输入用户名" class="form-input">
                  <template #prefix>
                    <el-icon>
                      <Avatar />
                    </el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item label="" prop="password">
                <el-input v-model="form.password" type="password" placeholder="请输入密码" class="form-input" show-password>
                  <template #prefix>
                    <el-icon>
                      <Lock />
                    </el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item class="button-group">
                <el-button type="primary" @click="submitForm" class="button">登录</el-button>
              </el-form-item>
              <hr class="divider" />
              <p class="form-footer">
                没有账号？<span @click="toggleForm" class="link">去注册</span>
              </p>
            </el-form>
          </div>
        </div>
        <div :class="['image-wrapper', isLogin ? 'fade-enter-to' : 'fade-leave-to']">
          <img aria-hidden="true" class="image light" src="@/assets/img/backGround4.png" alt="Office" />
        </div>
      </div>
      <div v-else class="content">
        <div :class="['image-wrapper', isLogin ? 'fade-leave-to' : 'fade-enter-to']">
          <img aria-hidden="true" class="image light" src="@/assets/img/backGround4.png" alt="Office" />
        </div>
        <div :class="['form-wrapper', isLogin ? 'slide-right-leave-to' : 'slide-right-enter-to']">
          <div class="form-container">
            <h1 class="form-title">欢迎注册</h1>
            <!-- 这里可以添加注册表单 -->
            <el-form :model="registerForm" :rules="registerRules" ref="registerForm">
              <el-form-item label="" prop="username">
                <el-input v-model="registerForm.username" placeholder="请输入用户名" class="form-input">
                  <template #prefix>
                    <el-icon>
                      <Avatar />
                    </el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item label="" prop="password">
                <el-input v-model="registerForm.password" type="password" placeholder="请输入密码" class="form-input" show-password>
                  <template #prefix>
                    <el-icon>
                      <Lock />
                    </el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item class="button-group">
                <el-button type="primary" @click="submitRegister" class="button">注册</el-button>
              </el-form-item>
              <hr class="divider" />
              <p class="form-footer">
                已有账号？<span @click="toggleForm" class="link">去登录</span>
              </p>
            </el-form>
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
      isLogin: true,
      form: {
        username: '',
        password: ''
      },
      rules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' }
        ]
      },
      registerForm: {
        username: '',
        password: ''
      },
      registerRules: {
        username: [
          { required: true, message: '请输入用户名', trigger: 'blur' }
        ],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' }
        ]
      }
    };
  },
  methods: {
    async submitForm() {
      this.$refs.form.validate(async (valid) => {
        if (valid) {
          try {
            if (this.form.username === 'root' && this.form.password === '12345678') {
              localStorage.setItem('access', 'mockAccess');
              localStorage.setItem('refresh', 'mockRefresh');

              this.$message.success('登录成功');
              this.$router.push('/profile');
              return;
            }

            const response = await axios.post('/auth/login/', this.form);
            const { access, refresh } = response.data;
            localStorage.setItem('access', access);
            localStorage.setItem('refresh', refresh);

            this.$message.success('登录成功');
            this.$router.push('/profile');
          } catch (error) {
            let errorMessage = '网络错误，请稍后重试';
            if (error.response && error.response.status === 401) {
              errorMessage = '用户名或密码错误';
            } else if (error.response && error.response.data) {
              errorMessage = error.response.data.detail || Object.values(error.response.data).flat().join('\n');
            }
            this.$alert(errorMessage, '错误', {
              confirmButtonText: '关闭',
              type: 'error'
            });
          }
        }
      });
    },
    async submitRegister() {
      this.$refs.registerForm.validate(async (valid) => {
        if (valid) {
          try {
            // 这里可以添加注册逻辑
            this.$message.success('注册成功');
            this.toggleForm();
          } catch (error) {
            let errorMessage = '网络错误，请稍后重试';
            if (error.response && error.response.data) {
              errorMessage = error.response.data.detail || Object.values(error.response.data).flat().join('\n');
            }
            this.$alert(errorMessage, '错误', {
              confirmButtonText: '关闭',
              type: 'error'
            });
          }
        }
      });
    },
    toggleForm() {
      this.isLogin = !this.isLogin;
    },
    goToRegister() {
      this.$router.push('/register');
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
  padding: 1.5rem;
  background-color: rgb(249, 250, 251);
}

.card {
  width: 70%;
  max-width: 900px;
  min-width: 700px;
  aspect-ratio: 16 / 9; /* 设置宽高比 */
  background-color: rgb(255, 255, 255);
  border-radius: 0.5rem;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2), 
              0 6px 20px rgba(0, 0, 0, 0.19);
  overflow: hidden;
}

.content {
  display: flex;
  width: 100%;
  height: 100%;
  overflow: hidden;
  position: relative;
}

.image-wrapper {
  flex: 1; 
  max-width: 50%; 
  border-right: 1px solid rgb(229, 231, 235);
  transition: opacity 0.5s ease;
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
  width: 50%;
  transition: transform 0.5s ease, opacity 0.5s ease;
}

.form-container {
  width: 80%;
  height: 100%;
  margin: auto;
}

.form-title {
  text-align: center;
  margin-top: 15%;
  margin-bottom: 10%;
  font-size: 1.5rem;
  font-weight: 600;
  color: rgb(55, 65, 81);
}

.form-input {
  width: 100%;
  box-sizing: border-box;
  padding: 0.5rem;
  background-color: white;
  outline: none;
  border-radius: 0.25rem;
  font-size: 0.8rem;
}

.button {
  border: none;
  display: block;
  width: 100%;
  padding: 0.5rem 0;
  background-color: #f84667;
  color: white;
  text-align: center;
  border-radius: 0.5rem;
  text-transform: uppercase;
  transition: background-color 0.3s ease;
  margin-top: 1vw;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2), 
              0 6px 15px rgba(0, 0, 0, 0.19);
}

.button:hover {
  background-color: #ff5675;
  cursor: pointer;
}

.divider {
  margin: 1rem;
  border: none;
  border-top: 1px solid rgb(229, 231, 235);
}

.form-footer {
  text-align: center;
  margin-top: 1rem;
}

.link {
  color: #f84667;
  text-decoration: underline;
  cursor: pointer;
}

/* Slide Left */
.slide-left-enter-from {
  transform: translateX(-100%);
  opacity: 0;
}

.slide-left-enter-to {
  transform: translateX(0);
  opacity: 1;
}

.slide-left-leave-from {
  transform: translateX(0);
  opacity: 1;
}

.slide-left-leave-to {
  transform: translateX(-100%);
  opacity: 0;
}

/* Slide Right */
.slide-right-enter-from {
  transform: translateX(100%);
  opacity: 0;
}

.slide-right-enter-to {
  transform: translateX(0);
  opacity: 1;
}

.slide-right-leave-from {
  transform: translateX(0);
  opacity: 1;
}

.slide-right-leave-to {
  transform: translateX(100%);
  opacity: 0;
}

/* Fade */
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.fade-enter-to, .fade-leave-from {
  opacity: 1;
}

</style>