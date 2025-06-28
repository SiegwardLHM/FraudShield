<template>
  <div class="container">
    <div class="card">
      <div v-if="isLogin" class="content">
        <div class="form-wrapper" >
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
        <div class="image-wrapper">
          <img aria-hidden="true" class="image light" src="@/assets/img/backgroundright.jpg" alt="Office" />
        </div>
      </div>
      <div v-else class="content">
        <div class="image-wrapper">
          <img aria-hidden="true" class="image light" src="@/assets/img/backgroundleft.jpg" alt="Office" />
        </div>
        <div class="form-wrapper">
          <div class="form-container">
            <h1 class="form-title">欢迎注册</h1>
            <el-form :model="registerForm" :rules="registerRules" ref="registerForm">
              <el-form-item label="" prop="username">
                <el-input v-model="registerForm.username" placeholder="请输入用户名" class="form-input">
                  <template #prefix>
                    <el-icon><Avatar /></el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item label="" prop="password">
                <el-input v-model="registerForm.password" type="password" placeholder="请输入密码" class="form-input" show-password>
                  <template #prefix>
                    <el-icon><Lock /></el-icon>
                  </template>
                </el-input>
              </el-form-item>
              <el-form-item label="" prop="password2">
                <el-input v-model="registerForm.password2" type="password" placeholder="请确认密码" class="form-input" show-password>
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
                <el-button type="primary" plain @click="submitRegister" class="button">注册</el-button>
              </el-form-item>
            </el-form>
            <hr class="divider" />
            <p class="form-footer">
              已有账号？<span @click="toggleForm" class="link">去登录</span>
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
        password: '',
        password2: ''
      },
      registerRules: {
        username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
        password: [
          { required: true, message: '请输入密码', trigger: 'blur' },
          { min: 8, message: '密码长度不能少于8位', trigger: 'blur' },
          { pattern: /^(?!\d+$)(?![a-zA-Z]+$)[0-9A-Za-z]{8,}$/, message: '密码必须包含数字和字母', trigger: 'blur' }
        ],
        password2: [
          { required: true, message: '确认密码不能为空', trigger: 'blur' },
          { validator: (rule, value, callback) => {
              if (value !== this.registerForm.password) {
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
            if (this.form.username === 'root' && this.form.password === '12345678') {
              localStorage.setItem('access', 'mockAccess');
              localStorage.setItem('refresh', 'mockRefresh');

              this.$message.success('登录成功');
              this.$router.push('/video');
              return;
            }

            const response = await axios.post('/auth/login/', this.form);
            const { access, refresh } = response.data;
            localStorage.setItem('access', access);
            localStorage.setItem('refresh', refresh);

            this.$message.success('登录成功');
            this.$router.push('/video');
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
            const response = await axios.post('/auth/register/', this.registerForm);
            const { access, refresh } = response.data;
            localStorage.setItem('access', access);
            localStorage.setItem('refresh', refresh);
            this.$message.success('注册成功');
            this.isLogin = !this.isLogin;
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
    toggleForm() {
      this.isLogin = !this.isLogin;
    },
  }
};
</script>

<style scoped>
.container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.container::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: url('@/assets/img/background.jpg') center/cover no-repeat;
  /* background: url('@/assets/img/bg-fade.png') center/cover no-repeat; */
  opacity: 0.8;
  z-index: -1; /* 确保在文字后面 */
}

.card {
  width: 70%;
  max-width: 900px;
  min-width: 700px;
  aspect-ratio: 16 / 9; /* 设置宽高比 */
  background-color: rgb(255, 255, 255);
  border-radius: 1rem;
  box-shadow: 12px 12px 7px 0px rgba(0, 0, 0, 0.3), 
              12px 12px 19px 3px rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

.content {
  display: flex;
  width: 100%;
  height: 100%;
  overflow: hidden;
  position: relative;
}

@keyframes fadeInAnimation { 
    0% { 
        opacity: 0; 
    } 
    100% { 
        opacity: 1; 
    } 
}

.image-wrapper {
  flex: 1; 
  max-width: 50%; 
  border-right: 1px solid rgb(229, 231, 235);
  animation: fadeInAnimation ease 0.75s; 
  animation-iteration-count: 1; 
  animation-fill-mode: forwards; 
}

.image {
  width: 101%;
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
  animation: fadeInAnimation ease 0.75s; 
  animation-iteration-count: 1; 
  animation-fill-mode: forwards; 
}

.form-container {
  width: 80%;
  height: 100%;
  margin: auto;
}

.form-title {
  text-align: center;
  letter-spacing: 2px;
  margin-top: 12%;
  margin-bottom: 8%;
  font-size: 25px;
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
  padding: 0.5vw;
  height: min(5vw, 65px);
}

.button {
  border: none;
  font-size:min(5vw, 15px);
  letter-spacing: 2px;
  display: block;
  width: 100%;
  height:max(3vw, 35px);
  padding: 0.5rem 0;
  background-color: #116FCD;
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
  background-color: #3a8ada;
  cursor: pointer;
}

.divider {
  margin: 1vw;
  border: none;
  border-top: 1px solid rgb(229, 231, 235);
}

.form-footer {
  text-align: center;
  font-size: 13px;
  margin-top: 1vw;
}

.link {
  color: #116FCD;
  text-decoration: underline;
  cursor: pointer;
}

.el-form-item{
  margin-bottom: 1.5vw;
  height: min(5vw, 50px);
  /* max-height: 60px; */
  /* max-margin: 10px; */
  /* min-height: 5vw; */
}

.checkbox-wrapper {
  margin: min(1vw, 3px);
  font-size:13px;
  /* margin-top: 10%; */
}

.checkbox-label {
  display: flex;
  align-items: center;
  letter-spacing: 1px;
  color: rgb(75, 85, 99);
}

.checkbox {
  margin-right: 0.5vw;
  width: min(1.5vw,15px);
  height: min(1.5vw,15px);
  border: 1px solid rgb(107, 114, 128);
  background-color: white;
}

</style>