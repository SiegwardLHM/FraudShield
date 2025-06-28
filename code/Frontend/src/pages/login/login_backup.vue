<template>
  <div class="container">
    <div class="card">
      <div class="content">
        <div class="form-wrapper">
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
                没有账号？<span @click="goToRegister" class="link">去注册</span>
              </p>
            </el-form>
          </div>
        </div>
        <div class="image-wrapper">
          <img aria-hidden="true" class="image light" src="@/assets/img/backGround4.png" alt="Office" />
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
        password: ''
      },
      rules: {
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
              // 直接设置本地存储中的 access 和 refresh token（可用占位值，如 'mockAccess' 和 'mockRefresh'）
              localStorage.setItem('access', 'mockAccess');
              localStorage.setItem('refresh', 'mockRefresh');

              this.$message.success('登录成功');
              this.$router.push('/profile');
              return; // 直接返回，跳过后续的请求发送
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
  /* padding: 1.5rem;
  background-color: rgb(249, 250, 251); */
}
.card {
  width: 80%;
  max-width: 1000px;
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
}
.image-wrapper {
  flex: 1; 
  max-width: 50%; 
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
  display: block;
}
.form-wrapper {
  flex: 1; 
  width: 50%;
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
  font-size: 1.5rem;
  font-weight: 600;
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
  /* margin-top: 1; */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2), 
              0 6px 15px rgba(0, 0, 0, 0.19);
}
.button:hover {
  background-color: #ff5675;
  cursor: pointer;
}
.divider {
  /* margin: 1.5rem 0; */
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
</style>
