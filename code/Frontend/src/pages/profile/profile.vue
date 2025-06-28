<template>
  <div class="page mine">
    <div class="content" @click="onModifyInfo">
      <img :src="'/src/assets/img/fake.jpg'" class="avatar" alt="头像">
      <div class="info-item">
        <div class="info-primary">
          <span class="info-username">{{ userInfo.username }}</span>
          <span v-show="userInfo.sex === 0" class="iconfont icon-man"></span>
          <span v-show="userInfo.sex === 1" class="iconfont icon-girl"></span>
        </div>
        <span class="info-text">昵称 ：{{ userInfo.nickName }}</span>
        <span class="info-text">签名 ：{{ userInfo.signature }}</span>
      </div>
      <div class="info-arrow">></div>
    </div>
    <div class="line"></div>
    <div class="btn-group">
      <el-card class="custom-card" @click="onModifyPassword">
        <div class="card-content">
          <img src="@/static/icon/password.png" alt="修改密码" class="card-icon">
          <span class="card-text">修改密码</span>
          <div class="info-arrow">></div>
        </div>
      </el-card>
      <el-card class="custom-card" @click="onViewHistory">
        <div class="card-content">
          <img src="@/static/icon/history.png" alt="查看历史检测记录" class="card-icon">
          <span class="card-text">历史检测记录 &nbsp / &nbsp 诈骗个性化分析</span>
          <div class="info-arrow">></div>
        </div>
      </el-card>
      <el-button @click="onQuit" class="custom-button">退出登录</el-button>
    </div>
  </div>
</template>

<script>
import axios from '../../axios';
import { ElMessageBox, ElMessage } from 'element-plus';
import HeadIcon from '@/assets/img/fake.jpg';
export default {
  data() {
    return {
      userInfo: {
        username: '',
        nickName: '123',
        headImage: '/src/assets/img/fake.jpg',
        sex: 0,
        signature: '123'
      }
    };
  },
  methods: {
    onModifyInfo() {
      this.$router.push('/profile-edit');
    },
    onModifyPassword() {
      this.$router.push('/profile-password');
    },
    onViewHistory(){
      this.$router.push('/profile-history');
    },
    onQuit() {
      ElMessageBox.confirm('确认退出?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        localStorage.clear();
        this.$router.replace('/login');
      }).catch(() => {});
    },
    async fetchUserInfo() {
      try {
        const response = await axios.get('/auth/me/');
        this.userInfo = response.data;
        localStorage.setItem('userInfo', JSON.stringify(this.userInfo));
      } catch (error) {
        console.error('Error fetching user info:', error);
      }
    },
  },
  created() {
    const storedUserInfo = JSON.parse(localStorage.getItem('userInfo'));
    if (storedUserInfo) {
      this.userInfo = storedUserInfo;
    } else {
      this.fetchUserInfo();
    }
  }
};
</script>

<style scoped>
.page{
  /* border: 1px solid gray; */
  margin-top: 0%;
  /* border-left: 2px dashed gray; */
  height: 80%;
  padding-top: 10%;
}

.mine .content {
  display: flex;
  align-items: center;
  /* padding: 20px; */
  /* margin-top: 10%; */
  background: #ffffff;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  cursor: pointer;
  transition: transform 0.2s;
  width: 80%;
  margin-left: 10%;

}

.mine .content:hover {
  transform: translateY(-5px);
}

.info-item {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
  padding-left: 20px;
  flex: 1;
}

.info-primary {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.info-username {
  font-size: 18px;
  font-weight: 600;
  margin-right: 10px;
}

.icon-man {
  color: darkblue;
  margin-left: 5px;
}

.icon-girl {
  color: darkred;
  margin-left: 5px;
}

.info-arrow {
  font-size: 20px;
  font-weight: 600;
}

.content .info-arrow {
  margin-right: 20px;
}

.line {
  margin: 20px 0;
  border-bottom: 1px solid #eeeeee;
  width: 80%;
  margin-left: 10%;
}

.btn-group {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 80%;
  margin-left: 10%;
  gap: 10px; /* 使用 gap 来控制子元素之间的间距 */
}

.custom-button {
  width: 80%;
  height: 60px;

  border-radius: 20px;
  font-size: 20px;
  font-weight: bolder;

  background-color:#c43d60;
  color:#ffffff;
}

.custom-card {
  width: 100%;
  cursor: pointer;
  margin-bottom: 2%;
}

.card-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight:bolder;
}

.card-icon {
  width: 20px;
  height: 20px;
  margin-right: 10px;
}

.card-text {
  flex: 1;
  text-align:left;
  margin-left: 20px;
}

.avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: 2px solid #eee;
  transition: box-shadow 0.2s;
}

.avatar:hover {
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
}

.info-text {
  font-size: 14px;
  color: #666;
  margin-bottom: 5px;
}
</style>
