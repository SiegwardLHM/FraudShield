<template>
  <div class="layout-container">
    <!-- 侧边菜单 -->

    <div class="menu_out" @mouseenter="expandMenu" @mouseleave="collapseMenu">
      <el-menu
        default-active="1"
        class="el-menu-vertical-demo"
        :collapse="isCollapse"
        @open="handleOpen" 
        @close="handleClose"
      >
      <router-link 
          v-for="(item, index) in tabBarList"
          :key="item.pagePath"
          :to="item.pagePath"
          class="menu-item-link"
        >
          <el-menu-item
            :index="String(index + 1)"
            :class="{ active: isActive(item.pagePath) }"
          >
            <img :src="isActive(item.pagePath) ? item.selectedIconPath : item.iconPath" class="menu-icon" />
            <span v-if="!isCollapse" class="menu-text">{{ item.text }}</span>
            <!-- <span v-if="!isCollapse">{{ item.text }}</span> -->
          </el-menu-item>
        </router-link>
      </el-menu>
    </div>

  </div>
</template>

<script>
import videoIcon from '@/static/tabbar/video.png';
import videoActiveIcon from '@/static/tabbar/video_active.png';
import chatIcon from '@/static/tabbar/chat.png';
import chatActiveIcon from '@/static/tabbar/chat_active.png';
import profileIcon from '@/static/tabbar/profile.png';
import profileActiveIcon from '@/static/tabbar/profile_active.png';
import communityIcon from '@/static/tabbar/community.png';
import communityActiveIcon from '@/static/tabbar/community_active.png';

export default {
  data() {
    return {
      isCollapse: true,
      tabBarList: [
        {
          pagePath: '/video',
          text: '音视频通话',
          iconPath: videoIcon,
          selectedIconPath: videoActiveIcon
        },
        {
          pagePath: '/robot',
          text: '离线检测',
          iconPath: chatIcon,
          selectedIconPath: chatActiveIcon
        },
        {
          pagePath: '/profile',
          text: '个人档案',
          iconPath: profileIcon,
          selectedIconPath: profileActiveIcon
        },
        {
          pagePath: '/community',
          text: '防诈社区',
          iconPath: communityIcon,
          selectedIconPath: communityActiveIcon
        }
      ]
    };
  },
  methods: {
    expandMenu() {
      this.isCollapse = false;
    },
    collapseMenu() {
      this.isCollapse = true;
    },
    isActive(path) {
      return this.$route.path === path;
    }
  }
};
</script>

<style scoped>
/* 设置父容器为flex布局 */
.layout-container {
  height: 100%; 
}

/* 主内容区域自动扩展，填满剩余空间 */
.content-container {
  flex: 1;
  /* padding: 20px; 主内容区域的内边距 */
  overflow-y: auto;
}

/* 侧边菜单样式 */
.menu_out {
  /* width: 70%; */
  /* margin-left: 20%;  */
  height: 100%;

  cursor: pointer;
  /* font-size: 0.3vw; */
  /* border-radius: 1vw; */
  display: flex;
  flex-direction: column;
  justify-content: space-around;

  /* border: 1px solid gray; */
}
::v-deep .el-menu {
  display: block; 
  width: 100%;
  height: 100%; /* 设置菜单总高度 */
  border: 0;
  background-color: #116FCD; /* 深蓝色背景 */
}

::v-deep .el-menu-item {
  width: 100%;
  height: 10%;
  display: flex; 
  align-items: center;
  justify-content: flex-start; /* 水平居中 */  
  color: #ffffff; /* 默认文字颜色为白色 */
  font-size: 20px;
  /* padding-left: 0%; */
  /* margin-bottom: 0.8vw; */
}

.el-menu-item .menu-text {
  font-size: 20px;
  letter-spacing: 2px;
  /* word-spacing: 3px; */
  /* color: #ffffff; */
  margin-left: 15px;
  /* 图标和文字之间的间距 */
}


.el-menu-item.active {
  font-weight: bold;
  /* font-size: 14px; */
  align-items: center;
  justify-content: flex-start; /* 水平居中 */  
  color: #116FCD;
  background-color: #ffffff;
  /* padding: 0; */
}
.el-menu-vertical-demo{
  /* border: 1px solid gray; */
  height: 100%;
  /* padding-top: 0.7vw; */
  /* border-radius: 1vw; */
  overflow: hidden; /* 防止内容溢出 */
  box-shadow: 0 2px 4px rgba(138, 136, 248, 0.2), 
  0 4px 8px rgba(153, 138, 241, 0.19);

}

.el-menu-vertical-demo:not(.el-menu--collapse) {
  width: 100%;


}

.el-menu-vertical-demo.el-menu--collapse {
  width: 80px;


}

.menu-icon {
  /* left: 1vw; */
  width: 35px;
  height: auto;
  position: relative;
}


.menu-item-link {
  text-decoration: none;
  color: inherit;

}

::v-deep .el-menu-item:hover {
  /* border-radius: 10px; */
  background-color: rgb(255, 255, 255);
  color:#116FCD; 

}
</style>
