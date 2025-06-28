<template>
  <div class="app-layout">

    <div v-if="showTabBar" class="menu-container">
      <TabBar />
    </div>
    
    <!-- 主内容区域，始终显示 -->
    <div :class="['content-container', { 'full-width': !showTabBar }]">
      <router-view />
    </div>
  </div>
</template>

<script>
import TabBar from '@/components/TabBar.vue'; // 侧边菜单的组件

export default {
  components: {
    TabBar
  },
  computed: {
    showTabBar() {
      const pathsWithoutTabBar = ['/login', '/profile-edit', '/profile-password', '/profile-history', '/register', '/audio-chat', '/video-chat'];
      return !pathsWithoutTabBar.includes(this.$route.path);
    }
  }
}
</script>

<style scoped>
/* 父容器使用flex布局，确保侧边菜单和内容区域并排显示 */
.app-layout {
  display: flex;
  height: 100%;
  /* background-image: url(@/assets/img/backGround2.png); */
  /* background-size: cover;
  background-position: center;
  background-color: rgba(255, 255, 255, 0);
  background-blend-mode: overlay; */
}

/* 侧边菜单的宽度 */
.menu-container {
  /* margin-top: 28vh; */
  width: 200px; /* 可以根据需要调整宽度 */
}

/* 主内容区域 */
.content-container {
  flex: 1; /* 占据剩余空间 */
  /* padding: 20px; */
  overflow-y: auto;
  /* border: 1px solid #eee; */
}

/* 当不显示侧边菜单时，内容区域全宽 */
.full-width {
  width: 100%;
}
</style>
