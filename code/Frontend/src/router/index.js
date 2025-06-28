import { createRouter, createWebHashHistory } from 'vue-router';

import Register from '../pages/register/register.vue';
import Login from '../pages/login/login.vue';
import RobotChat from '../pages/robot/robot.vue';
import VideoChat from '../pages/chat/video-chat.vue';
import AudioChat from '../pages/chat/audio-chat.vue';
import Video from '../pages/video/video.vue';
import Audio from '../pages/audio/audio.vue';
import Profile from '../pages/profile/profile.vue';
import ProfileEdit from '../pages/profile/profile-edit.vue';
import ProfilePassword from '../pages/profile/profile-password.vue';
import ProfileHistory from '../pages/profile/profile-history.vue';
import Community from '../pages/community/community.vue';

const routes = [
  { path: '/', redirect: '/login' },
  { path: '/register', component: Register, meta: { title: '注册' } },
  { path: '/login', component: Login, meta: { title: '登录' } },
  { path: '/robot', component: RobotChat, meta: { title: '机器人聊天' } },
  { path: '/video-chat', component: VideoChat, meta: { title: '视频聊天' } },
  { path: '/audio-chat', component: AudioChat, meta: { title: '语音聊天' } },
  { path: '/video', component: Video },
  { path: '/audio', component: Audio },
  { path: '/profile', component: Profile },
  { path: '/profile-edit', component: ProfileEdit },
  { path: '/profile-password', component: ProfilePassword },
  { path: '/profile-history', component: ProfileHistory },
  { path: '/community', component: Community }
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = to.meta.title;
  }
  next();
});

export default router;
