// src/axios.js

import axios from 'axios';
import { ElMessage } from 'element-plus';
import router from './router';

// 创建 Axios 实例
const instance = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 10000000,
});

// 添加请求拦截器
instance.interceptors.request.use(
  config => {
    const token = localStorage.getItem('access');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// 添加响应拦截器
instance.interceptors.response.use(
  response => {
    return response;
  },
  async error => {
    if (error.response.status === 401) {
      ElMessage.error('认证失败，请重新登录');
      localStorage.clear();
      router.push('/login');
    }
    return Promise.reject(error);
  }
);

export default instance;
