<template>
  <div class="upload-image" @click="selectAndUpload">
    <slot></slot>
    <img v-if="imageUrl" :src="imageUrl" class="image-preview" alt="预览图片"/>
  </div>
</template>

<script>
import axios from 'axios';
import { ElMessage } from 'element-plus';

export default {
  name: "ImageUpload",
  data() {
    return {
      imageUrl: this.initialUrl,
      uploadHeaders: {
        "Authorization": `Bearer ${localStorage.getItem('access')}`
      }
    };
  },
  props: {
    initialUrl: {
      type: String,
      default: ''
    },
    maxCount: {
      type: Number,
      default: 1
    },
    maxSize: {
      type: Number,
      default: 5 * 1024 * 1024
    },
    sourceType: {
      type: String,
      default: 'album'
    },
    onBefore: {
      type: Function,
      default: null
    },
    onSuccess: {
      type: Function,
      default: null
    },
    onError: {
      type: Function,
      default: null
    }
  },
  methods: {
    selectAndUpload() {
      this.$refs.fileInput.click();
    },
    async handleFileChange(event) {
      const file = event.target.files[0];
      if (file) {
        if (this.maxSize && file.size > this.maxSize) {
          ElMessage({
            message: `文件大小不能超过 ${this.fileSizeStr}!`,
            type: 'error'
          });
          this.$emit("fail", file);
          return;
        }

        if (!this.onBefore || this.onBefore(file)) {
          this.uploadImage(file);
        }
      }
    },
    async uploadImage(file) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await axios.post('http://localhost:8000/api/auth/upload/', formData, {
          headers: {
            ...this.uploadHeaders,
            'Content-Type': 'multipart/form-data'
          }
        });
        if (response.status !== 201) {
          this.onError && this.onError(file, response.data);
        } else {
          this.imageUrl = response.data.thumbUrl; // 更新显示的图片URL
          this.onSuccess && this.onSuccess(file, response.data);
        }
      } catch (err) {
        console.log(err);
        this.onError && this.onError(file, err);
      }
    }
  },
  computed: {
    fileSizeStr() {
      if (this.maxSize > 1024 * 1024) {
        return Math.round(this.maxSize / 1024 / 1024) + "M";
      }
      if (this.maxSize > 1024) {
        return Math.round(this.maxSize / 1024) + "KB";
      }
      return this.maxSize + "B";
    }
  },
  watch: {
    initialUrl(newUrl) {
      this.imageUrl = newUrl;
    }
  }
};
</script>

<style scoped>
.upload-image {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image-preview {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  object-fit: cover;
}
</style>
