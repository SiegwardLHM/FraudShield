<template>
  <div class="head-image" @click="showUserInfo($event)" :title="name">
    <img v-if="url" :src="url" :style="avatarImageStyle" class="avatar-image" alt="头像"/>
    <div class="avatar-text" v-if="!url" :style="avatarTextStyle">
      {{ name.substring(0, 1).toUpperCase() }}
    </div>
    <div v-if="online" class="online" title="用户当前在线"></div>
  </div>
</template>

<script>
export default {
  name: "HeadImage",
  data() {
    return {
      colors: ["#7dd24b", "#c7515a", "#db68ef", "#15d29b", "#85029b", "#c9b455", "#fb2609", "#bda818", "#af0831", "#326eb6"]
    };
  },
  props: {
    id: {
      type: Number
    },
    size: {
      type: Number,
      default: 20
    },
    url: {
      type: String
    },
    name: {
      type: String,
      default: "?"
    },
    online: {
      type: Boolean,
      default: false
    }
  },
  methods: {
    showUserInfo(e) {
      if (this.id && this.id > 0) {
        this.$router.push({ path: `/user-info`, query: { id: this.id } });
      }
    }
  },
  computed: {
    avatarImageStyle() {
      return `width:${this.size}px; height:${this.size}px;`;
    },
    avatarTextStyle() {
      return `width: ${this.size}px; height:${this.size}px; color:${this.textColor}; font-size:${this.size * 0.6}px;`;
    },
    textColor() {
      let hash = 0;
      for (let i = 0; i < this.name.length; i++) {
        hash += this.name.charCodeAt(i);
      }
      return this.colors[hash % this.colors.length];
    }
  }
};
</script>

<style scoped>
.head-image {
  position: relative;
  cursor: pointer;

  .avatar-image {
    position: relative;
    overflow: hidden;
    border-radius: 10%;
    border: 1px solid #ccc;
    vertical-align: bottom;
  }

  .avatar-text {
    background-color: #f2f2f2;
    border-radius: 10%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #ccc;
  }

  .online {
    position: absolute;
    right: -10%;
    bottom: 0;
    width: 24px;
    height: 24px;
    background: limegreen;
    border-radius: 50%;
    border: 6px solid white;
  }
}
</style>
