<template>
    <div>
      <!-- Banner start -->
      <div class="header-title">
        <i class="float-left"><img src="images/title-left.png" alt="left icon" /></i>
        <span class="main-title float-left">大屏数据统计</span>
        <i class="float-right"><img src="images/title-right.png" alt="right icon" /></i>
      </div>
  
      <!-- Main content -->
      <div class="content-wrapper">
        <div class="left-section float-left" style="position: relative; top: 0px;">
          <!-- Project Personnel Info Status -->
          <div class="data-box left-container float-left">
            <i class="corner-top-left"></i>
            <i class="corner-top-right"></i>
            <i class="corner-bottom-left"></i>
            <i class="corner-bottom-right"></i>
            <div class="data-title">
              <b class="data-title-left float-left">[</b>
              <span class="data-title-text">项目人员信息状态</span>
              <b class="data-title-right float-right">]</b>
            </div>
  
            <div class="project-overview" style="text-align: center;">
              <!-- Each pie chart display -->
              <div class="chart-display">
                <span class="overview-title">务工人员工种统计</span>
                <canvas id="pie1" width="180" height="180"></canvas>
              </div>
  
              <div class="chart-display">
                <span class="overview-title">务工人员工种统计</span>
                <canvas id="pie2" width="180" height="180"></canvas>
              </div>
  
              <div class="chart-display">
                <span class="overview-title">各公司现场人员统计</span>
                <canvas id="pie3" width="180" height="180"></canvas>
              </div>
            </div>
          </div>
        </div>
  
        <!-- Center display -->
        <div class="center-section float-left">
          <div class="data-box background-container">
            <i class="corner-top-left"></i>
            <i class="corner-top-right"></i>
            <i class="corner-bottom-left"></i>
            <i class="corner-bottom-right"></i>
            <div class="data-title">
              <b class="data-title-left">[</b>
              <span class="data-title-text">仪表盘等展示</span>
              <b class="data-title-right">]</b>
            </div>
            <p class="info-text">
              是否伪造：是
              类别：冒充政府人员，客服
              1.身份不明的客服：正规的公司不会通过电话或社交平台主动联系用户，特别是在涉及续费或财务问题时。
              2.要求下载APP：正规公司不会要求用户下载陌生的第三方APP解决账户问题。
              3.冒充“金融客服专员”：正规公司不会要求用户提供银行卡的具体余额，尤其是通过非官方渠道。
              4.验证码发送要求：正规银行或平台不会要求用户发送验证码，这是典型的诈骗手法。
            </p>
          </div>
        </div>
  
        <!-- Right display -->
        <div class="right-section float-left">
          <div class="data-box right-container">
            <i class="corner-top-left"></i>
            <i class="corner-top-right"></i>
            <i class="corner-bottom-left"></i>
            <i class="corner-bottom-right"></i>
            <div class="data-title">
              <b class="data-title-left">[</b>
              <span class="data-title-text">巡检风险状态</span>
              <b class="data-title-right">]</b>
            </div>
  
            <div class="inspection-status">
              <div id="risk-status">
                <div class="video-block">
                  <video src="./output.mp4" class="video-display" controls></video>
                  <video src="./output.mp4" class="video-display" controls></video>
                  <video src="./output.mp4" class="video-display" controls></video>
                </div>
                <div class="video-block">
                  <video src="./output.mp4" class="video-display" controls></video>
                  <video src="./output.mp4" class="video-display" controls></video>
                  <video src="./output.mp4" class="video-display" controls></video>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import Chart from './acc.js';
  
  export default {
    name: 'DataDashboard',
    mounted() {
      new Chart("pie1").ratePie(85);
      new Chart("pie2").ratePie(90);
      new Chart("pie3").ratePie(80);
    },
  };
  </script>
  
  <style scoped>
  * {
    padding: 0;
    margin: 0;
    font-family: "微软雅黑";
    font-size: 12px;
    font-style: normal;
  }
  
  a {
    text-decoration: none;
  }
  
  img {
    border: 0;
  }
  
  ul,
  li {
    padding: 0;
    margin: 0;
    list-style: none;
  }
  
  .float-left {
    float: left;
  }
  
  .float-right {
    float: right;
  }
  
  body {
    margin: 0;
    padding: 0;
    color: #000;
    background: #000c3b;
  }
  
  .header-title {
    font-size: 22px;
    font-family: "黑体";
    color: #fff;
    max-width: 900px;
    margin: 10px auto;
    height: 48px;
  }
  
  .main-title {
    line-height: 48px;
    display: inline-block;
    width: 276px;
    text-align: center;
    font-size: 28px;
  }
  
  .data-box {
    border: 2px solid #032d60;
    box-shadow: inset 0 0 30px #07417a;
    position: relative;
  }
  
  .content-wrapper {
    width: 96%;
    margin: 0 auto;
    overflow: hidden;
    padding-top: 30px;
  }
  
  .left-section {
    width: 20%;
  }
  
  .left-container {
    width: 95%;
    height: 85vh;
  }
  
  .right-section {
    width: 40%;
  }
  
  .right-container {
    width: 95%;
    float: right;
    height: 85vh;
  }
  
  .corner-top-left, .corner-top-right, .corner-bottom-left, .corner-bottom-right {
    width: 20px;
    height: 20px;
    position: absolute;
  }
  
  .corner-top-left, .corner-top-right {
    top: -2px;
  }
  
  .corner-bottom-left, .corner-bottom-right {
    bottom: -2px;
  }
  
  .corner-top-left, .corner-bottom-left {
    left: -2px;
  }
  
  .corner-top-right, .corner-bottom-right {
    right: -2px;
  }
  
  .corner-top-left, .corner-top-right {
    border-top: 2px solid #26c6f0;
  }
  
  .corner-bottom-left, .corner-bottom-right {
    border-bottom: 2px solid #26c6f0;
  }
  
  .data-title {
    background-color: #000c3b;
    width: 178px;
    margin: -18px auto 0;
    color: #83c7e3;
    font-size: 20px;
  }
  
  .data-title-text {
    text-align: center;
    width: 150px;
    display: inline-block;
    font-size: 18px;
  }
  
  .project-overview {
    width: 100%;
    color: #fff;
    padding: 20px 0;
  }
  
  .overview-title {
    height: 100%;
    padding: 10px;
    display: block;
    border-top: 1px #032d60 solid;
    color: #fff;
    font-size: 1.5vh;
    margin-top: 2vh;
  }
  
  .center-section {
    width: 40%;
  }
  
  .background-container {
    height: 85vh;
    background-image: radial-gradient(250px 250px at 50% 50%, hsla(209, 99%, 32%, 0.28), hsla(228, 100%, 0%, 0));
  }
  
  .video-block {
    border: 1px solid white;
    width: 100%;
    margin-top: 5vh;
    height: 35vh;
  }
  
  .video-display {
    height: 100%;
    width: 30%;
    display: flex;
    margin-left: 3%;
    float: left;
  }
  
  .info-text {
    font-size: 2.5vh;
    color: white;
    white-space: pre-line;
  }
  </style>
  