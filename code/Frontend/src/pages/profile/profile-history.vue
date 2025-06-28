<template>
    <div class="page mine-edit">
        <div class="header">
            <span @click="goBack" class="back-button">&lt;</span>
            <h2 class="title">历史检测记录 &nbsp / &nbsp 诈骗个性化分析</h2>
        </div>
        <div class="content">
            <div class="side-bar">
                <!-- 个性化定制 -->
                <div @click="toggleCustomOption" class="sidebar-item custom-option"
                    :class="{ 'active': activeItem === 'custom-option' }">
                    个性化分析
                </div>
                <!-- 分割线 -->
                <div class="divider"></div>
                <!-- 历史检测记录 -->
                <div class="sidebar-item history-title">历史检测记录</div>
                <!-- 动态生成的历史记录选项 -->
                <div v-for="(record, index) in historyRecords" :key="index" class="sidebar-item"
                    :class="{ 'active': activeItem === index }" @click="setActiveItem(index)">
                    {{ record }}
                </div>
            </div>
            <!-- 右侧模块 -->
            <div v-if="showCustomOption" class="right-module">
                <div class="top-module">
                    <div class="title-bar">历史检测记录分析</div>
                    <!-- 上层模块内容 -->
                    <div class="module-content">
                        <div class="content-block">
                            <div ref="chart" style="width: 100%; height: 100%;"></div>
                        </div>
                        <div class="content-block">
                            <span class="analysis-title">诈骗类型分析</span>
                            <span class="analysis-content">
                                <ul>
                                    <li><strong>刷单诈骗</strong>占比60%。勿信虚假交易，避免预付费用，确认平台正规性。</li>
                                    <li><strong>中奖诈骗</strong>占比30%。警惕中奖信息，确认来源真实性，不轻易支付手续费。</li>
                                    <li><strong>冒充公检法诈骗</strong>占比10%。核实身份信息，不转账给陌生人，遇可疑情况立即报警。</li>
                                </ul>
                            </span>
                        </div>
                        <div class="content-block">
                            <span class="analysis-title">易骗特征分析</span>
                            <span class="analysis-content">
                                <ul>
                                    <li><strong>常见诈骗词汇：</strong>刷单、中奖、公安、法院等高频词汇。</li>
                                    <li><strong>常见诈骗话术：</strong>“轻松赚钱”、“恭喜您中奖”、“涉嫌犯罪需配合调查”。</li>
                                    <li><strong>易受骗情况：</strong>贪小便宜心理、缺乏安全意识、对权威机构盲目信任。</li>
                                </ul>
                            </span>
                        </div>
                        <div class="content-block">
                            <span class="analysis-title">防诈建议</span>
                            <span class="analysis-content">
                                <ul>
                                    <li>保持警惕，不轻信陌生人的信息。</li>
                                    <li>核实信息来源，确认对方身份。</li>
                                    <li>避免泄露个人信息，尤其是财务信息。</li>
                                    <li>不轻易转账或支付费用。</li>
                                    <li>遇到可疑情况，及时报警或咨询专业人士。</li>
                                </ul>
                            </span>
                        </div>
                    </div>
                </div>
                <div class="bottom-module">
                    <div class="left-bottom-module">
                        <div class="title-bar">个性化防诈知识集锦</div>
                        <!-- 下层左侧模块内容 -->
                        <div class="module-content">
                            <div class="left-knowledge">
                                <div class="left-knowledge-top">
                                    <div class="knowledge-title">相关热点时讯：</div>
                                    <div v-if="currentImageIndex == 0" class="image-container">
                                        <img src="@/assets/img/news_1.png" alt="News Image" class="news-image" />
                                        <a href="https://www.chinacourt.org/article/detail/2021/04/id/5967868.shtml"
                                            target="_blank" class="image-link"><span>坚持以人民为中心 坚决打击电信网络诈骗</span></a>
                                    </div>
                                    <div v-else-if="currentImageIndex == 1" class="image-container">
                                        <img src="@/assets/img/news_2.jpg" alt="News Image" class="news-image" />
                                        <a href="https://www.msn.cn/zh-cn/money/%E9%80%9A%E7%94%A8/%E8%AD%A6%E6%96%B9%E7%B4%A7%E6%80%A5%E6%8F%90%E9%86%92-%E4%B8%AD%E9%93%B6%E4%BC%9A%E8%AE%AE-%E9%93%B6%E8%81%94%E4%BC%9A%E8%AE%AE-%E9%83%BD%E6%98%AF%E8%AF%88%E9%AA%97%E8%BD%AF%E4%BB%B6/ar-AA1A2u0E?ocid=BingNewsSerp"
                                            target="_blank"
                                            class="image-link"><span>警方紧急提醒：“中银会议”“银联会议”都是诈骗软件</span></a>
                                    </div>
                                    <div v-else-if="currentImageIndex == 2" class="image-container">
                                        <img src="@/assets/img/news_3.jpg" alt="News Image" class="news-image" />
                                        <a href="https://www.163.com/dy/article/JPJLEH1H0525KHJ9.html?f=post2020_dy_recommends"
                                            target="_blank" class="image-link"><span>警惕披着赌博外衣的网络诈骗，仁怀已有人中招</span></a>
                                    </div>
                                </div>
                                <div class="left-knowledge-bottom">
                                    <div class="knowledge-title">相关防诈资料：</div>
                                    <ul class="fraud-resources">
                                        <li>
                                            <a href="https://www.thepaper.cn/newsDetail_forward_25109331"
                                                target="_blank">
                                                国家反诈中心公布10类高发诈骗类型
                                            </a>
                                        </li>
                                        <li>
                                            <a href="https://www.bilibili.com/video/BV1R64y1e711/?vd_source=724e3e100d4ba655efa15c8b199fdc70"
                                                target="_blank">
                                                反电信诈骗宣传片
                                            </a>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                            <div class="right-knowledge">
                                <div class="knowledge-title">相关诈骗语句：</div>
                                <ul class="fraud-phrases">
                                    <li>“您好，您中奖了，请提供您的个人信息。”</li>
                                    <li>“请尽快转账到指定账户，否则将失去中奖机会。”</li>
                                    <li>“我是XX公安局的，您涉嫌洗钱，请配合调查。”</li>
                                    <li>“您的包裹有问题，请点击链接查看。”</li>
                                    <li>“您有一笔未处理的订单，请登录网站确认。”</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="right-bottom-module">
                        <div class="title-bar">高风险诈骗预警</div>
                        <!-- 下层右侧模块内容 -->
                        <div class="module-content">
                            <div class="left-warn-content">
                                <div class="warn-title">
                                    近期高风险案例分析🔍
                                </div>

                                <strong>刷单诈骗</strong>

                                系统近期监测到多起刷单诈骗案件，犯罪分子通常会通过社交平台、短信或电话，声称“动动手指就能轻松赚钱”，诱导受害者下载指定的App或加入刷单群。
                                <br>

                                受害者完成初次刷单后可能会收到小额返利，犯罪分子利用这种“正反馈”逐渐建立信任，随后要求受害者进行大额刷单或支付“解冻费”、“保证金”等，最终在受害者支付高额款项后直接拉黑或失联。
                            </div>

                            <div class="right-warn-content">
                                <div class="right-top-warn-content">
                                    <div class="warn-title">
                                        ✅ 特征提示:
                                    </div>
                                    <ul class="custom-ul">
                                        <li>出现“高额汇报”、“轻松赚钱”等宣传语时务必提高警惕。</li>
                                        <li>正规客服从不通过私人电话或非官方渠道联系用户。</li>
                                        <li>官方客服不会要求下载任何第三方定位应用。</li>
                                    </ul>
                                </div>
                                <div class="right-bottom-warn-content">
                                    <div class="warn-title">
                                        🚨 安全建议:
                                    </div>
                                    遇到类似情况，建议立刻挂断电话，切勿透露个人信息。
                                    如已涉及财产损失，请第一时间联系银行冻结账户并报警。
                                    下载国家反诈中心App，实时接收反诈预警和最新诈骗动态。
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div v-else class="right-module">
                <div class="container">
                    <div class="all-content">
                        <!-- Main content -->
                        <div class="content-wrapper">
                            <div class="left-content">
                                <div class="title-bar">诈骗数据</div>
                                <div class="data-fruad"
                                    style="justify-content: center;align-items: center;display: flex;color: white;">
                                    <span class="data-title-text">诈骗概率{{ displayContent.Confidence.fraud * 100 }}</span>
                                </div>
                                <div class="project-overview" style="text-align: center;">
                                    <!-- Each pie chart display -->
                                    <div class="chart-display" v-show="displayContent.Confidence.carrier * 100 > 0">
                                        <span class="overview-title">通话载体</span>
                                        <canvas id="pie1" width="120" height="120"></canvas>
                                    </div>

                                    <div class="chart-display" v-show="displayContent.Confidence.action * 100 > 0">
                                        <span class="overview-title">通话行为</span>
                                        <canvas id="pie2" width="120" height="120"></canvas>
                                    </div>

                                    <div class="chart-display" v-show="displayContent.Confidence.content * 100 > 0">
                                        <span class="overview-title">{{ displayContent.Confidence.action > 0 ? '通话语义' :
                                            '内容语义' }}</span>
                                        <canvas id="pie3" width="120" height="120"></canvas>
                                    </div>
                                </div>
                            </div>

                            <!-- Center display -->
                            <div class="mid-content">
                                <div class="title-bar">诈骗语义分析</div>
                                <p class="info-text"
                                    style="white-space:pre-line;overflow-y: auto;height: 100%;padding-bottom: 20px;">
                                    {{ displayContent.text }}
                                </p>
                            </div>

                            <!-- Right display -->
                            <div class="right-content">
                                <div class="title-bar">诈骗载体 / 行为分析</div>
                                <div class="inspection-status">
                                    <div v-if="displayContent.video.length > 0">
                                        <div class="video-block">
                                            <div v-for="(url, index) in displayContent.video" :key="index"
                                                class="video-display">
                                                <div class="text-content">{{ imgLabel[index] }}</div>
                                                <img :src="url" :alt="url" />
                                            </div>
                                        </div>
                                        <div class="video-block">
                                            <div v-for="(url, index) in displayContent.action" :key="index"
                                                class="video-display">
                                                <div class="text-content">{{ imgLabel[index + 3] }}</div>
                                                <img :src="url" :alt="url" />
                                            </div>
                                        </div>
                                    </div>
                                    <p v-else class="info-text" style="white-space:pre-line;">{{ displayContent.transfer
                                        }}</p>
                                </div>
                            </div>
                        </div>

                        <div class="chart-content">
                            <div class="timeline-chart" ref="timelineChart"></div>
                        </div>
                    </div>

                    <!-- 新增的 robot-chat -->
                    <div class="robot-chat">
                        <div class="chat-title">智能诈骗风险摘要助手</div>
                        <div class="chat-box">
                            <div class="message user">你好！</div>
                            <div class="message bot">你好！有什么可以帮助你的吗？</div>
                            <div class="message user">我想了解一下你们的产品。</div>
                            <div class="message bot">当然可以。我们的产品主要有……</div>
                        </div>
                        <div class="chat-input">
                            <textarea v-model="userInput" placeholder="请输入消息..."></textarea>
                            <button @click="sendMessage">发送</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import axios from '../../axios'; // 使用自定义的 axios 实例
import * as echarts from 'echarts';
import Chart from '@/assets/styles/acc';

export default {
    data() {
        return {
            // 模拟历史检测记录数据
            historyRecords: [
                "2023/10/01  语音检测",
                "2023/10/05  视频检测",
                "2023/10/10  离线检测",
                "2023/10/15  语音检测"
            ],
            showCustomOption: true, // 控制右侧模块的显示与隐藏
            activeItem: 'custom-option', // 存储当前激活的 sidebar-item
            chartInstance: null, // 用于存储 ECharts 实例
            currentImageIndex: 0,
            userInput: '', // 用户输入的消息
            displayContent: {
                Confidence: {
                    fraud: 0.8,
                    carrier: 0.6,
                    action: 0.7,
                    content: 0.9
                },
                text: "这是诈骗语义分析的内容...",
                video: ["https://example.com/video1.mp4", "https://example.com/video2.mp4"],
                action: ["https://example.com/action1.jpg", "https://example.com/action2.jpg"],
                transfer: "其他内容..."
            },
            imgLabel: ['人脸伪造', '表情心虚', '摸鼻子', '摸脖子', '斜视', '摸下巴'],
            timelineChartData: [
                { time: '00:00', confidence: 0.1, faceForgery: '正常', textDetection: '正常', emotionDetection: '正常' },
                { time: '00:01', confidence: 0.2, faceForgery: '可疑', textDetection: '正常', emotionDetection: '正常' }
            ],
            timelineChartInstance: null
        };
    },
    methods: {
        goBack() {
            this.$router.go(-1);
        },
        toggleCustomOption() {
            this.showCustomOption = true;
            this.activeItem = 'custom-option';
        },
        setActiveItem(index) {
            this.showCustomOption = false;
            this.activeItem = index;
        },
        initChart() {
            if (this.chartInstance) {
                this.chartInstance.dispose();
            }
            this.chartInstance = echarts.init(this.$refs.chart);

            const option = {
                title: {
                    text: '诈骗类型及比例',
                    left: 'center'
                },
                tooltip: {
                    trigger: 'item'
                },
                legend: {
                    orient: 'vertical',
                    left: 'left'
                },
                series: [
                    {
                        name: '诈骗类型',
                        type: 'pie',
                        radius: '60%',
                        label: {
                            show: false
                        },
                        data: [
                            { value: 60, name: '刷单诈骗' },
                            { value: 30, name: '中奖诈骗' },
                            { value: 10, name: '冒充公检法诈骗' }
                        ],
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 10,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }
                ]
            };

            this.chartInstance.setOption(option);
        },
        initTimelineChart() {
            const chart = echarts.init(this.$refs.timelineChart);
            const option = {
                title: {
                    text: '检测时间轴图',
                    left: 'center'
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: (params) => {
                        const param = params[0];
                        return `
                        <div>
                            <p>时间: ${param.name}</p>
                            <p>置信度: ${param.value}</p>
                            <p>人脸伪造检测: ${param.data.faceForgery}</p>
                            <p>文本检测: ${param.data.textDetection}</p>
                            <p>情绪检测: ${param.data.emotionDetection}</p>
                        </div>
                    `;
                    },
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    borderColor: '#ccc',
                    borderWidth: 1,
                    textStyle: {
                        color: '#ffffff',
                        fontSize: 15
                    },
                    padding: 10,
                    extraCssText: 'z-index: 1000;'
                },
                xAxis: {
                    type: 'category',
                    data: this.timelineChartData.map(item => item.time),
                    boundaryGap: false
                },
                yAxis: {
                    type: 'value',
                    min: 0,
                    max: 1,
                    interval: 1,
                    splitNumber: 1
                },
                series: [
                    {
                        data: this.timelineChartData.map(item => ({
                            value: item.confidence,
                            faceForgery: item.faceForgery,
                            textDetection: item.textDetection,
                            emotionDetection: item.emotionDetection
                        })),
                        type: 'line',
                        smooth: true,
                        symbol: 'circle',
                        symbolSize: 10,
                        lineStyle: {
                            color: '#116FCD',
                            width: 2
                        },
                        itemStyle: {
                            color: '#116FCD'
                        },
                        areaStyle: {
                            color: {
                                type: 'linear',
                                x: 0,
                                y: 0,
                                x2: 0,
                                y2: 1,
                                colorStops: [
                                    { offset: 0, color: 'rgba(17, 111, 205, 0.3)' },
                                    { offset: 1, color: 'rgba(17, 111, 205, 0)' }
                                ]
                            }
                        }
                    }
                ],
                grid: {
                    left: '4%',
                    right: '4%',
                    bottom: '0',
                    containLabel: true
                }
            };
            chart.setOption(option, true);
            this.timelineChartInstance = chart;
        },
        sendMessage() {
            if (this.userInput.trim()) {
                const userMessage = document.createElement('div');
                userMessage.className = 'message user';
                userMessage.textContent = this.userInput;

                const botMessage = document.createElement('div');
                botMessage.className = 'message bot';
                botMessage.textContent = '这是机器人的回复：' + this.userInput;

                const chatBox = document.querySelector('.chat-box');
                chatBox.appendChild(userMessage);
                chatBox.appendChild(botMessage);

                this.userInput = '';
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        },
    },
    watch: {
        showCustomOption(newVal) {
            if (newVal && this.activeItem === 'custom-option') {
                this.$nextTick(() => {
                    this.initChart(); // 当显示自定义选项时重新初始化图表
                });
            }
            if (!newVal) {
                this.$nextTick(() => {
                    new Chart("pie1").ratePie(this.displayContent.Confidence.carrier * 100);
                    new Chart("pie2").ratePie(this.displayContent.Confidence.action * 100);
                    new Chart("pie3").ratePie(this.displayContent.Confidence.content * 100);
                    this.initTimelineChart();
                });
            }
        }
    },
    mounted() {
        this.initChart(); // 在组件挂载时初始化图表

        // 添加定时器来切换图片
        setInterval(() => {
            this.currentImageIndex = (this.currentImageIndex + 1) % 3;
        }, 5000); // 每隔3秒切换一次图片
    },
};
</script>

<style scoped>
.mine-edit {
    height: 100%;
    background: #f7f8fa;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.header {
    height: 7%;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    background-color: #fff;
    box-shadow: 2px 5px 10px 0 rgba(47, 56, 111, 0.1);
    position: relative;
}

.back-button {
    font-size: 24px;
    cursor: pointer;
    /* margin-left: 5%; */
    margin-right: auto;
    padding-left: 10px;
    /* 适当调整左边距 */
}

.title {
    font-size: 20px;
    font-weight: bold;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
}

.content {
    width: 100%;
    background: #fff;
    height: 100%;
    display: flex;
}

.side-bar {
    width: 13%;
    height: 100%;
    background-color: #116FCD;
    box-shadow: 7px 0px 15px 0px rgba(47, 56, 111, 0.5);
    /* display: flex; */
    flex-direction: column;
    align-items: flex-start;
    box-sizing: border-box;
}

.sidebar-item {
    color: white;
    font-size: 18px;
    height: 8%;
    width: 100%;
    cursor: pointer;
    transition: color 0.2s;
    display: flex;
    justify-content: center;
    align-items: center;
}

.sidebar-item:hover {
    background-color: #fff;
    color: #116FCD;
}

.sidebar-item.active {
    background-color: #fff;
    color: #116FCD;
}

.custom-option {
    font-weight: bold;
    font-size: 23px;
}

.history-title {
    font-weight: bold;
    font-size: 23px;
    cursor: default;
    /* margin-top: 10px; */
}

.history-title:hover {
    background-color: transparent;
    /* 移除悬停效果 */
    color: white;
    /* 保持文字颜色不变 */
}

.divider {
    width: 100%;
    height: 1px;
    background-color: white;
}


.right-module {
    width: 85%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    /* 水平居中 */
    justify-content: center;
}

.top-module {
    height: 40%;
    width: 90%;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.2);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    /* 设置为垂直布局 */
    align-items: center;
    justify-content: flex-start;
    margin-bottom: 15px;
    /* display: flex; */
    /* justify-content: center; */
    /* align-items: center; */
}

.module-content {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    /* padding: 10px; */
    /* 添加内边距 */
}

.bottom-module {
    height: 47%;
    width: 90%;
    display: flex;
    align-items: center;
    /* 水平居中 */
    justify-content: space-between;
    margin-top: 15px;
}

.left-bottom-module,
.right-bottom-module {
    width: 48%;
    height: 100%;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.2);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    /* 设置为垂直布局 */
    align-items: center;
    justify-content: flex-start;
}

.title-bar {
    background-color: #116FCD;
    color: #fff;
    height: 50px;
    width: 100%;
    font-size: 25px;
    letter-spacing: 1.5px;
    font-weight: bold;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    padding-left: 1vw;
    padding-right: 0.5vw;
    box-sizing: border-box;
    /* 确保 padding 和 border 被包含在宽度内 */
}

.content-block {
    width: 25%;
    height: 100%;
    display: flex;
    flex-direction: column;
    /* 使用列布局，使内容按行排列 */
    align-items: center;
    /* 水平居中 */
    /* justify-content: space-around; */
    /* 垂直居中 */
    box-sizing: border-box;
    /* 确保 padding 和 border 被包含在宽度内 */
}

.analysis-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}

.analysis-content {
    font-size: 16px;
    line-height: 1.6;
    padding: 0 25px;
}

.analysis-content ul {
    padding: 0;
}

.analysis-content li {
    margin-bottom: 8px;
}

.left-knowledge {
    width: 60%;
    height: 100%;
    padding-left: 15px;
}

.left-knowledge-top {
    width: 100%;
    height: 70%;
    justify-content: space-between;
    align-items: flex-start;
}

/* 添加新的选择器来设置字体大小 */
.knowledge-title {
    font-size: 20px;
    font-weight: bold;
    /* margin-bottom: 5px; */
}


.left-knowledge-bottom {
    width: 100%;
    height: 30%;
    padding-left: 5px;
}

.right-knowledge {
    width: 40%;
    height: 100%;
    padding-left: 15px;
}

.image-container {
    position: relative;
    width: 100%;
    height: 80%;
    animation: fadeInAnimation ease 1s;
    /* margin-top: 5px; */
    animation-iteration-count: 1;
    animation-fill-mode: forwards;
}

.news-image {
    width: 100%;
    height: 100%;
    object-fit: fill;
}

.image-link {
    position: absolute;
    height: 10%;
    width: 100%;
    bottom: 0px;
    left: 0px;
    color: #fff;
    background-color: rgba(0, 0, 0, 0.5);
    text-decoration: none;
    font-size: 15px;
    display: flex;
    align-items: center;
    /* justify-content: center; */
}

.image-link span {
    padding-left: 10px;
}

.image-link:hover {
    color: #62a8ef;
}

@keyframes fadeInAnimation {
    0% {
        opacity: 0;
    }

    100% {
        opacity: 1;
    }
}

.fraud-resources {
    /* list-style-type: none; */
    padding-left: 10px;
}

.fraud-resources li {
    margin-bottom: 5px;
}

.fraud-resources a {
    color: #116FCD;
    text-decoration: none;
    font-size: 16px;
}

.fraud-resources a:hover {
    text-decoration: underline;
}

.fraud-phrases {
    list-style-type: disc;
    padding-left: 20px;
}

.fraud-phrases li {
    margin-bottom: 10px;
}

.all-content {
    width: 70%;
    height: 100%;
    flex-direction: column;
    align-items: center;
    display: flex;
    /* justify-content: center; */
    /* justify-content: space-around; */
}

.content-wrapper {
    width: 95%;
    height: 72%;
    padding-left: 10px;
    padding-right: 10px;
    overflow: hidden;
    padding-top: 30px;
    justify-content: space-around;
    display: flex;
}

.left-content {
    width: 20%;
    height: 95%;
    text-align: center;
    align-items: flex;
    justify-content: flex;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.1);
    margin-right: 10px;
    margin-left: 10px;
    border-radius: 8px;
}

.mid-content {
    width: 40%;
    height: 95%;
    text-align: center;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.1);
    margin-right: 10px;
    margin-left: 10px;
    border-radius: 8px;
}

.right-content {
    width: 40%;
    height: 95%;
    text-align: center;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.1);
    margin-left: 10px;
    margin-right: 10px;
    border-radius: 8px;
}

.timeline-chart {
    width: 90%;
    height: 90%;
}

.chart-content {
    width: 93%;
    height: 20%;
    text-align: center;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.1);
    justify-content: space-around;
    padding-top: 10px;
    display: flex;
    overflow: hidden;
    border-radius: 8px;
}

.container {
    width: 100%;
    height: 100%;
    display: flex;
    /* 使用 Flexbox 布局 */
    flex-direction: row;
    /* 子元素水平排列 */
    justify-content: space-between;
    /* 子元素之间有间距 */
}

.robot-chat {
    width: 30%;
    /* 可根据需要调整宽度 */
    height: min(100%, 928);
    /* 高度占据整个页面 */
    background-color: #fff;
    /* 背景颜色 */
    /* border-left: 7px solid #1567bb; 添加边框，颜色为 #1567bb */
    /* border-radius: 8px; 圆角 */
    box-shadow: -4px 4px 8px rgba(0, 0, 0, 0.2);
    /* 阴影效果 */
    /* overflow-y: auto; */
    /* 内容超出时显示滚动条 */
    padding: 20px;
    /* 内边距 */
}

/* 标题样式 */
.robot-chat .chat-title {
    font-size: 24px;
    font-weight: bold;
    color: #333;
    margin-bottom: 20px;
    text-align: center;
    letter-spacing: 1px;
}


/* 模拟与大语言模型交流的界面 */
.robot-chat .chat-box {
    width: 100%;
    height: 83%;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
    /* 每个消息之间的间距 */
}

.robot-chat .message {
    padding: 10px;
    border-radius: 8px;
    max-width: 70%;
    /* 消息的最大宽度 */
}

.robot-chat .message.user {
    background-color: #e0e0e0;
    /* 用户消息的颜色 */
    align-self: flex-end;
    /* 用户消息靠右对齐 */
}

.robot-chat .message.bot {
    background-color: #d0e6ff;
    /* 机器人消息的颜色 */
    align-self: flex-start;
    /* 机器人消息靠左对齐 */
}

/* 输入框和发送按钮样式 */
.robot-chat .chat-input {
    display: flex;
    margin-top: 20px;
}

.robot-chat .chat-input textarea {
    flex: 1;
    padding: 10px;
    border: 2px solid #ccc;
    border-radius: 4px;
    font-size: 16px;
    resize: none;
    /* 禁止手动调整大小 */
    min-height: 50px;
    /* 最小高度 */
    max-height: 150px;
    /* 最大高度 */
    overflow-y: auto;
    /* 超出最大高度时显示滚动条 */
    white-space: pre-wrap;
    /* 自动换行 */
    word-wrap: break-word;
    /* 长单词自动换行 */
}

.robot-chat .chat-input button {
    padding: 10px 20px;
    margin-left: 10px;
    background-color: #1567bb;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
}

.robot-chat .chat-input button:hover {
    background-color: #2a82db;
}

.data-fruad {
    width: 100%;
    height: 4%;
}

.data-title-text {
    text-align: center;
    font-size: 18px;
    color: #333;
    display: center;
    justify-content: center;
}

.project-overview {
    width: 100%;
    color: #fff;
    padding: 0 0;
    font-size: 3vh;
}

.overview-title {
    height: 100%;
    padding: 7px;
    display: block;
    border-top: 1px #ccc solid;
    color: #333;
    font-size: 2vh;
    font-size: 18px;
}

.info-text {
    font-size: 2.5vh;
    color: white;
    white-space: pre-line;
}

.video-block {

    width: 100%;
    margin-top: 2vh;
    height: 20%;
    display: flex;
}

.video-display {
    height: 100%;
    width: 30%;
    display: flex;
    margin-left: 3%;
    /* float: left; */

    /* flex-direction: column;  */
}

.video-display img {
    width: 100%;
    height: 20%;
    margin-top: 10%;
    object-fit: contain;
}

.left-warn-content {
    width: 47%;
    height: 100%;
    font-size: 17px;
    line-height: 1.5;
    letter-spacing: 1px;
    text-indent: 2em;
    margin-right: 10px;
    /* 添加首行缩进 */
}

.right-warn-content {
    width: 47%;
    height: 100%;
}

.right-top-warn-content {
    width: 100%;
    height: 50%;
    font-size: 16px;
    line-height: 1.5;
}

.right-bottom-warn-content {
    width: 100%;
    height: 50%;
    font-size: 16px;
    line-height: 1.4;
    letter-spacing: 1px;
}

.warn-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 8px;
}

.custom-ul{
    padding-left:20px;
}

</style>