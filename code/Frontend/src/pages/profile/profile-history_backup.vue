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
                                            <a href="https://www.thepaper.cn/newsDetail_forward_25109331" target="_blank">
                                                国家反诈中心公布10类高发诈骗类型
                                            </a>
                                        </li>
                                        <li>
                                            <a href="https://www.bilibili.com/video/BV1R64y1e711/?vd_source=724e3e100d4ba655efa15c8b199fdc70" target="_blank">
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
                            这是一个长的模块
                        </div>
                    </div>
                </div>
            </div>
            <div v-else class="right-module">
                nihao
            </div>
        </div>
    </div>
</template>

<script>
import axios from '../../axios'; // 使用自定义的 axios 实例
import * as echarts from 'echarts';

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
        }
    },
    watch: {
        showCustomOption(newVal) {
            if (newVal && this.activeItem === 'custom-option') {
                this.$nextTick(() => {
                    this.initChart(); // 当显示自定义选项时重新初始化图表
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
    height: 12%;
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
.knowledge-title{
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
</style>