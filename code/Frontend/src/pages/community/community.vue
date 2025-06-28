<template>
    <div class="join">
        <div class="content-container">
            <!-- 左边模块框 -->
            <div class="left-content">
                <div class="title-bar">
                    社区动态
                    <el-select v-model="selectedType" placeholder="请选择动态类型" class="type-select">
                        <el-option v-for="type in typeOptions" :key="type" :label="type" :value="type" />
                    </el-select>
                </div>
                <div class="post-container">
                    <!-- 动态列表 -->
                    <div v-for="(post, index) in filteredPosts" :key="index" class="post">
                        <div class="post-header">
                            <img :src=post.avatar alt="User Avatar" class="post-avatar">
                            <div class="post-info">
                                <div class="post-nickname">{{ post.nickname }}</div>
                                <div class="post-timestamp">{{ post.timestamp }}</div>
                            </div>
                        </div>
                        <div class="post-content">
                            {{ post.content }}
                        </div>
                        <div class="post-actions">
                            <button class="action-button like-button">点赞</button>
                            <button class="action-button comment-button">评论</button>
                            <span class="post-type">{{ post.type }}</span>
                        </div>
                    </div>
                </div>
                <div class="input-container">
                    <input v-model="newPost" type="text" placeholder="友善发言，共享和谐社区...">
                    <button @click="addPost">发送</button>
                </div>
            </div>

            <!-- 右边模块框 -->
            <div class="right-content">
                <div class="notes notes-first">
                    <div class="title-bar">防诈知识问答</div>
                    <div class="quiz-container">
                        <div class="quiz-content">
                            <div class="question">
                                {{ currentQuestion.question }}
                            </div>
                            <div class="options">
                                <div v-for="(option, index) in currentQuestion.options" :key="index" class="option">
                                    <input type="radio" :id="'option' + index" :value="option" v-model="selectedOption">
                                    <label :for="'option' + index">{{ option }}</label>
                                </div>
                            </div>
                        </div>
                        <div class="quiz-explanation" v-if="showExplanation">
                            <div class="explanation-title">本题解析</div>
                            <div class="explanation-content">
                                {{ currentQuestion.explanation }}
                            </div>
                        </div>
                    </div>
                    <div class="quiz-actions">
                        <button class="action-button view-explanation" @click="showExplanation = true">查看解析</button>
                        <button class="action-button next-question" @click="nextQuestion">下一题</button>
                    </div>
                </div>
                <div class="notes notes-second">
                    <div class="title-bar">近期高频诈骗案例</div>
                    <div class="warn-container">
                        <div class="left-warn-content">
                            <div class="warn-title">
                                案例分析🔍
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
    </div>
</template>

<script>
import NERTC from 'nertc-web-sdk';
// import * as echarts from 'echarts';

export default {
    name: 'VideoChatJoin',
    data() {
        return {
            posts: [
                {
                    avatar: "/src/assets/img/avatar_1.jpg",
                    nickname: '云淡风清',
                    content: '我在网上找房子租，看到一个房东发的视频，感觉很真实，对方还主动开视频通话，确实是跟房东发布的照片长得一样，说话也很自然。我当时没多想，就按要求交了2000元押金。结果第二天再联系，发现人已经消失了！后来我仔细想了一下，感觉哪里不对劲，查了一些资料，才知道骗子可能是用了deepfake换脸技术，把别人发布的房东视频替换成了自己的口型，加上AI语音合成，伪造了这次视频通话！以后再遇到这种情况，一定要亲自去实地看房，千万别相信网络上的视频！',
                    timestamp: '2025-2-05 8:32',
                    type: '案例分享'
                },
                {
                    avatar: "/src/assets/img/avatar_2.jpeg",
                    nickname: '(*/ω＼*)暖青',
                    content: '前几天，我妈接到一个电话，对方自称是我，说自己出车祸了，急需5万元医药费，让她立刻转账！我妈当时都吓哭了，连我爸都信了，差点就转账了。还好我妹妹机智，直接打了我的手机，发现我根本没事！后来我们回拨那个电话，发现已经打不通了。我上网查了才知道，这些骗子用了AI语音合成技术，可以模仿任何人的声音，简直太可怕了！大家一定要告诉家里人，不管电话里听到什么，一定要多方确认，千万别直接转账！',
                    timestamp: '2025-3-02 11:29',
                    type: '案例分享'
                },
                {
                    avatar: "/src/assets/img/avatar_3.jpg",
                    nickname: '海阔天空',
                    content: '平台的诈骗案例分析很接地气，特别是那个关于“解冻费”的案例，我身边的朋友就遇到过类似的情况。希望这类内容能继续更新，帮助更多人提高警惕！',
                    timestamp: '2025-1-02 17:08',
                    type: '产品评论'
                },
                {
                    avatar: "/src/assets/img/avatar_4.jpg",
                    nickname: 'echo hello world',
                    content: '作为一个经常网购的人，平台的安全建议对我帮助很大。特别是提醒不要随意点击陌生链接，这让我在收到可疑短信时更加谨慎。感谢平台的用心！',
                    timestamp: '2025-1-02 17:08',
                    type: '产品评论'
                },
                // 更多动态...
            ],
            selectedType: '全部', // 默认选中“全部”
            typeOptions: ['全部', '知识分享', '案例分享', '产品评论'], // 动态类型选项
            newPost: '',
            questions: [
                {
                    question: '以下哪种行为最容易被网络诈骗分子利用？',
                    options: [
                        'A. 在社交平台上公开分享个人旅行计划',
                        'B. 在购物网站上使用信用卡支付',
                        'C. 在公共Wi-Fi环境下登录银行账户',
                        'D. 在收到陌生短信后点击链接并填写个人信息'
                    ],
                    explanation: 'D选项最容易被网络诈骗分子利用。点击陌生链接并填写个人信息可能导致账户被盗或资金损失。'
                },
                {
                    question: '如果你收到一条自称是银行客服的短信，要求你点击链接更新账户信息，你应该怎么做？',
                    options: [
                        'A. 立即点击链接并更新信息',
                        'B. 忽略短信，直接删除',
                        'C. 通过官方渠道联系银行核实信息',
                        'D. 将短信转发给朋友询问建议'
                    ],
                    explanation: 'C选项是最安全的做法。通过官方渠道联系银行核实信息可以避免落入诈骗陷阱。'
                },
                {
                    question: '以下哪种情况可能是网络诈骗的征兆？',
                    options: [
                        'A. 收到一封来自陌生人的邮件，声称你中了大奖',
                        'B. 在购物网站上看到一件商品价格远低于市场价',
                        'C. 在社交媒体上收到一条私信，要求你提供验证码',
                        'D. 以上都是'
                    ],
                    explanation: 'D选项正确。以上情况都可能是网络诈骗的征兆，需要提高警惕。'
                },
                {
                    question: '如果你在网购时发现商家要求你通过私人转账支付，你应该怎么做？',
                    options: [
                        'A. 按照商家要求进行转账',
                        'B. 拒绝支付并取消订单',
                        'C. 联系平台客服核实商家信息',
                        'D. 忽略商家要求，继续使用平台支付'
                    ],
                    explanation: 'C选项是最佳做法。联系平台客服核实商家信息可以避免被骗。'
                },
                {
                    question: '以下哪种行为可以有效防范网络诈骗？',
                    options: [
                        'A. 定期更换密码并使用强密码',
                        'B. 在多个网站使用相同的密码',
                        'C. 随意点击邮件中的链接',
                        'D. 在公共场合大声讨论个人财务信息'
                    ],
                    explanation: 'A选项是最佳做法。定期更换密码并使用强密码可以有效防范网络诈骗。'
                },
                {
                    question: '如果你在社交媒体上收到一条私信，声称你的账户存在安全问题，要求你点击链接进行验证，你应该怎么做？',
                    options: [
                        'A. 立即点击链接进行验证',
                        'B. 忽略私信，直接删除',
                        'C. 通过官方渠道联系平台客服核实信息',
                        'D. 将私信转发给朋友询问建议'
                    ],
                    explanation: 'C选项是最安全的做法。通过官方渠道联系平台客服核实信息可以避免落入诈骗陷阱。'
                },
                {
                    question: '以下哪种情况可能是钓鱼网站的征兆？',
                    options: [
                        'A. 网站URL与官方网站相似但略有不同',
                        'B. 网站要求你提供敏感信息，如密码或验证码',
                        'C. 网站页面设计粗糙，存在大量拼写错误',
                        'D. 以上都是'
                    ],
                    explanation: 'D选项正确。以上情况都可能是钓鱼网站的征兆，需要提高警惕。'
                },
                {
                    question: '如果你在网购时发现商家要求你通过私人转账支付，你应该怎么做？',
                    options: [
                        'A. 按照商家要求进行转账',
                        'B. 拒绝支付并取消订单',
                        'C. 联系平台客服核实商家信息',
                        'D. 忽略商家要求，继续使用平台支付'
                    ],
                    explanation: 'C选项是最佳做法。联系平台客服核实商家信息可以避免被骗。'
                },
                {
                    question: '以下哪种行为可以有效防范网络诈骗？',
                    options: [
                        'A. 定期更换密码并使用强密码',
                        'B. 在多个网站使用相同的密码',
                        'C. 随意点击邮件中的链接',
                        'D. 在公共场合大声讨论个人财务信息'
                    ],
                    explanation: 'A选项是最佳做法。定期更换密码并使用强密码可以有效防范网络诈骗。'
                },
                {
                    question: '如果你在社交媒体上收到一条私信，声称你的账户存在安全问题，要求你点击链接进行验证，你应该怎么做？',
                    options: [
                        'A. 立即点击链接进行验证',
                        'B. 忽略私信，直接删除',
                        'C. 通过官方渠道联系平台客服核实信息',
                        'D. 将私信转发给朋友询问建议'
                    ],
                    explanation: 'C选项是最安全的做法。通过官方渠道联系平台客服核实信息可以避免落入诈骗陷阱。'
                },
                {
                    question: '以下哪种情况可能是钓鱼网站的征兆？',
                    options: [
                        'A. 网站URL与官方网站相似但略有不同',
                        'B. 网站要求你提供敏感信息，如密码或验证码',
                        'C. 网站页面设计粗糙，存在大量拼写错误',
                        'D. 以上都是'
                    ],
                    explanation: 'D选项正确。以上情况都可能是钓鱼网站的征兆，需要提高警惕。'
                },
                {
                    question: '如果你在网购时发现商家要求你通过私人转账支付，你应该怎么做？',
                    options: [
                        'A. 按照商家要求进行转账',
                        'B. 拒绝支付并取消订单',
                        'C. 联系平台客服核实商家信息',
                        'D. 忽略商家要求，继续使用平台支付'
                    ],
                    explanation: 'C选项是最佳做法。联系平台客服核实商家信息可以避免被骗。'
                },
                {
                    question: '以下哪种行为可以有效防范网络诈骗？',
                    options: [
                        'A. 定期更换密码并使用强密码',
                        'B. 在多个网站使用相同的密码',
                        'C. 随意点击邮件中的链接',
                        'D. 在公共场合大声讨论个人财务信息'
                    ],
                    explanation: 'A选项是最佳做法。定期更换密码并使用强密码可以有效防范网络诈骗。'
                },
                {
                    question: '如果你在社交媒体上收到一条私信，声称你的账户存在安全问题，要求你点击链接进行验证，你应该怎么做？',
                    options: [
                        'A. 立即点击链接进行验证',
                        'B. 忽略私信，直接删除',
                        'C. 通过官方渠道联系平台客服核实信息',
                        'D. 将私信转发给朋友询问建议'
                    ],
                    explanation: 'C选项是最安全的做法。通过官方渠道联系平台客服核实信息可以避免落入诈骗陷阱。'
                },
                {
                    question: '以下哪种情况可能是钓鱼网站的征兆？',
                    options: [
                        'A. 网站URL与官方网站相似但略有不同',
                        'B. 网站要求你提供敏感信息，如密码或验证码',
                        'C. 网站页面设计粗糙，存在大量拼写错误',
                        'D. 以上都是'
                    ],
                    explanation: 'D选项正确。以上情况都可能是钓鱼网站的征兆，需要提高警惕。'
                }
                // 更多问题...
            ],
            currentQuestionIndex: 0,
            selectedOption: '',
            showExplanation: false,
        };
    },

    computed: {
        currentQuestion() {
            return this.questions[this.currentQuestionIndex];
        },
        filteredPosts() {
            let filtered = this.posts;
            if (this.selectedType !== '全部') {
                filtered = this.posts.filter(post => post.type === this.selectedType);
            }
            // 按照时间排序
            return filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        }
    },

    watch: {

    },
    mounted() {
        if (!NERTC.checkSystemRequirements()) {
            this.isSupport = false;
        }
    },
    methods: {
        addPost() {
            if (this.newPost.trim() !== '') {
                const newPost = {
                    avatar: "/src/assets/img/fake.jpg", // 默认头像
                    nickname: '你', // 默认昵称
                    content: this.newPost,
                    type: '产品评论',
                    timestamp: new Date().toLocaleString() // 当前时间
                };
                this.posts.push(newPost);
                this.newPost = ''; // 清空输入框
            }
        },
        nextQuestion() {
            this.currentQuestionIndex = (this.currentQuestionIndex + 1) % this.questions.length;
            this.selectedOption = '';
            this.showExplanation = false;
        }
    },
};
</script>

<style scoped>
.join {
    height: 100%;
    display: flex;
    justify-content: center;
    flex-direction: column;
}

.content-container {
    width: 80%;
    height: 80%;
    padding: 40px 20px;
    margin-left: 4%;
    display: flex;
    justify-content: flex;
}

.left-content {
    width: 50%;
    text-align: center;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.2);
    margin-right: 60px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
}

.right-content {
    width: 50%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.notes {
    width: 110%;
    background: #fff;
    box-shadow: 10px 10px 14px 6px rgba(47, 56, 111, 0.2);
    border-radius: 8px;
}

.title-bar {
    background-color: #116FCD;
    color: #fff;
    height: 50px;
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
}

/* 新增的样式 */
.divider {
    margin-top: 2vw;
    margin-bottom: 2vw;
    border-top: 1px solid #ccc;
}

.post-container {
    padding-left: 8px;
    padding-right: 8px;
    height: 80%;
    /* 设置一个固定高度 */
    overflow-y: auto;
    /* 允许垂直滚动 */
    overflow-x: hidden;
}

.post {
    margin-bottom: 15px;
    padding: 10px;
    width: 96%;
    background-color: #f9f9f9;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    /* 确保内容从左侧开始 */
}

.post-header {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}

.post-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    margin-right: 10px;
}

.post-info {
    display: flex;
    flex-direction: column;
}

.post-nickname {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 5px;
}

.post-timestamp {
    font-size: 12px;
    color: #888;
}

.post-content {
    margin-bottom: 10px;
    font-size: 16px;
    word-wrap: break-word;
    /* 确保内容可以自动换行 */
    overflow-wrap: break-word;
    /* 确保内容可以自动换行 */
    text-align: left;
    /* 确保内容左对齐 */
    margin-top: 10px;
    margin-bottom: 15px;
}

.post-actions {
    display: flex;
    justify-content: flex-start;
}

.action-button {
    padding: 5px 10px;
    margin-right: 10px;
    background-color: #116FCD;
    color: #fff;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 12px;
    display: flex;
    /* 添加 */
    align-items: center;
    /* 添加 */
    justify-content: center;
    /* 添加 */
}

.action-button:hover {
    background-color: #0d59b5;
}

.like-button {
    background-color: #c43d60;
}

.like-button:hover {
    background-color: #e6557c;
}

.comment-button {
    background-color: #414656;
}

.comment-button:hover {
    background-color: #6c748f;
}

.input-container {
    display: flex;
    width: 100%;
    justify-content: space-around;
    align-items: center;
    margin-top: 10px;
}

.input-container input {
    width: 70%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
}

.input-container button {
    width: 15%;
    padding: 10px;
    background-color: #116FCD;
    color: #fff;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.input-container button:hover {
    background-color: #0d59b5;
}

.notes-first {
    display: flex;
    flex-direction: column;
    height: 45%;
    /* justify-content: space-around; */
}

.quiz-container {
    display: flex;
    flex-direction: row;
    height: 70%;
    /* margin-bottom: 10px; */
    padding-left: 18px;
    padding-right: 18px;
}

.quiz-content {
    width: 50%;
    display: flex;
    flex-direction: column;
    margin-bottom: 10px;
    line-height: 1.5;
    margin-right: 10px;
}

.question {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}

.options {
    display: flex;
    flex-direction: column;
}

.option {
    margin-bottom: 5px;
}

.option input[type="radio"] {
    margin-right: 10px;
}

.quiz-explanation {
    width: 50%;
    display: flex;
    flex-direction: column;
    /* margin-top: 20px; */
}

.explanation-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}

.explanation-content {
    font-size: 16px;
    color: #555;
}

.quiz-actions {
    height: 10%;
    display: flex;
    justify-content: space-around;
    margin-bottom: 20px;
}

.view-explanation,
.next-question {
    padding: 10px 20px;
    background-color: #116FCD;
    color: #fff;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
}

.view-explanation:hover,
.next-question:hover {
    background-color: #0d59b5;
}

.notes-second {
    height: 50%;
}

.warn-container {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.left-warn-content {
    width: 47%;
    height: 100%;
    font-size: 16px;
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
    height: 45%;
    font-size: 15px;
    line-height: 1.4;
}

.right-bottom-warn-content {
    width: 100%;
    height: 45%;
    font-size: 15px;
    line-height: 1.4;
    letter-spacing: 1px;
}

.warn-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 6px;
    text-align: center;
}

.custom-ul {
    padding-left: 23px;
}

.post-type {
    margin-left: 10px;
    font-size: 14px;
    color: #666;
    padding: 5px 10px;
    background-color: #f0f0f0;
    border-radius: 5px;
}

.type-select {
    margin-left: auto;
    /* 将 select 推到右侧 */
    width: 20%;
    /* 设置宽度 */
}

.type-select ::v-deep .el-select__wrapper {
    height: 80%;
    /* 设置高度 */
    font-size: 14px;
    /* 设置字体大小 */
}
</style>