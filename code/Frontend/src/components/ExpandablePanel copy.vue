<template>
    <div>
        <el-button  :disabled="isReportButtonDisabled" type="info" class="icon-button" @click="togglePanel" round>检测报告</el-button>
        <!-- v-if="isPanelVisible" -->
        <!-- :style="{ width: panelWidth + 'px', height: panelHeight + 'px' }" -->
        <div  class="overlay">
            <div class="panel" :style="{ width: panelWidth + 'px', height: panelHeight + 'px' }">
                <div class="close-button" @click="togglePanel">×</div>

                <div class="panel-content">
                    <div class="fraud_analysis">
                        <p class="inner_title">风险摘要</p>
                        <p class="inner_doc1">诈骗：{{ displayContent.Confidence.fraud }}</p>
                        <p v-if="displayContent.Confidence.carrier !== -1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 载体：{{ displayContent.Confidence.carrier }}</p>
                        <div class="inner_doc2" v-if="displayContent.Confidence.video !== -1">
                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 视频：{{ displayContent.Confidence.video }}</p>
                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ displayContent.videoText }}</p>
                            <div class="inner_picture">
                                <img
                                    v-for="(url, index) in displayContent.video"
                                    :key="index"
                                    :src="url"
                                    :alt="url"
                                />
                            </div>
                        </div>

                        <div class="inner_doc3" v-if="displayContent.Confidence.sound !== -1">
                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 语音：{{ displayContent.Confidence.sound }}</p>
                            
                            <div class="inner_text">
                                <p
                                    v-for="(text, index) in displayContent.sound"
                                    :key="index"
                                >&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ text }}</p>
                            </div>
                        </div>
                        <p v-if="displayContent.Confidence.action !== -1">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 行为：{{ displayContent.Confidence.action }}</p>
                        <div class="inner_doc2" v-if="displayContent.action.length > 0">

                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 动作：</p>
                            <div class="inner_picture">
                                <img
                                    v-for="(url, index) in displayContent.action"
                                    :key="index"
                                    :src="url"
                                    :alt="url"
                                />
                            </div>
                        </div>

                        <div class="inner_doc2" v-if="displayContent.Confidence.emotion !== -1">
                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 表情：</p>
                            <div class="inner_picture">
                                <p
                                    v-for="(text, index) in displayContent.emotion"
                                    :key="index"
                                >&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ text }}</p>
                            </div>
                        </div>

                        <div class="inner_doc3" v-if="displayContent.Confidence.soundEmotion !== -1">
                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 语音情绪：</p>
                            <div class="inner_text">
                                <p
                                    v-for="(text, index) in displayContent.soundEmotion"
                                    :key="index"
                                >&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ text }}</p>
                            </div>
                        </div>

                        <div class="inner_doc2" v-if="displayContent.Confidence.content  !== -1">
                            <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;内容：{{ displayContent.Confidence.content }}</p>
                            <p style="white-space:pre-line;">{{ displayContent.text }}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            isPanelVisible: false,
            panelWidth: 1200,
            panelHeight: 800
        };
    },
    props: {
        displayContent: {},
        isReportButtonDisabled: {
            type: Boolean,
            default: true
        }
    },
    methods: {
        togglePanel() {
            this.isPanelVisible = !this.isPanelVisible;
        },
    },
};
</script>

<style scoped>
/* 样式保持不变 */
.icon-button {
    position: fixed;
    top: 50%;
    right: 20px;
}

.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.panel {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: relative;
    overflow: hidden;
}

.panel-content {
    height: 100%;
    overflow-y: auto;
    padding-right: 10px;
}

.close-button {
    position: absolute;
    top: 10px;
    right: 10px;
    cursor: pointer;
    font-size: 20px;
    color: #333;
}

.fraud_analysis {
    height: 200%;
}

.inner_title {
    font-size: 5vh;
    font-weight: bold;
    height: 5%;
    text-align: center;
    border-bottom: 1px solid gray;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.inner_doc1 {
    font-size: 3vh;
    font-weight: bold;
    height: 3%;
    text-align: center;
    border-bottom: 1px solid gray;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.inner_doc3 {
    border-bottom: 1px solid gray;
}

.inner_doc2 p {
    font-size: 2vh;
    font-weight: bold;
    margin: 0;
}
.inner_doc3 p {
    font-size: 2vh;
    font-weight: bold;
    margin: 0;
}

.inner_picture {
    width: 100%;
    height: auto;
}

.inner_picture img {
    width: 30%;
    height: auto;
    margin-left: 2.5%;
}
</style>
