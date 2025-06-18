<template>
    <div class="app-container">
        <!-- Header -->
        <header class="app-header">
            <div class="header-content">
                <h1 class="app-title">RVC 实时语音转换</h1>
                <div class="header-actions">
                    <div class="connection-status">
                        <span class="status-label">状态:</span>
                        <span
                            :class="['status-indicator', connectionStatus]"
                            :title="lastError || '正常'"
                        >
                            {{ getConnectionStatusText() }}
                        </span>
                    </div>

                    <button
                        class="settings-toggle"
                        @click="toggleSettings"
                        :class="{ active: showSettings }"
                    >
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path
                                d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11.03L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.22,8.95 2.27,9.22 2.46,9.37L4.57,11.03C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.22,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.68 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"
                            />
                        </svg>
                    </button>
                </div>
            </div>
        </header>

        <!-- Notification Area -->
        <div v-if="errorMessage || successMessage" class="notification-area">
            <div
                v-if="errorMessage"
                class="notification error"
                @click="clearError"
            >
                <svg
                    class="notification-icon"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                >
                    <path
                        d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,17A1.5,1.5 0 0,1 10.5,15.5A1.5,1.5 0 0,1 12,14A1.5,1.5 0 0,1 13.5,15.5A1.5,1.5 0 0,1 12,17M12,5.5A3.5,3.5 0 0,1 15.5,9C15.5,10.38 14.75,11.44 13.5,12.28C13.17,12.53 13,12.69 13,13.5H11C11,12.06 11.83,11.21 12.83,10.35C13.28,9.96 13.5,9.5 13.5,9A1.5,1.5 0 0,0 12,7.5A1.5,1.5 0 0,0 10.5,9H9A3,3 0 0,1 12,6A3,3 0 0,1 15,9C15,10.88 13.37,12 12,12.63V13.5H10V12.63C8.63,12 7,10.88 7,9A5,5 0 0,1 12,4A5,5 0 0,1 17,9C17,11.12 15.12,12.75 13.5,13.37V14.5H10.5V13.37C8.88,12.75 7,11.12 7,9A3,3 0 0,1 10,6A3,3 0 0,1 13,9C13,9.5 12.72,9.96 12.27,10.35C11.27,11.21 10.44,12.06 10.44,13.5H12.56C12.56,12.69 12.73,12.53 13.06,12.28C14.31,11.44 15.06,10.38 15.06,9A3.56,3.56 0 0,0 11.5,5.44A3.56,3.56 0 0,0 7.94,9A1.06,1.06 0 0,0 9,10.06C9.59,10.06 10.06,9.59 10.06,9A2,2 0 0,1 12.06,7A2,2 0 0,1 14.06,9C14.06,9.89 13.54,10.64 12.81,11.2C11.77,12 10.94,12.94 10.94,14.5H13.06C13.06,13.66 13.4,13.12 14.06,12.59C15.19,11.73 16.06,10.5 16.06,9A4.06,4.06 0 0,0 12,4.94A4.06,4.06 0 0,0 7.94,9H9.5A2.5,2.5 0 0,1 12,6.5A2.5,2.5 0 0,1 14.5,9C14.5,10.25 13.75,11.31 12.5,12.15C12.17,12.4 12,12.56 12,13.37V14.5H10V13.37C10,12.56 10.17,12.4 10.5,12.15C11.75,11.31 12.5,10.25 12.5,9A2.5,2.5 0 0,0 10,6.5A2.5,2.5 0 0,0 7.5,9H9A1,1 0 0,1 10,8A1,1 0 0,1 11,9A1,1 0 0,1 10,10A1,1 0 0,1 9,9"
                    />
                </svg>
                <span class="notification-text">{{ errorMessage }}</span>
                <button
                    class="notification-copy"
                    @click.stop="copyError"
                    title="复制错误信息"
                >
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path
                            d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"
                        />
                    </svg>
                </button>
                <button class="notification-close" @click.stop="clearError">
                    ×
                </button>
            </div>
            <div
                v-if="successMessage"
                class="notification success"
                @click="clearSuccess"
            >
                <svg
                    class="notification-icon"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                >
                    <path
                        d="M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M11,16.5L18,9.5L16.59,8.09L11,13.67L7.91,10.59L6.5,12L11,16.5Z"
                    />
                </svg>
                <span class="notification-text">{{ successMessage }}</span>
                <button
                    class="notification-copy"
                    @click.stop="copySuccess"
                    title="复制成功信息"
                >
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path
                            d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"
                        />
                    </svg>
                </button>
                <button class="notification-close" @click.stop="clearSuccess">
                    ×
                </button>
            </div>
        </div>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Top Row: Controls + Quick Settings -->
            <div class="top-row">
                <!-- Control Panel -->
                <div class="control-panel">
                    <div class="main-controls">
                        <div class="control-section">
                            <h3>语音转换控制</h3>
                            <div class="status-indicator">
                                <div
                                    class="status-light"
                                    :class="{ active: isRunning }"
                                ></div>
                                <span class="status-text">
                                    {{ isRunning ? "转换中..." : "就绪" }}
                                </span>
                            </div>
                            <div class="control-buttons">
                                <button
                                    class="control-btn start-btn"
                                    @click="startVc"
                                    :disabled="!canStartVc"
                                    :class="{ pulse: isRunning }"
                                >
                                    <svg
                                        viewBox="0 0 24 24"
                                        fill="currentColor"
                                    >
                                        <path
                                            d="M8,5.14V19.14L19,12.14L8,5.14Z"
                                        />
                                    </svg>
                                    {{ isRunning ? "转换中" : "开始转换" }}
                                </button>

                                <button
                                    class="control-btn stop-btn"
                                    @click="stopVc"
                                    :disabled="!canStopVc"
                                >
                                    <svg
                                        viewBox="0 0 24 24"
                                        fill="currentColor"
                                    >
                                        <path d="M18,18H6V6H18V18Z" />
                                    </svg>
                                    停止转换
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="mode-selector">
                        <label class="mode-option">
                            <input
                                type="radio"
                                value="im"
                                v-model="functionMode"
                            />
                            <span>输入监听</span>
                        </label>
                        <label class="mode-option">
                            <input
                                type="radio"
                                value="vc"
                                v-model="functionMode"
                            />
                            <span>输出变声</span>
                        </label>
                    </div>
                </div>

                <!-- Quick Settings -->
                <div class="quick-settings">
                    <div class="quick-setting">
                        <label>响应阈值</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="-60"
                                max="0"
                                v-model.number="threshold"
                                class="slider"
                            />
                            <span class="value">{{ threshold }}dB</span>
                        </div>
                    </div>

                    <div class="quick-setting">
                        <label>音调设置</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="-16"
                                max="16"
                                step="1"
                                v-model.number="pitch"
                                class="slider"
                            />
                            <span class="value">{{ pitch }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bottom Row: Visualizations + Performance -->
            <div class="bottom-row">
                <!-- Visualizations -->
                <div class="visualizations">
                    <div class="viz-header">
                        <h3>音频可视化</h3>
                        <button
                            class="viz-toggle"
                            @click="toggleVisualizations"
                            :class="{ active: showVisualizations }"
                        >
                            {{ showVisualizations ? "收起" : "展开" }}
                        </button>
                    </div>

                    <div v-show="showVisualizations" class="viz-content">
                        <div class="visualizer-grid">
                            <AudioVisualizer
                                title="输入音频"
                                :audio-data="inputAudioData"
                                :sample-rate="sampleRate"
                                primary-color="#ff6b6b"
                                accent-color="#ff8e8e"
                                theme="light"
                                ref="inputVisualizer"
                                @level-change="(level) => (inputVolume = level)"
                                @error="handleVisualizationError"
                            />
                            <AudioVisualizer
                                title="输出音频"
                                :audio-data="outputAudioData"
                                :sample-rate="sampleRate"
                                primary-color="#4ecdc4"
                                accent-color="#7ed8d1"
                                theme="light"
                                ref="outputVisualizer"
                                @level-change="
                                    (level) => (outputVolume = level)
                                "
                                @error="handleVisualizationError"
                            />
                        </div>
                    </div>
                </div>

                <!-- Performance Monitor -->
                <div class="performance-panel">
                    <PerformanceMonitor
                        :inference-time="inferenceTime"
                        :algorithm-latency="algorithmLatency"
                        :buffer-latency="bufferLatency"
                        :total-latency="totalLatency"
                        :is-streaming="isStreaming"
                        @reset="handlePerformanceReset"
                    />
                </div>
            </div>
        </main>

        <!-- Settings Panel -->
        <aside class="settings-panel" :class="{ active: showSettings }">
            <div class="settings-header">
                <h2>设置</h2>
                <button class="close-btn" @click="toggleSettings">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path
                            d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z"
                        />
                    </svg>
                </button>
            </div>

            <!-- Settings Tabs -->
            <div class="settings-tabs">
                <button
                    v-for="tab in settingsTabs"
                    :key="tab.id"
                    class="tab-btn"
                    :class="{ active: activeTab === tab.id }"
                    @click="activeTab = tab.id"
                >
                    {{ tab.label }}
                </button>
            </div>

            <div class="settings-content">
                <!-- Model Settings -->
                <div v-show="activeTab === 'model'" class="settings-section">
                    <h3>模型配置</h3>
                    <div class="form-group">
                        <label>PTH 模型文件</label>
                        <input
                            v-model="pth"
                            type="text"
                            placeholder="请输入模型文件路径 (如: assets/weights/kikiV1.pth)"
                            class="file-input"
                        />
                        <div class="file-hint">
                            常见路径: assets/weights/kikiV1.pth 或完整的绝对路径
                        </div>
                        <div class="path-suggestions">
                            <button
                                @click="pth = 'assets/weights/kikiV1.pth'"
                                class="suggestion-btn"
                                type="button"
                            >
                                使用默认模型
                            </button>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Index 文件</label>
                        <input
                            v-model="index"
                            type="text"
                            placeholder="请输入索引文件路径 (可选)"
                            class="file-input"
                        />
                        <div class="file-hint">
                            可选：输入 .index 文件路径以提高音色相似度，如:
                            logs/kikiV1.index
                        </div>
                        <div class="path-suggestions">
                            <button
                                @click="index = 'logs/kikiV1.index'"
                                class="suggestion-btn"
                                type="button"
                            >
                                使用默认索引
                            </button>
                            <button
                                @click="index = ''"
                                class="suggestion-btn secondary"
                                type="button"
                            >
                                不使用索引
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Audio Settings -->
                <div v-show="activeTab === 'audio'" class="settings-section">
                    <h3>音频设备</h3>
                    <div class="form-group">
                        <button
                            type="button"
                            @click="reloadDevices"
                            class="reload-btn"
                        >
                            重载设备列表
                        </button>
                    </div>

                    <div class="form-group">
                        <label>设备类型</label>
                        <select v-model="hostapi" class="select-input">
                            <option v-for="h in hostapis" :key="h" :value="h">
                                {{ h }}
                            </option>
                        </select>
                        <label class="checkbox-label">
                            <input type="checkbox" v-model="wasapiExclusive" />
                            独占 WASAPI 设备
                        </label>
                    </div>

                    <div class="form-group">
                        <label>输入设备</label>
                        <select v-model="inputDevice" class="select-input">
                            <option
                                v-for="d in inputDevices"
                                :key="d"
                                :value="d"
                            >
                                {{ d }}
                            </option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>输出设备</label>
                        <select v-model="outputDevice" class="select-input">
                            <option
                                v-for="d in outputDevices"
                                :key="d"
                                :value="d"
                            >
                                {{ d }}
                            </option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>采样率设置</label>
                        <div class="radio-group">
                            <label class="radio-label">
                                <input
                                    type="radio"
                                    value="sr_model"
                                    v-model="srType"
                                />
                                使用模型采样率
                            </label>
                            <label class="radio-label">
                                <input
                                    type="radio"
                                    value="sr_device"
                                    v-model="srType"
                                />
                                使用设备采样率
                            </label>
                        </div>
                        <div class="info-text">
                            当前采样率: {{ sampleRate }} Hz
                        </div>
                    </div>
                </div>

                <!-- Processing Settings -->
                <div
                    v-show="activeTab === 'processing'"
                    class="settings-section"
                >
                    <h3>处理参数</h3>

                    <div class="form-group">
                        <label>性别因子</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="-2"
                                max="2"
                                step="0.05"
                                v-model.number="formant"
                                class="slider"
                            />
                            <span class="value">{{ formant.toFixed(2) }}</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Index Rate</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                v-model.number="indexRate"
                                class="slider"
                            />
                            <span class="value">{{
                                indexRate.toFixed(2)
                            }}</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>响度因子</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                v-model.number="rmsMixRate"
                                class="slider"
                            />
                            <span class="value">{{
                                rmsMixRate.toFixed(2)
                            }}</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>音高算法</label>
                        <div class="radio-group">
                            <label
                                v-for="method in f0Methods"
                                :key="method"
                                class="radio-label"
                            >
                                <input
                                    type="radio"
                                    :value="method"
                                    v-model="f0method"
                                />
                                {{ method }}
                            </label>
                        </div>
                    </div>
                </div>

                <!-- Performance Settings -->
                <div
                    v-show="activeTab === 'performance'"
                    class="settings-section"
                >
                    <h3>性能设置</h3>

                    <div class="form-group">
                        <label>采样长度</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="0.02"
                                max="1.5"
                                step="0.01"
                                v-model.number="blockTime"
                                class="slider"
                            />
                            <span class="value"
                                >{{ blockTime.toFixed(2) }}s</span
                            >
                        </div>
                    </div>

                    <div class="form-group">
                        <label>淡入淡出长度</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="0.01"
                                max="0.15"
                                step="0.01"
                                v-model.number="crossfadeLength"
                                class="slider"
                            />
                            <span class="value"
                                >{{ crossfadeLength.toFixed(2) }}s</span
                            >
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Harvest进程数</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="1"
                                max="8"
                                step="1"
                                v-model.number="nCpu"
                                class="slider"
                            />
                            <span class="value">{{ nCpu }}</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>额外推理时长</label>
                        <div class="slider-container">
                            <input
                                type="range"
                                min="0.05"
                                max="5"
                                step="0.01"
                                v-model.number="extraTime"
                                class="slider"
                            />
                            <span class="value"
                                >{{ extraTime.toFixed(2) }}s</span
                            >
                        </div>
                    </div>

                    <div class="form-group">
                        <label>降噪设置</label>
                        <div class="checkbox-group">
                            <label class="checkbox-label">
                                <input type="checkbox" v-model="iNoiseReduce" />
                                输入降噪
                            </label>
                            <label class="checkbox-label">
                                <input type="checkbox" v-model="oNoiseReduce" />
                                输出降噪
                            </label>
                            <label class="checkbox-label">
                                <input type="checkbox" v-model="usePv" />
                                启用相位声码器
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </aside>

        <!-- Overlay -->
        <div v-if="showSettings" class="overlay" @click="toggleSettings"></div>
    </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
// import { open } from "@tauri-apps/plugin-dialog";
import AudioVisualizer from "./components/AudioVisualizer.vue";
import PerformanceMonitor from "./components/PerformanceMonitor.vue";
import { useAudioStream } from "./composables/useAudioStream.js";

// UI State
const showSettings = ref(false);
const showVisualizations = ref(false);
const activeTab = ref("model");

const settingsTabs = [
    { id: "model", label: "模型" },
    { id: "audio", label: "音频" },
    { id: "processing", label: "处理" },
    { id: "performance", label: "性能" },
];

const f0Methods = ["pm", "harvest", "crepe", "rmvpe", "fcpe"];

// Device and Model State
const hostapis = ref([]);
const inputDevices = ref([]);
const outputDevices = ref([]);
const pth = ref("");
const index = ref("");
const hostapi = ref("");
const wasapiExclusive = ref(false);
const inputDevice = ref("");
const outputDevice = ref("");
const srType = ref("sr_model");

// Processing Parameters
const threshold = ref(-60);
const pitch = ref(0);
const formant = ref(0.0);
const indexRate = ref(0.75);
const rmsMixRate = ref(0.25);
const f0method = ref("fcpe");

// Performance Parameters
const blockTime = ref(0.25);
const crossfadeLength = ref(0.05);
const nCpu = ref(2);
const extraTime = ref(2.5);
const iNoiseReduce = ref(true);
const oNoiseReduce = ref(true);
const usePv = ref(false);
const functionMode = ref("vc");
const sampleRate = ref(0);
const delayTime = ref(0);
const inferTime = ref(0);

// Error and success messages
const errorMessage = ref("");
const successMessage = ref("");

// Audio stream composable
const {
    inputAudioData,
    outputAudioData,
    isStreaming,
    inputVolume,
    outputVolume,
    stats,
    lastError,
    connectionStatus,
    inferenceTime,
    algorithmLatency,
    bufferLatency,
    totalLatency,
    initializeAudioStream,
    stopAudioStream,
    clearBuffers,
} = useAudioStream();

// Voice conversion state
const isRunning = ref(false);
const statusCheckInterval = ref(null);

// Audio visualizer component refs
const inputVisualizer = ref(null);
const outputVisualizer = ref(null);

// Computed properties
const canStartVc = computed(() => {
    return (
        !isRunning.value &&
        pth.value &&
        hostapi.value &&
        inputDevice.value &&
        outputDevice.value
    );
});

const canStopVc = computed(() => {
    return isRunning.value;
});

// Methods
function toggleSettings() {
    showSettings.value = !showSettings.value;
}

function toggleVisualizations() {
    showVisualizations.value = !showVisualizations.value;
}

function send(event, value) {
    try {
        console.log("📤 Sending event:", event, "with value:", value);
        invoke("event_handler", {
            event,
            value: value?.toString() || "",
        }).catch((e) => {
            console.error("❌ Failed to send event:", event, e);
        });
    } catch (e) {
        console.error("❌ Error preparing event:", event, e);
    }
}

async function fetchDevices() {
    try {
        console.log("🔍 Fetching device information...");
        const list = await invoke("get_device_info", {
            hostapi: hostapi.value,
        });

        if (!list || typeof list !== "object") {
            throw new Error("Invalid device information received");
        }

        hostapis.value = list.hostapis || [];
        inputDevices.value = list.input_devices || [];
        outputDevices.value = list.output_devices || [];

        // Validate and fallback for hostapi
        if (!hostapis.value.includes(hostapi.value)) {
            hostapi.value = hostapis.value[0] || "";
            console.log("🔧 Hostapi reset to:", hostapi.value);
        }

        // Validate and fallback for input device
        if (!inputDevices.value.includes(inputDevice.value)) {
            inputDevice.value = inputDevices.value[0] || "";
            console.log("🎤 Input device reset to:", inputDevice.value);
        }

        // Validate and fallback for output device
        if (!outputDevices.value.includes(outputDevice.value)) {
            outputDevice.value = outputDevices.value[0] || "";
            console.log("🔊 Output device reset to:", outputDevice.value);
        }

        console.log("✅ Device information updated successfully");
    } catch (e) {
        console.error("❌ Failed to fetch devices:", e);
        showError(`获取设备信息失败: ${e.message || e}`);

        // Provide fallback empty arrays to prevent UI errors
        hostapis.value = [];
        inputDevices.value = [];
        outputDevices.value = [];
    }
}

async function applyDevices() {
    try {
        console.log("🔧 Applying device settings...", {
            hostapi: hostapi.value,
            inputDevice: inputDevice.value,
            outputDevice: outputDevice.value,
        });

        const result = await invoke("set_audio_devices", {
            hostapi: hostapi.value,
            inputDevice: inputDevice.value,
            outputDevice: outputDevice.value,
        });

        if (typeof result === "number" && result > 0) {
            sampleRate.value = result;
            console.log(
                "✅ Audio devices applied successfully, sample rate:",
                result,
            );
        } else {
            console.warn("⚠️ Unexpected sample rate result:", result);
            sampleRate.value = 44100; // fallback
        }
    } catch (e) {
        console.error("❌ Failed to apply devices:", e);
        showError(`应用设备设置失败: ${e.message || e}`);
        sampleRate.value = 44100; // fallback
    }
}

async function loadConfig() {
    try {
        const config = await invoke("get_init_config");

        // Apply loaded configuration with better defaults
        pth.value = config.pth_path || "";
        index.value = config.index_path || "";
        hostapi.value = config.sg_hostapi || "";
        wasapiExclusive.value = config.sg_wasapi_exclusive || false;
        inputDevice.value = config.sg_input_device || "";
        outputDevice.value = config.sg_output_device || "";
        srType.value = config.sr_type || "sr_model";

        // Processing parameters with better defaults
        threshold.value = config.threshold ?? -60;
        pitch.value = config.pitch ?? 0;
        formant.value = config.formant ?? 0.0;
        indexRate.value = config.index_rate ?? 0.75;
        rmsMixRate.value = config.rms_mix_rate ?? 0.25;
        f0method.value = config.f0method || "fcpe";

        // Performance parameters with better defaults
        blockTime.value = config.block_time ?? 0.25;
        crossfadeLength.value = config.crossfade_length ?? 0.05;
        nCpu.value = config.n_cpu ?? 2;
        extraTime.value = config.extra_time ?? 2.5;
        iNoiseReduce.value = config.i_noise_reduce ?? true;
        oNoiseReduce.value = config.o_noise_reduce ?? true;
        usePv.value = config.use_pv ?? false;
        functionMode.value = "vc"; // not stored in backend config

        console.log("✅ Configuration loaded successfully");
        showSuccess("配置加载成功");
    } catch (e) {
        console.error("❌ Failed to load config:", e);
        showError(`配置加载失败: ${e.message || e}`);
        // Still apply defaults even if loading fails
        console.log("📋 Applying default configuration...");
    }
}

async function reloadDevices() {
    await fetchDevices();
    await applyDevices();
}

async function startVc() {
    if (!canStartVc.value) return;

    try {
        console.log("🚀 Starting voice conversion...");

        // Validate required files
        if (!pth.value) {
            showError("请先输入模型文件路径 (.pth)");
            return;
        }

        // Validate file extensions
        if (!validateModelPath() || !validateIndexPath()) {
            return;
        }

        console.log("📋 Configuration:", {
            pth: pth.value,
            index: index.value,
            hostapi: hostapi.value,
            inputDevice: inputDevice.value,
            outputDevice: outputDevice.value,
            f0method: f0method.value,
            pitch: pitch.value,
            indexRate: indexRate.value,
        });

        // Save current configuration before starting
        await saveConfig();

        await invoke("start_voice_conversion", {
            pth: pth.value,
            index: index.value,
            hostapi: hostapi.value,
            inputDevice: inputDevice.value,
            outputDevice: outputDevice.value,
            wasapiExclusive: wasapiExclusive.value,
            srType: srType.value,
            threshold: threshold.value,
            pitch: pitch.value,
            formant: formant.value,
            indexRate: indexRate.value,
            rmsMixRate: rmsMixRate.value,
            f0method: f0method.value,
            blockTime: blockTime.value,
            crossfadeLength: crossfadeLength.value,
            nCpu: nCpu.value,
            extraTime: extraTime.value,
            iNoiseReduce: iNoiseReduce.value,
            oNoiseReduce: oNoiseReduce.value,
            usePv: usePv.value,
        });

        // Initialize audio stream first
        try {
            await initializeAudioStream();
            console.log("🎧 Audio stream initialized successfully");
        } catch (streamError) {
            console.warn(
                "⚠️ Audio stream initialization failed, continuing without visualization:",
                streamError,
            );
        }

        // Start status checking
        startStatusChecking();

        console.log("✅ Voice conversion started successfully");
        showSuccess("语音转换启动成功");
    } catch (e) {
        console.error("❌ Failed to start voice conversion:", e);
        showError(`启动语音转换失败: ${e.message || e}`);
        // Check status after error to sync state
        await checkStatus();
    }
}

async function stopVc() {
    try {
        console.log("🛑 Stopping voice conversion...");

        // Stop audio stream first
        try {
            await stopAudioStream();
            console.log("🎧 Audio stream stopped");
        } catch (streamError) {
            console.warn("⚠️ Error stopping audio stream:", streamError);
        }

        // Stop voice conversion
        await invoke("stop_voice_conversion");

        // Clean up
        clearBuffers();
        stopStatusChecking();

        console.log("✅ Voice conversion stopped successfully");
        showSuccess("语音转换已停止");
    } catch (e) {
        console.error("❌ Failed to stop voice conversion:", e);
        showError(`停止语音转换失败: ${e.message || e}`);
    } finally {
        // Always check status after stop attempt
        setTimeout(async () => {
            await checkStatus();
        }, 500);
    }
}

// Save configuration function
async function saveConfig() {
    try {
        console.log("💾 Saving configuration...");
        const config = {
            pth_path: pth.value || "",
            index_path: index.value || "",
            sg_hostapi: hostapi.value || "",
            sg_wasapi_exclusive: wasapiExclusive.value || false,
            sg_input_device: inputDevice.value || "",
            sg_output_device: outputDevice.value || "",
            sr_type: srType.value || "sr_model",
            threhold: threshold.value || -40,
            pitch: pitch.value || 0,
            formant: formant.value || 0,
            index_rate: indexRate.value || 0.5,
            rms_mix_rate: rmsMixRate.value || 0.25,
            block_time: blockTime.value || 0.25,
            crossfade_length: crossfadeLength.value || 0.04,
            extra_time: extraTime.value || 2.5,
            n_cpu: nCpu.value || 4,
            f0method: f0method.value || "rmvpe",
            i_noise_reduce: iNoiseReduce.value || false,
            o_noise_reduce: oNoiseReduce.value || false,
            use_pv: usePv.value || false,
        };

        await invoke("save_config", { config });
        console.log("✅ Configuration saved successfully");
    } catch (e) {
        console.error("❌ Failed to save configuration:", e);
        showError(`保存配置失败: ${e.message || e}`);
    }
}

function handlePerformanceReset() {
    // Reset performance metrics
    clearBuffers();
}

function getConnectionStatusText() {
    switch (connectionStatus.value) {
        case "connected":
            return "已连接";
        case "connecting":
            return "连接中";
        case "disconnected":
            return "已断开";
        case "error":
            return "错误";
        default:
            return "未知";
    }
}

function handleVisualizationError(error) {
    console.error("Visualization error:", error);
}

function clearError() {
    lastError.value = null;
    errorMessage.value = "";
}

function clearSuccess() {
    successMessage.value = "";
}

// Startup diagnostics function
async function runStartupDiagnostics() {
    try {
        console.log("🔧 Running startup diagnostics...");

        // Test basic API connectivity
        try {
            const status = await invoke("get_vc_status");
            console.log("✅ API connectivity test passed:", status);
        } catch (e) {
            console.warn("⚠️ API connectivity test failed:", e);
        }

        // Validate device information
        if (hostapis.value.length === 0) {
            console.warn("⚠️ No host APIs detected");
        } else {
            console.log("✅ Host APIs available:", hostapis.value.length);
        }

        if (inputDevices.value.length === 0) {
            console.warn("⚠️ No input devices detected");
        } else {
            console.log(
                "✅ Input devices available:",
                inputDevices.value.length,
            );
        }

        if (outputDevices.value.length === 0) {
            console.warn("⚠️ No output devices detected");
        } else {
            console.log(
                "✅ Output devices available:",
                outputDevices.value.length,
            );
        }

        // Check configuration integrity
        if (!pth.value && !index.value) {
            console.log(
                "ℹ️ No model files configured - this is normal for first run",
            );
        } else {
            console.log("✅ Model configuration present");
        }

        // Test audio stream composable
        if (
            typeof initializeAudioStream === "function" &&
            typeof stopAudioStream === "function"
        ) {
            console.log("✅ Audio stream functions available");
        } else {
            console.error("❌ Audio stream functions missing");
        }

        console.log("🎯 Startup diagnostics completed");
    } catch (error) {
        console.error("❌ Startup diagnostics failed:", error);
    }
}

async function copyError() {
    try {
        await navigator.clipboard.writeText(errorMessage.value);
        showSuccess("错误信息已复制到剪贴板");
    } catch (err) {
        console.error("复制失败:", err);
        // 降级方案：使用传统的复制方法
        try {
            const textArea = document.createElement("textarea");
            textArea.value = errorMessage.value;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);
            showSuccess("错误信息已复制到剪贴板");
        } catch (fallbackErr) {
            console.error("降级复制也失败:", fallbackErr);
            showError("复制失败，请手动复制错误信息");
        }
    }
}

async function copySuccess() {
    try {
        await navigator.clipboard.writeText(successMessage.value);
        console.log("成功信息已复制到剪贴板");
    } catch (err) {
        console.error("复制失败:", err);
        // 降级方案：使用传统的复制方法
        try {
            const textArea = document.createElement("textarea");
            textArea.value = successMessage.value;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);
            console.log("成功信息已复制到剪贴板");
        } catch (fallbackErr) {
            console.error("降级复制也失败:", fallbackErr);
        }
    }
}

function showError(message) {
    errorMessage.value = message;
    console.error("❌", message);
    // Auto-clear after 10 seconds
    setTimeout(() => {
        if (errorMessage.value === message) {
            clearError();
        }
    }, 10000);
}

// Simplified file path input - users can manually enter paths
function validateModelPath() {
    if (pth.value && !pth.value.toLowerCase().endsWith(".pth")) {
        showError("模型文件必须是 .pth 格式");
        return false;
    }
    return true;
}

function validateIndexPath() {
    if (index.value && !index.value.toLowerCase().endsWith(".index")) {
        showError("索引文件必须是 .index 格式");
        return false;
    }
    return true;
}

function showSuccess(message) {
    successMessage.value = message;
    console.log("✅", message);
    // Auto-clear after 3 seconds
    setTimeout(() => {
        if (successMessage.value === message) {
            clearSuccess();
        }
    }, 3000);
}

// Watchers
watch([hostapi, inputDevice, outputDevice], async () => {
    if (hostapi.value && inputDevice.value && outputDevice.value) {
        await applyDevices();
    }
});

// Auto-save configuration when parameters change
watch(
    [
        pth,
        index,
        hostapi,
        wasapiExclusive,
        inputDevice,
        outputDevice,
        srType,
        threshold,
        pitch,
        formant,
        indexRate,
        rmsMixRate,
        f0method,
        blockTime,
        crossfadeLength,
        nCpu,
        extraTime,
        iNoiseReduce,
        oNoiseReduce,
        usePv,
        functionMode,
    ],
    () => {
        // Debounce auto-save to avoid excessive saves
        clearTimeout(window.configSaveTimeout);
        window.configSaveTimeout = setTimeout(async () => {
            try {
                await saveConfig();
                console.log("🔄 Configuration auto-saved");
            } catch (e) {
                console.error("❌ Auto-save failed:", e);
                // Don't show error notification for auto-save failures to avoid spam
            }
        }, 1000); // Save after 1 second of no changes
    },
    { deep: true },
);

// Lifecycle
onMounted(async () => {
    try {
        console.log("🚀 Initializing RVC application...");

        console.log("📋 Loading configuration...");
        await loadConfig();

        console.log("🔍 Fetching audio devices...");
        await fetchDevices();

        console.log("🔧 Applying device settings...");
        await applyDevices();

        console.log("✅ Application initialized successfully");

        // Run startup diagnostics
        await runStartupDiagnostics();

        // Audio data and metrics are now handled by useAudioStream
    } catch (error) {
        console.error("❌ Failed to initialize application:", error);
        showError(`应用初始化失败: ${error.message || error}`);
    }

    // Listen for RVC status events
    const unlistenStatus = await listen("rvc_status", (event) => {
        const { status, message } = event.payload;
        console.log(`🔄 RVC Status: ${status} - ${message}`);

        if (status === "started") {
            isRunning.value = true;
            showSuccess(message);
        } else if (status === "stopped") {
            isRunning.value = false;
            showSuccess(message);
        }

        connectionStatus.value = status;
    });

    // Listen for RVC error events
    const unlistenError = await listen("rvc_error", (event) => {
        const { error, type, file, resolved_path, timestamp } = event.payload;
        console.error(`❌ RVC Error [${type}]:`, error);

        let userMessage = error;
        if (type === "file_not_found") {
            if (resolved_path) {
                userMessage = `文件未找到: ${file}\n解析路径: ${resolved_path}`;
            } else {
                userMessage = `文件未找到: ${file}`;
            }
        } else if (type === "already_running") {
            userMessage = "语音转换已在运行中";
        }

        // Reset running state on errors
        if (type === "file_not_found" || type === "device_error") {
            isRunning.value = false;
        }

        showError(userMessage);
    });

    // Initial status check
    await checkStatus();

    // Store unlisten functions for cleanup
    window.rvcUnlisteners = {
        status: unlistenStatus,
        error: unlistenError,
    };
});

// Status checking functions
async function checkStatus() {
    try {
        const status = await invoke("get_vc_status");

        if (!status || typeof status !== "object") {
            console.warn("⚠️ Invalid status response:", status);
            return;
        }

        const wasRunning = isRunning.value;
        isRunning.value = Boolean(status.is_running);

        // Log status changes
        if (wasRunning !== isRunning.value) {
            console.log(
                `🔄 Status changed: ${wasRunning ? "running" : "stopped"} -> ${isRunning.value ? "running" : "stopped"}`,
            );

            // Update connection status based on running state
            if (isRunning.value && connectionStatus.value === "disconnected") {
                connectionStatus.value = "connected";
            } else if (
                !isRunning.value &&
                connectionStatus.value === "connected"
            ) {
                connectionStatus.value = "disconnected";
            }
        }
    } catch (e) {
        console.error("❌ Failed to check status:", e);
        // Set disconnected status on error
        if (connectionStatus.value === "connected") {
            connectionStatus.value = "error";
        }
    }
}

function startStatusChecking() {
    stopStatusChecking(); // Clear any existing interval
    statusCheckInterval.value = setInterval(checkStatus, 1000); // Check every second
    console.log("📊 Started status checking");
}

function stopStatusChecking() {
    if (statusCheckInterval.value) {
        clearInterval(statusCheckInterval.value);
        statusCheckInterval.value = null;
        console.log("📊 Stopped status checking");
    }
}

// Cleanup on unmount
onUnmounted(() => {
    // Stop status checking
    stopStatusChecking();

    if (window.rvcUnlisteners) {
        Object.values(window.rvcUnlisteners).forEach((unlisten) => {
            if (typeof unlisten === "function") {
                unlisten();
            }
        });
        delete window.rvcUnlisteners;
    }
});
</script>

<style scoped>
/* App Layout */
.app-container {
    height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family:
        -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Header */
.app-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    padding: 0.75rem 0;
    flex-shrink: 0;
    z-index: 100;
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.app-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #2d3748;
    margin: 0;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.settings-toggle {
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 50%;
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}

.settings-toggle:hover,
.settings-toggle.active {
    background: #667eea;
    color: white;
    transform: rotate(90deg);
}

.settings-toggle svg {
    width: 18px;
    height: 18px;
}

/* Main Content */
.main-content {
    flex: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    overflow: hidden;
}

/* Top Row */
.top-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    flex-shrink: 0;
}

/* Bottom Row */
.bottom-row {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 1rem;
    flex: 1;
    min-height: 0;
}

/* Control Panel */
.control-panel {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.main-controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
}

.control-section {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.control-section h3 {
    margin: 0 0 1rem 0;
    color: #333;
    font-size: 1rem;
    font-weight: 600;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.05);
    border-radius: 6px;
}

.status-light {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #9ca3af;
    transition: all 0.3s ease;
}

.status-light.active {
    background: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    animation: pulse-light 2s infinite;
}

@keyframes pulse-light {
    0%,
    100% {
        opacity: 1;
    }
    50% {
        opacity: 0.6;
    }
}

.status-text {
    font-size: 0.9rem;
    font-weight: 500;
    color: #555;
}

/* Control buttons */
.control-buttons {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
}

.control-btn {
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    min-width: 90px;
    justify-content: center;
}

.control-btn svg {
    width: 16px;
    height: 16px;
}

.start-btn {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}

.start-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(16, 185, 129, 0.4);
}

.start-btn.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

.stop-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
}

.stop-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(239, 68, 68, 0.4);
}

.control-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}

.mode-selector {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
}

.mode-option {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    background: rgba(102, 126, 234, 0.1);
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.9rem;
}

.mode-option:hover {
    background: rgba(102, 126, 234, 0.2);
}

.mode-option input[type="radio"] {
    accent-color: #667eea;
}

/* Quick Settings */
.quick-settings {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.quick-setting {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.quick-setting label {
    font-weight: 600;
    color: #2d3748;
    font-size: 0.9rem;
}

.slider-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.slider {
    flex: 1;
    height: 5px;
    border-radius: 3px;
    background: #e2e8f0;
    outline: none;
    appearance: none;
}

.slider::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
}

.slider::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #667eea;
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
}

.value {
    min-width: 60px;
    text-align: center;
    font-weight: 600;
    color: #667eea;
    background: rgba(102, 126, 234, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Visualizations */
.visualizations {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.viz-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    flex-shrink: 0;
}

.viz-header h3 {
    margin: 0;
    color: #2d3748;
    font-size: 1.1rem;
    font-weight: 600;
}

.viz-toggle {
    padding: 0.5rem 0.75rem;
    border: 1px solid #667eea;
    border-radius: 6px;
    background: transparent;
    color: #667eea;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
    font-size: 0.85rem;
}

.viz-toggle:hover,
.viz-toggle.active {
    background: #667eea;
    color: white;
}

.viz-content {
    animation: slideDown 0.3s ease-out;
    flex: 1;
    min-height: 0;
}

.visualizer-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    height: 100%;
}

/* Performance Panel */
.performance-panel {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

/* Settings Panel */
.settings-panel {
    position: fixed;
    top: 0;
    right: -380px;
    width: 380px;
    height: 100vh;
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(20px);
    box-shadow: -4px 0 20px rgba(0, 0, 0, 0.1);
    transition: right 0.3s ease-out;
    z-index: 200;
    overflow-y: auto;
}

.settings-panel.active {
    right: 0;
}

.settings-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    background: rgba(102, 126, 234, 0.05);
    flex-shrink: 0;
}

.settings-header h2 {
    margin: 0;
    color: #2d3748;
    font-size: 1.3rem;
    font-weight: 600;
}

.close-btn {
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 50%;
    background: rgba(244, 67, 54, 0.1);
    color: #f44336;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}

.close-btn:hover {
    background: #f44336;
    color: white;
}

.close-btn svg {
    width: 14px;
    height: 14px;
}

.settings-tabs {
    display: flex;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    flex-shrink: 0;
}

.tab-btn {
    flex: 1;
    padding: 0.75rem;
    border: none;
    background: transparent;
    color: #718096;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
    font-size: 0.9rem;
}

.tab-btn.active {
    color: #667eea;
    background: rgba(102, 126, 234, 0.1);
    border-bottom: 2px solid #667eea;
}

.tab-btn:hover:not(.active) {
    background: rgba(0, 0, 0, 0.05);
}

.settings-content {
    padding: 1.25rem;
    flex: 1;
}

.settings-section {
    display: block;
}

.settings-section h3 {
    margin: 0 0 1rem 0;
    color: #2d3748;
    font-size: 1.1rem;
    font-weight: 600;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
}

.form-group {
    margin-bottom: 1.25rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 600;
    color: #4a5568;
    font-size: 0.9rem;
}

.file-input,
.select-input {
    width: 100%;
    padding: 0.65rem;
    border: 2px solid #e2e8f0;
    border-radius: 6px;
    font-size: 0.9rem;
    transition: all 0.2s;
}

.file-input:focus,
.select-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.file-name {
    display: block;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(102, 126, 234, 0.1);
    border-radius: 4px;
    font-size: 0.8rem;
    color: #667eea;
    word-break: break-all;
}

.file-hint {
    display: block;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
    font-size: 0.8rem;
    color: #666;
    font-style: italic;
}

.path-suggestions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.suggestion-btn {
    padding: 0.25rem 0.5rem;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
    transition: background-color 0.2s;
}

.suggestion-btn:hover {
    background: #5a67d8;
}

.suggestion-btn.secondary {
    background: #9ca3af;
}

.suggestion-btn.secondary:hover {
    background: #6b7280;
}

.reload-btn {
    padding: 0.65rem 1.25rem;
    border: 2px solid #667eea;
    border-radius: 6px;
    background: transparent;
    color: #667eea;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
    font-size: 0.9rem;
}

.reload-btn:hover {
    background: #667eea;
    color: white;
}

.radio-group,
.checkbox-group {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.radio-label,
.checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    padding: 0.4rem;
    border-radius: 4px;
    transition: background 0.2s;
    font-size: 0.9rem;
}

.radio-label:hover,
.checkbox-label:hover {
    background: rgba(102, 126, 234, 0.05);
}

.radio-label input[type="radio"],
.checkbox-label input[type="checkbox"] {
    accent-color: #667eea;
}

.info-text {
    font-size: 0.8rem;
    color: #718096;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: rgba(113, 128, 150, 0.1);
    border-radius: 4px;
}

/* Connection Status */
.connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.75rem;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.1);
}

.status-label {
    font-weight: 500;
    color: #4a5568;
    font-size: 0.85rem;
}

.status-indicator {
    padding: 0.2rem 0.6rem;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 500;
    transition: all 0.2s;
}

.status-indicator.connected {
    background: #48bb78;
    color: white;
}

.status-indicator.connecting {
    background: #ed8936;
    color: white;
    animation: pulse 1.5s infinite;
}

.status-indicator.disconnected {
    background: #a0aec0;
    color: white;
}

.status-indicator.error {
    background: #f56565;
    color: white;
    animation: shake 0.5s;
}

/* Overlay */
.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.5);
    z-index: 150;
}

/* Animations */
@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%,
    100% {
        opacity: 1;
    }
    50% {
        opacity: 0.6;
    }
}

@keyframes shake {
    0%,
    100% {
        transform: translateX(0);
    }
    25% {
        transform: translateX(-2px);
    }
    75% {
        transform: translateX(2px);
    }
}

/* Responsive Design for 800x600 and smaller screens */
@media (max-width: 800px), (max-height: 600px) {
    .app-container {
        font-size: 12px;
        height: 100vh;
        overflow: hidden;
    }

    .app-header {
        padding: 0.3rem 0;
        min-height: 45px;
        flex-shrink: 0;
    }

    .header-content {
        padding: 0 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
    }

    .app-title {
        font-size: 0.85rem;
        margin: 0;
        white-space: nowrap;
    }

    .connection-status {
        font-size: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    .status-label {
        display: none;
    }

    .status-indicator {
        padding: 0.15rem 0.3rem;
        font-size: 0.65rem;
        border-radius: 3px;
    }

    .settings-toggle {
        padding: 0.3rem;
        width: 28px;
        height: 28px;
        flex-shrink: 0;
    }

    .settings-toggle svg {
        width: 14px;
        height: 14px;
    }

    .main-content {
        padding: 0.3rem;
        gap: 0.3rem;
        flex: 1;
        overflow-y: auto;
    }

    .top-row {
        grid-template-columns: 1fr;
        gap: 0.3rem;
    }

    .bottom-row {
        grid-template-columns: 1fr;
        gap: 0.3rem;
    }

    .control-panel,
    .quick-settings,
    .visualizations,
    .performance-panel {
        padding: 0.5rem;
        margin: 0;
        border-radius: 6px;
    }

    .control-section {
        margin-bottom: 0.5rem;
    }

    .control-section:last-child {
        margin-bottom: 0;
    }

    .control-section h3 {
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }

    .status-indicator.connected,
    .status-indicator.connecting,
    .status-indicator.disconnected,
    .status-indicator.error {
        font-size: 0.65rem;
    }

    .status-light {
        width: 6px;
        height: 6px;
    }

    .main-controls {
        flex-direction: row;
        flex-wrap: wrap;
        gap: 0.3rem;
        justify-content: center;
    }

    .control-buttons {
        gap: 0.3rem;
        justify-content: center;
    }

    .control-btn {
        padding: 0.35rem 0.7rem;
        font-size: 0.7rem;
        min-width: 70px;
        border-radius: 4px;
    }

    .control-btn svg {
        width: 12px;
        height: 12px;
    }

    .mode-selector {
        gap: 0.25rem;
        justify-content: center;
        flex-wrap: wrap;
    }

    .mode-option {
        padding: 0.25rem 0.5rem;
        font-size: 0.65rem;
        min-width: 45px;
        border-radius: 3px;
    }

    .quick-setting {
        margin-bottom: 0.4rem;
    }

    .quick-setting:last-child {
        margin-bottom: 0;
    }

    .quick-setting label {
        font-size: 0.7rem;
        margin-bottom: 0.15rem;
        display: block;
    }

    .slider-container {
        gap: 0.3rem;
        align-items: center;
    }

    .slider {
        height: 3px;
        min-width: 40px;
        flex: 1;
    }

    .slider::-webkit-slider-thumb {
        width: 12px;
        height: 12px;
    }

    .slider::-moz-range-thumb {
        width: 12px;
        height: 12px;
    }

    .value {
        font-size: 0.65rem;
        min-width: 30px;
        text-align: center;
    }

    .visualizer-grid {
        grid-template-columns: 1fr;
        gap: 0.3rem;
    }

    .settings-panel {
        width: 100%;
        height: 100vh;
        border-radius: 0;
        padding: 0.5rem;
        overflow-y: auto;
    }

    .settings-header {
        padding: 0.5rem;
        margin: -0.5rem -0.5rem 0.5rem -0.5rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .settings-header h2 {
        font-size: 0.9rem;
        margin: 0;
    }

    .close-btn {
        width: 24px;
        height: 24px;
        padding: 0.2rem;
    }

    .close-btn svg {
        width: 12px;
        height: 12px;
    }

    .settings-tabs {
        gap: 0.2rem;
        flex-wrap: wrap;
        margin-bottom: 0.5rem;
    }

    .tab-btn {
        padding: 0.25rem 0.5rem;
        font-size: 0.7rem;
        flex: 1;
        min-width: 55px;
        border-radius: 3px;
    }

    .settings-content {
        max-height: calc(100vh - 120px);
        overflow-y: auto;
    }

    .settings-section {
        margin-bottom: 0.75rem;
    }

    .settings-section:last-child {
        margin-bottom: 0;
    }

    .settings-section h3 {
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
    }

    .form-group {
        margin-bottom: 0.5rem;
    }

    .form-group:last-child {
        margin-bottom: 0;
    }

    .form-group label {
        font-size: 0.7rem;
        margin-bottom: 0.15rem;
        display: block;
    }

    .file-input,
    .select-input {
        padding: 0.3rem;
        font-size: 0.7rem;
        border-radius: 3px;
    }

    .file-name {
        font-size: 0.65rem;
        padding: 0.25rem;
        line-height: 1.2;
    }

    .file-hint {
        font-size: 0.6rem;
        line-height: 1.2;
    }

    .radio-group,
    .checkbox-group {
        gap: 0.3rem;
        flex-wrap: wrap;
    }

    .radio-label,
    .checkbox-label {
        font-size: 0.7rem;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
    }

    .suggestion-btn {
        padding: 0.25rem 0.5rem;
        font-size: 0.65rem;
        border-radius: 3px;
    }

    .reload-btn {
        padding: 0.25rem;
        width: 24px;
        height: 24px;
        border-radius: 3px;
    }

    .reload-btn svg {
        width: 12px;
        height: 12px;
    }

    .info-text {
        font-size: 0.65rem;
        line-height: 1.3;
        padding: 0.3rem;
    }
}

@media (max-width: 1200px) {
    .main-content {
        padding: 1rem;
    }

    .bottom-row {
        grid-template-columns: 1fr;
        gap: 1rem;
    }

    .visualizer-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) and (min-width: 481px) {
    .app-container {
        font-size: 14px;
    }

    .header-content {
        padding: 0 1rem;
    }

    .app-title {
        font-size: 1.1rem;
    }

    .top-row {
        grid-template-columns: 1fr;
    }

    .main-controls {
        flex-direction: column;
        gap: 0.5rem;
    }

    .mode-selector {
        flex-direction: column;
        gap: 0.5rem;
    }

    .settings-panel {
        width: 100%;
        height: 100vh;
        border-radius: 0;
    }

    .settings-tabs {
        overflow-x: auto;
    }

    .tab-btn {
        white-space: nowrap;
        min-width: 120px;
    }
}

@media (max-width: 480px) {
    .main-content {
        padding: 0.75rem;
    }

    .control-panel,
    .quick-settings,
    .visualizations,
    .performance-panel {
        padding: 1rem;
    }

    .slider-container {
        flex-direction: column;
        align-items: stretch;
        gap: 0.5rem;
    }

    .value {
        text-align: left;
        min-width: auto;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .app-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    }

    .control-panel,
    .quick-settings,
    .visualizations,
    .performance-panel,
    .settings-panel {
        background: rgba(45, 55, 72, 0.95);
        color: #e2e8f0;
    }

    .app-title,
    .viz-header h3,
    .settings-header h2,
    .settings-section h3 {
        color: #e2e8f0;
    }

    .quick-setting label,
    .form-group label {
        color: #cbd5e0;
    }

    .slider {
        background: #4a5568;
    }

    .file-input,
    .select-input {
        background: #2d3748;
        border-color: #4a5568;
        color: #e2e8f0;
    }

    .control-section h3 {
        color: #e2e8f0;
    }

    .status-indicator {
        background: rgba(255, 255, 255, 0.1);
    }

    .status-text {
        color: #e2e8f0;
    }

    .file-hint {
        background: rgba(255, 255, 255, 0.05);
        color: #a0a0a0;
    }

    .suggestion-btn {
        background: #4c51bf;
    }

    .suggestion-btn:hover {
        background: #553c9a;
    }

    .suggestion-btn.secondary {
        background: #4a5568;
    }

    .suggestion-btn.secondary:hover {
        background: #2d3748;
    }

    .info-text {
        background: rgba(113, 128, 150, 0.2);
        color: #a0aec0;
    }
}

/* Notification Area */
.notification-area {
    position: fixed;
    top: 60px;
    right: 10px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-width: 300px;
    pointer-events: none;
}

.notification {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(10px);
    cursor: pointer;
    transition: all 0.3s ease;
    animation: slideIn 0.3s ease;
}

.notification:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.notification.error {
    background: rgba(245, 101, 101, 0.95);
    color: white;
    border-left: 4px solid #e53e3e;
}

.notification.success {
    background: rgba(72, 187, 120, 0.95);
    color: white;
    border-left: 4px solid #38a169;
}

.notification-icon {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
}

.notification-text {
    flex: 1;
    font-size: 0.8rem;
    line-height: 1.3;
    font-weight: 500;
    word-break: break-word;
}

.notification-copy {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: background-color 0.2s ease;
    margin-right: 0.25rem;
}

.notification-copy:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

.notification-copy svg {
    width: 14px;
    height: 14px;
}

.notification-close {
    background: none;
    border: none;
    color: inherit;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: background-color 0.2s ease;
}

.notification-close:hover {
    background: rgba(255, 255, 255, 0.2);
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@media (max-width: 800px), (max-height: 600px) {
    .notification-area {
        top: 45px;
        right: 6px;
        left: 6px;
        max-width: none;
        gap: 4px;
    }

    .notification {
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        font-size: 0.65rem;
        gap: 0.4rem;
    }

    .notification-text {
        font-size: 0.65rem;
        line-height: 1.2;
    }

    .notification-icon {
        width: 12px;
        height: 12px;
    }

    .notification-copy,
    .notification-close {
        width: 18px;
        height: 18px;
        padding: 0.15rem;
    }

    .notification-copy svg {
        width: 10px;
        height: 10px;
    }
}

@media (max-width: 768px) and (min-width: 481px) {
    .notification-area {
        right: 10px;
        left: 10px;
        max-width: none;
    }

    .notification {
        padding: 10px 12px;
    }

    .notification-text {
        font-size: 0.85rem;
    }
}
</style>
