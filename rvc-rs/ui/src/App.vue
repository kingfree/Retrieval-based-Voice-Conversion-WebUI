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

        <!-- Main Content -->
        <main class="main-content">
            <!-- Top Row: Controls + Quick Settings -->
            <div class="top-row">
                <!-- Control Panel -->
                <div class="control-panel">
                    <div class="main-controls">
                        <button
                            class="control-btn start-btn"
                            @click="startVc"
                            :disabled="!canStartVc"
                        >
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M8,5.14V19.14L19,12.14L8,5.14Z" />
                            </svg>
                            开始转换
                        </button>
                        <button
                            class="control-btn stop-btn"
                            @click="stopVc"
                            :disabled="!isStreaming"
                        >
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <path d="M18,18H6V6H18V18Z" />
                            </svg>
                            停止转换
                        </button>
                    </div>

                    <div class="mode-selector">
                        <label class="mode-option">
                            <input type="radio" value="im" v-model="functionMode" />
                            <span>输入监听</span>
                        </label>
                        <label class="mode-option">
                            <input type="radio" value="vc" v-model="functionMode" />
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
                                @level-change="(level) => (outputVolume = level)"
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
                            type="file"
                            accept=".pth"
                            @change="(e) => (pth = e.target.files[0]?.name || '')"
                            class="file-input"
                        />
                        <span v-if="pth" class="file-name">{{ pth }}</span>
                    </div>

                    <div class="form-group">
                        <label>Index 文件</label>
                        <input
                            type="file"
                            accept=".index"
                            @change="(e) => (index = e.target.files[0]?.name || '')"
                            class="file-input"
                        />
                        <span v-if="index" class="file-name">{{ index }}</span>
                    </div>
                </div>

                <!-- Audio Settings -->
                <div v-show="activeTab === 'audio'" class="settings-section">
                    <h3>音频设备</h3>
                    <div class="form-group">
                        <button type="button" @click="reloadDevices" class="reload-btn">
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
                            <option v-for="d in inputDevices" :key="d" :value="d">
                                {{ d }}
                            </option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>输出设备</label>
                        <select v-model="outputDevice" class="select-input">
                            <option v-for="d in outputDevices" :key="d" :value="d">
                                {{ d }}
                            </option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>采样率设置</label>
                        <div class="radio-group">
                            <label class="radio-label">
                                <input type="radio" value="sr_model" v-model="srType" />
                                使用模型采样率
                            </label>
                            <label class="radio-label">
                                <input type="radio" value="sr_device" v-model="srType" />
                                使用设备采样率
                            </label>
                        </div>
                        <div class="info-text">当前采样率: {{ sampleRate }} Hz</div>
                    </div>
                </div>

                <!-- Processing Settings -->
                <div v-show="activeTab === 'processing'" class="settings-section">
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
                            <span class="value">{{ indexRate.toFixed(2) }}</span>
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
                            <span class="value">{{ rmsMixRate.toFixed(2) }}</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>音高算法</label>
                        <div class="radio-group">
                            <label v-for="method in f0Methods" :key="method" class="radio-label">
                                <input type="radio" :value="method" v-model="f0method" />
                                {{ method }}
                            </label>
                        </div>
                    </div>
                </div>

                <!-- Performance Settings -->
                <div v-show="activeTab === 'performance'" class="settings-section">
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
                            <span class="value">{{ blockTime.toFixed(2) }}s</span>
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
                            <span class="value">{{ crossfadeLength.toFixed(2) }}s</span>
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
                            <span class="value">{{ extraTime.toFixed(2) }}s</span>
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
import { ref, computed, watch, onMounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import AudioVisualizer from "./components/AudioVisualizer.vue";
import PerformanceMonitor from "./components/PerformanceMonitor.vue";
import { useAudioStream } from "./composables/useAudioStream.js";

// UI State
const showSettings = ref(false);
const showVisualizations = ref(false); // 默认收起可视化器
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
const indexRate = ref(0.0);
const rmsMixRate = ref(0.0);
const f0method = ref("fcpe");

// Performance Parameters
const blockTime = ref(0.25);
const crossfadeLength = ref(0.05);
const nCpu = ref(1);
const extraTime = ref(2.5);
const iNoiseReduce = ref(false);
const oNoiseReduce = ref(false);
const usePv = ref(false);
const functionMode = ref("vc");
const sampleRate = ref(0);
const delayTime = ref(0);
const inferTime = ref(0);

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
    clearBuffers,
} = useAudioStream();

// Audio visualizer component refs
const inputVisualizer = ref(null);
const outputVisualizer = ref(null);

// Computed properties
const canStartVc = computed(() => {
    return (
        pth.value &&
        index.value &&
        hostapi.value &&
        inputDevice.value &&
        outputDevice.value
    );
});

// Methods
function toggleSettings() {
    showSettings.value = !showSettings.value;
}

function toggleVisualizations() {
    showVisualizations.value = !showVisualizations.value;
}

function send(event, value) {
    invoke("event_handler", { event, value: value?.toString() });
}

async function fetchDevices() {
    try {
        const list = await invoke("update_devices", { hostapi: hostapi.value });
        hostapis.value = list.hostapis;
        inputDevices.value = list.input_devices;
        outputDevices.value = list.output_devices;
        if (!hostapis.value.includes(hostapi.value)) {
            hostapi.value = hostapis.value[0] || "";
        }
        if (!inputDevices.value.includes(inputDevice.value)) {
            inputDevice.value = inputDevices.value[0] || "";
        }
        if (!outputDevices.value.includes(outputDevice.value)) {
            outputDevice.value = outputDevices.value[0] || "";
        }
    } catch (e) {
        console.error("failed to fetch devices", e);
    }
}

async function applyDevices() {
    try {
        sampleRate.value = await invoke("set_devices", {
            hostapi: hostapi.value,
            inputDevice: inputDevice.value,
            outputDevice: outputDevice.value,
        });
    } catch (e) {
        console.error("failed to apply devices", e);
    }
}

async function loadConfig() {
    try {
        const config = await invoke("load_config");

        // Apply loaded configuration
        pth.value = config.pth || "";
        index.value = config.index || "";
        hostapi.value = config.hostapi || "";
        wasapiExclusive.value = config.wasapiExclusive || false;
        inputDevice.value = config.inputDevice || "";
        outputDevice.value = config.outputDevice || "";
        srType.value = config.srType || "sr_model";

        // Processing parameters
        threshold.value = config.threshold ?? -60;
        pitch.value = config.pitch ?? 0;
        formant.value = config.formant ?? 0.0;
        indexRate.value = config.indexRate ?? 0.0;
        rmsMixRate.value = config.rmsMixRate ?? 0.0;
        f0method.value = config.f0method || "fcpe";

        // Performance parameters
        blockTime.value = config.blockTime ?? 0.25;
        crossfadeLength.value = config.crossfadeLength ?? 0.05;
        nCpu.value = config.nCpu ?? 1;
        extraTime.value = config.extraTime ?? 2.5;
        iNoiseReduce.value = config.iNoiseReduce ?? false;
        oNoiseReduce.value = config.oNoiseReduce ?? false;
        usePv.value = config.usePv ?? false;
        functionMode.value = config.functionMode || "vc";
    } catch (e) {
        console.error("failed to load config", e);
    }
}

async function reloadDevices() {
    await fetchDevices();
    await applyDevices();
}

async function startVc() {
    if (!canStartVc.value) return;

    try {
        await invoke("start_vc", {
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
            functionMode: functionMode.value,
        });

        await initializeAudioStream();
    } catch (e) {
        console.error("failed to start vc", e);
    }
}

async function stopVc() {
    try {
        await invoke("stop_vc");
        clearBuffers();
    } catch (e) {
        console.error("failed to stop vc", e);
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

// Watchers
watch([hostapi, inputDevice, outputDevice], async () => {
    if (hostapi.value && inputDevice.value && outputDevice.value) {
        await applyDevices();
    }
});

// Lifecycle
onMounted(async () => {
    await loadConfig();
    await fetchDevices();
    await applyDevices();

    // Listen for audio data events
    const unlistenAudio = await listen("audio_data", (event) => {
        const { input_data, output_data, sample_rate } = event.payload;
        inputAudioData.value = input_data;
        outputAudioData.value = output_data;
        sampleRate.value = sample_rate;
    });

    // Listen for performance events
    const unlistenPerformance = await listen("performance_data", (event) => {
        const {
            inference_time,
            algorithm_latency,
            buffer_latency,
            total_latency,
        } = event.payload;
        inferenceTime.value = inference_time;
        algorithmLatency.value = algorithm_latency;
        bufferLatency.value = buffer_latency;
        totalLatency.value = total_latency;
    });
});
</script>

<style scoped>
/* App Layout */
.app-container {
    height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
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
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.control-btn {
    flex: 1;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.control-btn svg {
    width: 16px;
    height: 16px;
}

.start-btn {
    background: linear-gradient(135deg, #4caf50, #45a049);
    color: white;
}

.start-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
}

.stop-btn {
    background: linear-gradient(135deg, #f44336, #da190b);
    color: white;
}

.stop-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(244, 67, 54, 0.4);
}

.control-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
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
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.6;
    }
}

@keyframes shake {
    0%, 100% {
        transform: translateX(0);
    }
    25% {
        transform: translateX(-2px);
    }
    75% {
        transform: translateX(2px);
    }
}

/* Responsive Design */
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

@media (max-width: 768px) {
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
        width: 100vw;
        right: -100vw;
    }

    .settings-tabs {
        overflow-x: auto;
    }

    .tab-btn {
        min-width: 70px;
        white-space: nowrap;
        padding: 0.6rem;
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

    .info-text {
        background: rgba(113, 128, 150, 0.2);
        color: #a0aec0;
    }
}
</style>
