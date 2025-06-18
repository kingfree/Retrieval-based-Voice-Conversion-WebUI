<template>
    <div class="container">
        <!-- Audio Visualizers -->
        <div class="visualizer-section">
            <div class="visualizer-row">
                <AudioVisualizer
                    title="输入音频可视化"
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
                    title="输出音频可视化"
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

        <div class="row">
            <section>
                <h2>加载模型</h2>
                <div>
                    <label>pth 文件:</label>
                    <input
                        type="file"
                        accept=".pth"
                        @change="
                            (e) => (pth.value = e.target.files[0]?.name || '')
                        "
                    />
                </div>
                <div>
                    <label>index 文件:</label>
                    <input
                        type="file"
                        accept=".index"
                        @change="
                            (e) => (index.value = e.target.files[0]?.name || '')
                        "
                    />
                </div>
            </section>

            <section>
                <h2>音频设备</h2>
                <div>
                    <button type="button" @click="reloadDevices">
                        重载设备列表
                    </button>
                </div>
                <div>
                    <label>设备类型:</label>
                    <select v-model="hostapi">
                        <option v-for="h in hostapis" :key="h" :value="h">
                            {{ h }}
                        </option>
                    </select>
                    <label
                        ><input type="checkbox" v-model="wasapiExclusive" />
                        独占 WASAPI 设备</label
                    >
                </div>
                <div>
                    <label>输入设备:</label>
                    <select v-model="inputDevice">
                        <option v-for="d in inputDevices" :key="d" :value="d">
                            {{ d }}
                        </option>
                    </select>
                </div>
                <div>
                    <label>输出设备:</label>
                    <select v-model="outputDevice">
                        <option v-for="d in outputDevices" :key="d" :value="d">
                            {{ d }}
                        </option>
                    </select>
                </div>
                <div>
                    <label
                        ><input
                            type="radio"
                            value="sr_model"
                            v-model="srType"
                        />
                        使用模型采样率</label
                    >
                    <label
                        ><input
                            type="radio"
                            value="sr_device"
                            v-model="srType"
                        />
                        使用设备采样率</label
                    >
                    <span class="info">采样率: {{ sampleRate }}</span>
                </div>
            </section>
        </div>
        <div class="row">
            <section>
                <h2>常规设置</h2>
                <div class="slider">
                    <label>响应阈值 {{ threshold }}</label>
                    <input
                        type="range"
                        min="-60"
                        max="0"
                        v-model.number="threshold"
                    />
                </div>
                <div class="slider">
                    <label>音调设置 {{ pitch }}</label>
                    <input
                        type="range"
                        min="-16"
                        max="16"
                        step="1"
                        v-model.number="pitch"
                    />
                </div>
                <div class="slider">
                    <label>性别因子 {{ formant }}</label>
                    <input
                        type="range"
                        min="-2"
                        max="2"
                        step="0.05"
                        v-model.number="formant"
                    />
                </div>
                <div class="slider">
                    <label>Index Rate {{ indexRate }}</label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        v-model.number="indexRate"
                    />
                </div>
                <div class="slider">
                    <label>响度因子 {{ rmsMixRate }}</label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        v-model.number="rmsMixRate"
                    />
                </div>
                <div>
                    <label>音高算法:</label>
                    <label
                        ><input
                            type="radio"
                            value="pm"
                            v-model="f0method"
                        />pm</label
                    >
                    <label
                        ><input
                            type="radio"
                            value="harvest"
                            v-model="f0method"
                        />harvest</label
                    >
                    <label
                        ><input
                            type="radio"
                            value="crepe"
                            v-model="f0method"
                        />crepe</label
                    >
                    <label
                        ><input
                            type="radio"
                            value="rmvpe"
                            v-model="f0method"
                        />rmvpe</label
                    >
                    <label
                        ><input
                            type="radio"
                            value="fcpe"
                            v-model="f0method"
                        />fcpe</label
                    >
                </div>
            </section>

            <section>
                <h2>性能设置</h2>
                <div class="slider">
                    <label>采样长度 {{ blockTime }}</label>
                    <input
                        type="range"
                        min="0.02"
                        max="1.5"
                        step="0.01"
                        v-model.number="blockTime"
                    />
                </div>
                <div class="slider">
                    <label>淡入淡出长度 {{ crossfadeLength }}</label>
                    <input
                        type="range"
                        min="0.01"
                        max="0.15"
                        step="0.01"
                        v-model.number="crossfadeLength"
                    />
                </div>
                <div class="slider">
                    <label>harvest进程数 {{ nCpu }}</label>
                    <input
                        type="range"
                        min="1"
                        max="8"
                        step="1"
                        v-model.number="nCpu"
                    />
                </div>
                <div class="slider">
                    <label>额外推理时长 {{ extraTime }}</label>
                    <input
                        type="range"
                        min="0.05"
                        max="5"
                        step="0.01"
                        v-model.number="extraTime"
                    />
                </div>
                <div>
                    <label
                        ><input type="checkbox" v-model="iNoiseReduce" />
                        输入降噪</label
                    >
                    <label
                        ><input type="checkbox" v-model="oNoiseReduce" />
                        输出降噪</label
                    >
                    <label
                        ><input type="checkbox" v-model="usePv" />
                        启用相位声码器</label
                    >
                </div>
            </section>
        </div>
        <div class="actions">
            <button type="button" @click="startVc">开始音频转换</button>
            <button type="button" @click="stopVc">停止音频转换</button>
            <label
                ><input type="radio" value="im" v-model="functionMode" />
                输入监听</label
            >
            <label
                ><input type="radio" value="vc" v-model="functionMode" />
                输出变声</label
            >
            <span class="info">算法延迟: {{ delayTime }} ms</span>
            <span class="info">推理时间: {{ inferTime }} ms</span>

            <!-- Connection Status -->
            <div class="connection-status">
                <span class="status-label">连接状态:</span>
                <span
                    :class="['status-indicator', connectionStatus]"
                    :title="lastError || '正常'"
                >
                    {{ getConnectionStatusText() }}
                </span>
            </div>

            <!-- Demo Mode Controls -->
            <div v-if="isDemoMode" class="demo-controls">
                <span class="demo-label">演示模式:</span>
                <button
                    type="button"
                    @click="toggleDemoStream"
                    :class="{ active: isStreaming }"
                >
                    {{ isStreaming ? "停止演示" : "开始演示" }}
                </button>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, watch, onMounted } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import AudioVisualizer from "./components/AudioVisualizer.vue";
import { useAudioStream } from "./composables/useAudioStream.js";

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

const threshold = ref(-60);
const pitch = ref(0);
const formant = ref(0.0);
const indexRate = ref(0.0);
const rmsMixRate = ref(0.0);
const f0method = ref("fcpe");

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

// Audio stream composable (with demo mode for testing)
const isDemoMode = ref(!window.__TAURI__); // Use demo mode if not in Tauri
const {
    inputAudioData,
    outputAudioData,
    isStreaming,
    inputVolume,
    outputVolume,
    stats,
    lastError,
    connectionStatus,
    initializeAudioStream,
    clearBuffers,
    startDemoStream,
    stopDemoStream,
} = useAudioStream(isDemoMode.value);

// Audio visualizer component refs
const inputVisualizer = ref(null);
const outputVisualizer = ref(null);

function send(event, value) {
    if (isDemoMode.value) {
        console.log(`Demo mode: ${event} = ${value}`);
        return;
    }
    invoke("event_handler", { event, value: value?.toString() });
}

async function fetchDevices() {
    if (isDemoMode.value) {
        // Mock device data for demo mode
        hostapis.value = ["DirectSound", "WASAPI"];
        inputDevices.value = ["Default Input", "Microphone"];
        outputDevices.value = ["Default Output", "Speakers"];
        hostapi.value = "DirectSound";
        inputDevice.value = "Default Input";
        outputDevice.value = "Default Output";
        return;
    }
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
    if (isDemoMode.value) {
        sampleRate.value = 44100;
        return;
    }
    try {
        sampleRate.value = await invoke("set_devices", {
            hostapi: hostapi.value,
            inputDevice: inputDevice.value,
            outputDevice: outputDevice.value,
        });
    } catch (e) {
        console.error("failed to set devices", e);
    }
}

async function loadConfig() {
    if (isDemoMode.value) {
        // Load default demo values
        hostapi.value = "DirectSound";
        wasapiExclusive.value = false;
        inputDevice.value = "Default Input";
        outputDevice.value = "Default Output";
        srType.value = "sr_model";
        threshold.value = -60;
        pitch.value = 0;
        formant.value = 0.0;
        indexRate.value = 0.0;
        rmsMixRate.value = 0.0;
        blockTime.value = 0.25;
        crossfadeLength.value = 0.05;
        nCpu.value = 1;
        extraTime.value = 2.5;
        f0method.value = "fcpe";
        usePv.value = false;
        await fetchDevices();
        await applyDevices();
        return;
    }
    try {
        const cfg = await invoke("get_init_config");
        hostapi.value = cfg.sg_hostapi;
        wasapiExclusive.value = cfg.sg_wasapi_exclusive;
        inputDevice.value = cfg.sg_input_device;
        outputDevice.value = cfg.sg_output_device;
        srType.value = cfg.sr_type;
        threshold.value = cfg.threhold ?? cfg.threshold;
        pitch.value = cfg.pitch;
        formant.value = cfg.formant;
        indexRate.value = cfg.index_rate;
        rmsMixRate.value = cfg.rms_mix_rate;
        blockTime.value = cfg.block_time;
        crossfadeLength.value = cfg.crossfade_length;
        nCpu.value = cfg.n_cpu;
        extraTime.value = cfg.extra_time;
        f0method.value = cfg.f0method;
        usePv.value = cfg.use_pv;
        await fetchDevices();
        await applyDevices();
    } catch (e) {
        console.error("failed to load config", e);
    }
}

onMounted(async () => {
    // Always load config (demo or real)
    await loadConfig();

    // Initialize audio stream for visualization
    await initializeAudioStream();

    // Only set up Tauri event listeners if not in demo mode
    if (!isDemoMode.value) {
        // Listen for backend events
        await listen("vc_started", () => {
            console.log("Voice conversion started");
            clearBuffers(); // Clear audio buffers when starting
            // Clear visualizers
            if (inputVisualizer.value) inputVisualizer.value.clear();
            if (outputVisualizer.value) outputVisualizer.value.clear();
        });

        await listen("vc_stopped", () => {
            console.log("Voice conversion stopped");
            clearBuffers(); // Clear audio buffers when stopping
            // Clear visualizers
            if (inputVisualizer.value) inputVisualizer.value.clear();
            if (outputVisualizer.value) outputVisualizer.value.clear();
        });

        await listen("delay_time", (event) => {
            delayTime.value = Number(event.payload);
        });

        await listen("devices_updated", (event) => {
            console.log("Devices updated:", event.payload);
            const deviceInfo = event.payload;
            hostapis.value = deviceInfo.hostapis;
            inputDevices.value = deviceInfo.input_devices;
            outputDevices.value = deviceInfo.output_devices;
        });

        // Listen for inference time updates
        await listen("infer_time", (event) => {
            inferTime.value = Number(event.payload);
        });
    } else {
        console.log("Running in demo mode - Tauri event listeners disabled");
    }
});

watch(threshold, (v) => send("threshold", v));
watch(pitch, (v) => send("pitch", v));
watch(formant, (v) => send("formant", v));
watch(blockTime, (v) => send("block_time", v));
watch(crossfadeLength, (v) => send("crossfade_length", v));
watch(srType, (v) => send("sr_type", v));
watch(indexRate, (v) => send("index_rate", v));
watch(rmsMixRate, (v) => send("rms_mix_rate", v));
watch(f0method, (v) => send("f0method", v));
watch(nCpu, (v) => send("n_cpu", v));
watch(extraTime, (v) => send("extra_time", v));
watch(pth, (v) => send("pth_path", v));
watch(index, (v) => send("index_path", v));
watch(iNoiseReduce, (v) => send("I_noise_reduce", v));
watch(oNoiseReduce, (v) => send("O_noise_reduce", v));
watch(usePv, (v) => send("use_pv", v));
watch(functionMode, (v) => send("function_mode", v));
watch(wasapiExclusive, (v) => send("sg_wasapi_exclusive", v));
watch(hostapi, async (v) => {
    try {
        await invoke("event_handler", { event: "sg_hostapi", value: v });
        await fetchDevices();
        await applyDevices();
    } catch (error) {
        console.error("Failed to update hostapi:", error);
    }
});
watch(inputDevice, async (v) => {
    try {
        await invoke("event_handler", { event: "sg_input_device", value: v });
        await applyDevices();
    } catch (error) {
        console.error("Failed to update input device:", error);
    }
});
watch(outputDevice, async (v) => {
    try {
        await invoke("event_handler", { event: "sg_output_device", value: v });
        await applyDevices();
    } catch (error) {
        console.error("Failed to update output device:", error);
    }
});

async function reloadDevices() {
    if (isDemoMode.value) {
        console.log("Demo mode: reload devices");
        return;
    }
    try {
        await invoke("event_handler", { event: "reload_devices", value: null });
        await fetchDevices();
    } catch (error) {
        console.error("Failed to reload devices:", error);
    }
}

async function startVc() {
    if (isDemoMode.value) {
        startDemoStream();
        return;
    }

    try {
        // Save configuration first
        await invoke("set_values", {
            values: {
                pth_path: pth.value,
                index_path: index.value,
                sg_hostapi: hostapi.value,
                sg_wasapi_exclusive: wasapiExclusive.value,
                sg_input_device: inputDevice.value,
                sg_output_device: outputDevice.value,
                sr_type: srType.value,
                threhold: threshold.value,
                pitch: pitch.value,
                formant: formant.value,
                index_rate: indexRate.value,
                rms_mix_rate: rmsMixRate.value,
                block_time: blockTime.value,
                crossfade_length: crossfadeLength.value,
                extra_time: extraTime.value,
                n_cpu: nCpu.value,
                use_pv: usePv.value,
                f0method: f0method.value,
                I_noise_reduce: iNoiseReduce.value,
                O_noise_reduce: oNoiseReduce.value,
            },
        });

        // Start voice conversion
        await invoke("event_handler", { event: "start_vc", value: null });
    } catch (error) {
        console.error("Failed to start voice conversion:", error);
        alert("启动语音转换失败: " + error);
    }
}

async function stopVc() {
    if (isDemoMode.value) {
        stopDemoStream();
        return;
    }

    try {
        await invoke("event_handler", { event: "stop_vc", value: null });
    } catch (error) {
        console.error("Failed to stop voice conversion:", error);
    }
}

function toggleDemoStream() {
    if (isStreaming.value) {
        stopDemoStream();
    } else {
        startDemoStream();
    }
}

function getConnectionStatusText() {
    switch (connectionStatus.value) {
        case "connected":
            return "已连接";
        case "connecting":
            return "连接中";
        case "disconnected":
            return "未连接";
        case "error":
            return "错误";
        default:
            return "未知";
    }
}

// Handle audio visualization errors
function handleVisualizationError(error) {
    console.error("Visualization error:", error);
    // Could show a notification or error message to user
}
</script>

<style>
h2 {
    padding: 0;
    margin: 0;
}
.container {
    font-family: sans-serif;
    max-width: 800px;
    margin: 0 auto;
    font-size: small;
}
.row {
    display: flex;
    gap: 1rem;
    justify-items: stretch;
}
section {
    margin-bottom: 1rem;
    padding: 0.5rem;
    border: 1px solid #ccc;
    width: 100%;
}
.slider {
    margin: 0.5rem 0;
    display: flex;
    justify-content: stretch;
}
.slider label {
    width: 10em;
    display: block;
}
.actions {
    display: flex;
    gap: 1rem;
}
.info {
    margin-left: 1rem;
}

.demo-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-left: 1rem;
    padding: 0.25rem 0.5rem;
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 4px;
}

.demo-label {
    font-size: 0.8rem;
    color: #856404;
    font-weight: 500;
}

.demo-controls button {
    padding: 0.25rem 0.5rem;
    font-size: 0.8rem;
    border: 1px solid #fd7e14;
    background: white;
    color: #fd7e14;
    border-radius: 3px;
    cursor: pointer;
    transition: all 0.2s;
}

.demo-controls button:hover {
    background: #fd7e14;
    color: white;
}

.demo-controls button.active {
    background: #fd7e14;
    color: white;
}

.connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-left: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
}

.status-label {
    color: #666;
    font-weight: 500;
}

.status-indicator {
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    font-weight: 500;
    transition: all 0.2s;
}

.status-indicator.connected {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.status-indicator.connecting {
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
    animation: pulse 1.5s infinite;
}

.status-indicator.disconnected {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.status-indicator.error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
    animation: blink 2s infinite;
}

@keyframes blink {
    0%,
    50% {
        opacity: 1;
    }
    51%,
    100% {
        opacity: 0.5;
    }
}

.visualizer-section {
    margin-bottom: 1rem;
    padding: 1rem;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    background: #fafafa;
}

.visualizer-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

@media (max-width: 768px) {
    .visualizer-row {
        grid-template-columns: 1fr;
    }
}
</style>
