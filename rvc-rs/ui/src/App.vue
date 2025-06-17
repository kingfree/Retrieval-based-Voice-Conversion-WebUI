<template>
    <div class="container">
        <h1>RVC - GUI</h1>
        <div class="row">
            <section>
                <h2>加载模型</h2>
                <div>
                    <label>pth 文件:</label>
                    <input
                        type="file"
                        accept=".pth"
                        @change="(e) => (pth = e.target.files[0]?.name || '')"
                    />
                </div>
                <div>
                    <label>index 文件:</label>
                    <input
                        type="file"
                        accept=".index"
                        @change="(e) => (index = e.target.files[0]?.name || '')"
                    />
                </div>
            </section>

            <section>
                <h2>音频设备</h2>
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
                    <button type="button" @click="reloadDevices">
                        重载设备列表
                    </button>
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
                    <label><input type="radio" value="pm" v-model="f0method" />pm</label>
                    <label><input type="radio" value="harvest" v-model="f0method" />harvest</label>
                    <label><input type="radio" value="crepe" v-model="f0method" />crepe</label>
                    <label><input type="radio" value="rmvpe" v-model="f0method" />rmvpe</label>
                    <label><input type="radio" value="fcpe" v-model="f0method" />fcpe</label>
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
                    <label><input type="checkbox" v-model="iNoiseReduce" /> 输入降噪</label>
                    <label><input type="checkbox" v-model="oNoiseReduce" /> 输出降噪</label>
                    <label><input type="checkbox" v-model="usePv" /> 启用相位声码器</label>
                </div>
            </section>
        </div>
        <div class="actions">
            <button type="button" @click="startVc">开始音频转换</button>
            <button type="button" @click="stopVc">停止音频转换</button>
            <label><input type="radio" value="im" v-model="functionMode" /> 输入监听</label>
            <label><input type="radio" value="vc" v-model="functionMode" /> 输出变声</label>
            <span class="info">算法延迟: {{ delayTime }} ms</span>
            <span class="info">推理时间: {{ inferTime }} ms</span>
        </div>
    </div>
</template>

<script setup>
import { ref, watch, onMounted } from "vue";
import { invoke } from "@tauri-apps/api/core";

const hostapis = ["Default"];
const inputDevices = ["Input"];
const outputDevices = ["Output"];

let pth = "";
let index = "";
const hostapi = ref(hostapis[0]);
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

function send(event, value) {
    invoke("frontend_event", { event, value: value?.toString() });
}

async function loadConfig() {
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
    } catch (e) {
        console.error("failed to load config", e);
    }
}

onMounted(loadConfig);

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
watch(iNoiseReduce, (v) => send("I_noise_reduce", v));
watch(oNoiseReduce, (v) => send("O_noise_reduce", v));
watch(usePv, (v) => send("use_pv", v));
watch(functionMode, (v) => send("function_mode", v));

function reloadDevices() {
    send("reload_devices");
}

function startVc() {
    send("start_vc");
}

function stopVc() {
    send("stop_vc");
}
</script>

<style>
.container {
    font-family: sans-serif;
    max-width: 800px;
    margin: 0 auto;
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
}
.actions {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.info {
    margin-left: 1rem;
}
</style>
