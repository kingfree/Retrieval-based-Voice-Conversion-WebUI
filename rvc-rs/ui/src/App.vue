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
            </section>
        </div>
        <div class="actions">
            <button type="button" @click="startVc">开始音频转换</button>
            <button type="button" @click="stopVc">停止音频转换</button>
        </div>
    </div>
</template>

<script setup>
import { ref, watch } from "vue";
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

const blockTime = ref(0.25);
const crossfadeLength = ref(0.05);

function send(event, value) {
    invoke("frontend_event", { event, value: value?.toString() });
}

watch(threshold, (v) => send("threshold", v));
watch(pitch, (v) => send("pitch", v));
watch(formant, (v) => send("formant", v));
watch(blockTime, (v) => send("block_time", v));
watch(crossfadeLength, (v) => send("crossfade_length", v));

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
</style>
