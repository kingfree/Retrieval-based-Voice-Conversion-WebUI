<template>
    <div class="audio-visualizer">
        <div class="visualizer-header">
            <h3>{{ title }}</h3>
            <div class="visualizer-controls">
                <button @click="toggleVisualization" class="toggle-btn">
                    {{ isActive ? "暂停" : "开始" }}
                </button>
                <div class="view-mode">
                    <label>
                        <input
                            type="radio"
                            v-model="viewMode"
                            value="waveform"
                        />
                        波形
                    </label>
                    <label>
                        <input
                            type="radio"
                            v-model="viewMode"
                            value="spectrum"
                        />
                        频谱
                    </label>
                    <label>
                        <input type="radio" v-model="viewMode" value="both" />
                        双视图
                    </label>
                </div>
                <div class="level-meter">
                    <span class="level-label">音量:</span>
                    <div class="level-bar">
                        <div
                            class="level-fill"
                            :style="{
                                width: levelPercentage + '%',
                                backgroundColor: levelColor,
                            }"
                        ></div>
                    </div>
                    <span class="level-value"
                        >{{ volumeLevel.toFixed(1) }}dB</span
                    >
                </div>
            </div>
        </div>

        <div
            class="visualizer-content"
            :class="{ 'dual-view': viewMode === 'both' }"
        >
            <!-- Waveform Display -->
            <div
                v-show="viewMode === 'waveform' || viewMode === 'both'"
                class="waveform-container"
            >
                <div class="visualization-title">时域波形</div>
                <canvas
                    ref="waveformCanvas"
                    class="visualization-canvas"
                    :width="canvasWidth"
                    :height="waveformHeight"
                ></canvas>
            </div>

            <!-- Spectrum Display -->
            <div
                v-show="viewMode === 'spectrum' || viewMode === 'both'"
                class="spectrum-container"
            >
                <div class="visualization-title">频域谱图</div>
                <canvas
                    ref="spectrumCanvas"
                    class="visualization-canvas"
                    :width="canvasWidth"
                    :height="spectrumHeight"
                ></canvas>
                <div class="frequency-labels">
                    <span>0Hz</span>
                    <span>1kHz</span>
                    <span>5kHz</span>
                    <span>10kHz</span>
                    <span>20kHz</span>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick, computed } from "vue";

const props = defineProps({
    title: {
        type: String,
        default: "音频可视化",
    },
    audioData: {
        type: Array,
        default: () => [],
    },
    sampleRate: {
        type: Number,
        default: 44100,
    },
    theme: {
        type: String,
        default: "light", // 'light' or 'dark'
        validator: (value) => ["light", "dark"].includes(value),
    },
    primaryColor: {
        type: String,
        default: "#2196F3",
    },
    accentColor: {
        type: String,
        default: "#FF9800",
    },
});

const emit = defineEmits(["levelChange", "error"]);

// Component state
const isActive = ref(true);
const viewMode = ref("both");
const volumeLevel = ref(-60);
const canvasWidth = ref(800);
const waveformHeight = ref(120);
const spectrumHeight = ref(120);
const hasError = ref(false);
const errorMessage = ref("");
const lastUpdateTime = ref(0);

// Canvas refs
const waveformCanvas = ref(null);
const spectrumCanvas = ref(null);

// Animation and processing
let animationFrame = null;
let waveformContext = null;
let spectrumContext = null;

// Audio processing
const bufferSize = 2048;
const fftSize = 1024;
let audioBuffer = [];
let frequencyData = [];

// Computed properties
const levelPercentage = computed(() => {
    return Math.max(0, Math.min(100, ((volumeLevel.value + 60) * 100) / 60));
});

const levelColor = computed(() => {
    if (volumeLevel.value < -40) return "#4CAF50"; // Green for low levels
    if (volumeLevel.value < -20) return "#FF9800"; // Orange for medium levels
    return "#F44336"; // Red for high levels
});

const isDarkTheme = computed(() => props.theme === "dark");

onMounted(async () => {
    await nextTick();
    initializeCanvases();
    if (isActive.value) {
        startVisualization();
    }
});

onUnmounted(() => {
    stopVisualization();
});

function initializeCanvases() {
    // Initialize waveform canvas
    if (waveformCanvas.value) {
        waveformContext = waveformCanvas.value.getContext("2d");
        canvasWidth.value = waveformCanvas.value.offsetWidth || 800;
    }

    // Initialize spectrum canvas
    if (spectrumCanvas.value) {
        spectrumContext = spectrumCanvas.value.getContext("2d");
    }
}

function startVisualization() {
    if (animationFrame) return;

    const draw = () => {
        try {
            if (!isActive.value || (!waveformContext && !spectrumContext)) {
                return;
            }

            // Draw waveform
            if (
                waveformContext &&
                (viewMode.value === "waveform" || viewMode.value === "both")
            ) {
                drawWaveform();
            }

            // Draw spectrum
            if (
                spectrumContext &&
                (viewMode.value === "spectrum" || viewMode.value === "both")
            ) {
                drawSpectrum();
            }

            // Clear any previous errors
            if (hasError.value) {
                hasError.value = false;
                errorMessage.value = "";
            }

            animationFrame = requestAnimationFrame(draw);
        } catch (error) {
            console.error("Error in visualization draw loop:", error);
            hasError.value = true;
            errorMessage.value = error.message;
            emit("error", error);
        }
    };

    draw();
}

function stopVisualization() {
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
}

function drawWaveform() {
    try {
        const width = canvasWidth.value;
        const height = waveformHeight.value;
        const ctx = waveformContext;

        if (!ctx) {
            throw new Error("Waveform canvas context not available");
        }

        // Clear canvas
        ctx.fillStyle = isDarkTheme.value ? "#1e1e1e" : "#f8f9fa";
        ctx.fillRect(0, 0, width, height);

        if (hasError.value) {
            drawNoSignalMessage(
                ctx,
                width,
                height,
                `错误: ${errorMessage.value}`,
            );
            return;
        }

        if (audioBuffer.length === 0) {
            drawNoSignalMessage(ctx, width, height, "无音频信号");
            return;
        }

        // Draw grid
        drawGrid(ctx, width, height);

        // Draw waveform
        ctx.strokeStyle = props.primaryColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const sliceWidth = width / audioBuffer.length;
        let x = 0;

        for (let i = 0; i < audioBuffer.length; i++) {
            const amplitude = audioBuffer[i];
            const y = (amplitude * 0.5 + 0.5) * height;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        ctx.stroke();

        // Draw center line
        ctx.strokeStyle = isDarkTheme.value ? "#444" : "#ccc";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw amplitude labels
        drawAmplitudeLabels(ctx, width, height);
    } catch (error) {
        console.error("Error drawing waveform:", error);
        hasError.value = true;
        errorMessage.value = error.message;
        emit("error", error);
    }
}

function drawSpectrum() {
    try {
        const width = canvasWidth.value;
        const height = spectrumHeight.value;
        const ctx = spectrumContext;

        if (!ctx) {
            throw new Error("Spectrum canvas context not available");
        }

        // Clear canvas
        ctx.fillStyle = isDarkTheme.value ? "#1e1e1e" : "#f8f9fa";
        ctx.fillRect(0, 0, width, height);

        if (hasError.value) {
            drawNoSignalMessage(
                ctx,
                width,
                height,
                `错误: ${errorMessage.value}`,
            );
            return;
        }

        if (frequencyData.length === 0) {
            drawNoSignalMessage(ctx, width, height, "无频谱数据");
            return;
        }

        // Draw frequency bars
        const barWidth = width / frequencyData.length;
        const maxBarHeight = height * 0.8;

        for (let i = 0; i < frequencyData.length; i++) {
            const barHeight = (frequencyData[i] || 0) * maxBarHeight;
            const x = i * barWidth;
            const y = height - barHeight;

            // Create gradient
            const gradient = ctx.createLinearGradient(0, y, 0, height);
            gradient.addColorStop(0, props.accentColor);
            gradient.addColorStop(1, props.primaryColor);

            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth - 1, barHeight);
        }

        // Draw frequency grid
        drawFrequencyGrid(ctx, width, height);
    } catch (error) {
        console.error("Error drawing spectrum:", error);
        hasError.value = true;
        errorMessage.value = error.message;
        emit("error", error);
    }
}

function drawGrid(ctx, width, height) {
    ctx.strokeStyle = isDarkTheme.value ? "#333" : "#e0e0e0";
    ctx.lineWidth = 0.5;
    ctx.beginPath();

    // Horizontal lines
    for (let i = 1; i < 4; i++) {
        const y = (height / 4) * i;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
    }

    // Vertical lines
    for (let i = 1; i < 8; i++) {
        const x = (width / 8) * i;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
    }

    ctx.stroke();
}

function drawFrequencyGrid(ctx, width, height) {
    ctx.strokeStyle = isDarkTheme.value ? "#333" : "#e0e0e0";
    ctx.lineWidth = 0.5;
    ctx.beginPath();

    // Horizontal lines (dB levels)
    for (let i = 1; i < 5; i++) {
        const y = (height / 5) * i;
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
    }

    ctx.stroke();
}

function drawAmplitudeLabels(ctx, width, height) {
    ctx.fillStyle = isDarkTheme.value ? "#aaa" : "#666";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";

    ctx.fillText("+1.0", width - 5, 12);
    ctx.fillText("0.0", width - 5, height / 2 + 4);
    ctx.fillText("-1.0", width - 5, height - 4);
}

function drawNoSignalMessage(ctx, width, height, message) {
    ctx.fillStyle = isDarkTheme.value ? "#666" : "#999";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(message, width / 2, height / 2);
}

function calculateFrequencyData(samples) {
    if (samples.length < fftSize) return [];

    // Simple frequency analysis (in production, use proper FFT)
    const bins = 32;
    const binSize = Math.floor(fftSize / bins);
    const freqData = [];

    for (let i = 0; i < bins; i++) {
        const start = i * binSize;
        const end = Math.min(start + binSize, samples.length);
        const binSamples = samples.slice(start, end);

        if (binSamples.length > 0) {
            // Calculate RMS for this frequency bin
            const rms = Math.sqrt(
                binSamples.reduce((sum, sample) => sum + sample * sample, 0) /
                    binSamples.length,
            );
            freqData.push(Math.min(1, rms * 2)); // Normalize and scale
        } else {
            freqData.push(0);
        }
    }

    return freqData;
}

function calculateVolumeLevel(samples) {
    if (samples.length === 0) return -60;

    const rms = Math.sqrt(
        samples.reduce((sum, sample) => sum + sample * sample, 0) /
            samples.length,
    );

    return rms > 0 ? Math.max(-60, 20 * Math.log10(rms)) : -60;
}

function toggleVisualization() {
    isActive.value = !isActive.value;
    if (isActive.value) {
        startVisualization();
    } else {
        stopVisualization();
    }
}

// Watch for audio data changes
watch(
    () => props.audioData,
    (newData) => {
        try {
            if (!newData || newData.length === 0) {
                audioBuffer = [];
                frequencyData = [];
                volumeLevel.value = -60;
                return;
            }

            // Validate audio data
            if (!Array.isArray(newData)) {
                throw new Error("Audio data must be an array");
            }

            // Update audio buffer
            audioBuffer = [...newData];
            if (audioBuffer.length > bufferSize) {
                audioBuffer = audioBuffer.slice(-bufferSize);
            }

            // Calculate frequency data
            frequencyData = calculateFrequencyData(audioBuffer);

            // Calculate volume level
            const newVolumeLevel = calculateVolumeLevel(audioBuffer);
            volumeLevel.value = newVolumeLevel;

            // Update last update time
            lastUpdateTime.value = Date.now();

            // Clear any previous errors
            if (hasError.value) {
                hasError.value = false;
                errorMessage.value = "";
            }

            // Emit level change
            emit("levelChange", newVolumeLevel);
        } catch (error) {
            console.error("Error processing audio data:", error);
            hasError.value = true;
            errorMessage.value = error.message;
            emit("error", error);
        }
    },
    { deep: true },
);

// Watch for view mode changes
watch(viewMode, () => {
    if (isActive.value) {
        // Restart visualization to accommodate new view mode
        stopVisualization();
        nextTick(() => {
            startVisualization();
        });
    }
});

// Public methods
defineExpose({
    start: startVisualization,
    stop: stopVisualization,
    clear: () => {
        audioBuffer = [];
        frequencyData = [];
        volumeLevel.value = -60;
        hasError.value = false;
        errorMessage.value = "";
        lastUpdateTime.value = 0;
    },
    getStatus: () => ({
        isActive: isActive.value,
        hasError: hasError.value,
        errorMessage: errorMessage.value,
        lastUpdateTime: lastUpdateTime.value,
        bufferLength: audioBuffer.length,
    }),
});
</script>

<style scoped>
.audio-visualizer {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    background: white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.visualizer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 12px;
}

.visualizer-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: #333;
}

.visualizer-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}

.toggle-btn {
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
}

.toggle-btn:hover {
    background: #f5f5f5;
}

.view-mode {
    display: flex;
    gap: 8px;
    font-size: 12px;
}

.view-mode label {
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
}

.level-meter {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}

.level-label {
    color: #666;
}

.level-bar {
    width: 80px;
    height: 8px;
    background: #f0f0f0;
    border-radius: 4px;
    overflow: hidden;
}

.level-fill {
    height: 100%;
    transition: width 0.1s ease-out;
    border-radius: 4px;
}

.level-value {
    font-family: monospace;
    color: #666;
    min-width: 45px;
    text-align: right;
}

.visualizer-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.visualizer-content.dual-view {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.waveform-container,
.spectrum-container {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 8px;
    background: #fafafa;
}

.visualization-title {
    font-size: 12px;
    font-weight: 500;
    color: #666;
    margin-bottom: 8px;
    text-align: center;
}

.visualization-canvas {
    width: 100%;
    border-radius: 4px;
    background: #f8f9fa;
}

.frequency-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 10px;
    color: #666;
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
    .audio-visualizer {
        background: #2d2d2d;
        border-color: #555;
        color: #e0e0e0;
    }

    .visualizer-header h3 {
        color: #e0e0e0;
    }

    .toggle-btn {
        background: #3d3d3d;
        border-color: #555;
        color: #e0e0e0;
    }

    .toggle-btn:hover {
        background: #4d4d4d;
    }

    .waveform-container,
    .spectrum-container {
        background: #1e1e1e;
        border-color: #555;
    }

    .level-bar {
        background: #404040;
    }
}

/* Responsive design */
@media (max-width: 768px) {
    .visualizer-header {
        flex-direction: column;
        align-items: stretch;
    }

    .visualizer-controls {
        justify-content: space-between;
    }

    .visualizer-content.dual-view {
        grid-template-columns: 1fr;
    }

    .view-mode {
        justify-content: center;
    }
}

@media (max-width: 480px) {
    .audio-visualizer {
        padding: 12px;
    }

    .level-meter {
        flex-direction: column;
        gap: 4px;
    }

    .level-bar {
        width: 100%;
    }
}
</style>
