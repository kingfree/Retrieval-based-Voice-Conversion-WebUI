<template>
    <div class="performance-monitor">
        <div class="monitor-header">
            <h4>性能监控</h4>
            <div class="monitor-controls">
                <button @click="resetMetrics" class="reset-btn">重置</button>
            </div>
        </div>

        <div class="metrics-grid">
            <!-- Real-time metrics row -->
            <div class="metrics-row primary">
                <div class="metric-item inference">
                    <div class="metric-label">推理时间</div>
                    <div class="metric-value" :class="getInferenceClass()">
                        {{ inferenceTime.toFixed(1) }}ms
                    </div>
                    <div class="metric-trend">
                        <span class="trend-icon">{{
                            getInferenceTrend()
                        }}</span>
                    </div>
                </div>

                <div class="metric-item algorithm">
                    <div class="metric-label">算法延迟</div>
                    <div class="metric-value" :class="getAlgorithmClass()">
                        {{ algorithmLatency.toFixed(1) }}ms
                    </div>
                    <div class="metric-trend">
                        <span class="trend-icon">{{
                            getAlgorithmTrend()
                        }}</span>
                    </div>
                </div>

                <div class="metric-item total">
                    <div class="metric-label">总延迟</div>
                    <div class="metric-value" :class="getTotalLatencyClass()">
                        {{ totalLatency.toFixed(1) }}ms
                    </div>
                    <div class="metric-trend">
                        <span class="trend-icon">{{
                            getTotalLatencyTrend()
                        }}</span>
                    </div>
                </div>
            </div>

            <!-- Expanded metrics -->
            <div v-show="isExpanded" class="metrics-row secondary">
                <div class="metric-item">
                    <div class="metric-label">缓冲延迟</div>
                    <div class="metric-value">
                        {{ bufferLatency.toFixed(1) }}ms
                    </div>
                </div>

                <div class="metric-item">
                    <div class="metric-label">平均推理</div>
                    <div class="metric-value">
                        {{ averageInferenceTime.toFixed(1) }}ms
                    </div>
                </div>

                <div class="metric-item">
                    <div class="metric-label">最大延迟</div>
                    <div class="metric-value">
                        {{ maxLatency.toFixed(1) }}ms
                    </div>
                </div>
            </div>

            <!-- Performance charts (when expanded) -->
            <div v-show="isExpanded" class="charts-row">
                <div class="chart-container">
                    <div class="chart-title">推理时间趋势</div>
                    <canvas
                        ref="inferenceChart"
                        class="performance-chart"
                        width="200"
                        height="60"
                    ></canvas>
                </div>

                <div class="chart-container">
                    <div class="chart-title">延迟分布</div>
                    <canvas
                        ref="latencyChart"
                        class="performance-chart"
                        width="200"
                        height="60"
                    ></canvas>
                </div>
            </div>

            <!-- Additional stats (when expanded) -->
            <div v-show="isExpanded" class="stats-row">
                <div class="stat-group">
                    <div class="stat-label">音频处理</div>
                    <div class="stat-list">
                        <div class="stat-item">
                            <span>处理帧数:</span>
                            <span>{{ processedFrames }}</span>
                        </div>
                        <div class="stat-item">
                            <span>丢帧数:</span>
                            <span>{{ droppedFrames }}</span>
                        </div>
                        <div class="stat-item">
                            <span>成功率:</span>
                            <span>{{ successRate.toFixed(1) }}%</span>
                        </div>
                    </div>
                </div>

                <div class="stat-group">
                    <div class="stat-label">性能指标</div>
                    <div class="stat-list">
                        <div class="stat-item">
                            <span>实时因子:</span>
                            <span>{{ realtimeFactor.toFixed(2) }}x</span>
                        </div>
                        <div class="stat-item">
                            <span>CPU 使用:</span>
                            <span>{{ cpuUsage.toFixed(1) }}%</span>
                        </div>
                        <div class="stat-item">
                            <span>内存使用:</span>
                            <span>{{ memoryUsage.toFixed(1) }}MB</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from "vue";

const props = defineProps({
    inferenceTime: {
        type: Number,
        default: 0,
    },
    algorithmLatency: {
        type: Number,
        default: 0,
    },
    bufferLatency: {
        type: Number,
        default: 0,
    },
    totalLatency: {
        type: Number,
        default: 0,
    },
    isStreaming: {
        type: Boolean,
        default: false,
    },
});

const emit = defineEmits(["reset"]);

// Component state
const isExpanded = ref(true);
const processedFrames = ref(0);
const droppedFrames = ref(0);

// Historical data for trends and averages
const inferenceHistory = ref([]);
const algorithmHistory = ref([]);
const totalLatencyHistory = ref([]);

// Chart references
const inferenceChart = ref(null);
const latencyChart = ref(null);

// Chart contexts
let inferenceCtx = null;
let latencyCtx = null;

// Computed metrics
const averageInferenceTime = computed(() => {
    if (inferenceHistory.value.length === 0) return 0;
    const sum = inferenceHistory.value.reduce((a, b) => a + b, 0);
    return sum / inferenceHistory.value.length;
});

const maxLatency = computed(() => {
    if (totalLatencyHistory.value.length === 0) return 0;
    return Math.max(...totalLatencyHistory.value);
});

const successRate = computed(() => {
    const total = processedFrames.value + droppedFrames.value;
    if (total === 0) return 100;
    return (processedFrames.value / total) * 100;
});

const realtimeFactor = computed(() => {
    if (props.algorithmLatency === 0) return 0;
    return (props.bufferLatency || 22.67) / props.algorithmLatency; // Assume ~22.67ms buffer at 44.1kHz
});

const cpuUsage = computed(() => {
    // Estimated based on inference time
    return Math.min(100, (props.inferenceTime / 100) * 100);
});

const memoryUsage = computed(() => {
    // Estimated memory usage in MB
    return 50 + processedFrames.value * 0.001;
});

// Performance classification methods
const getInferenceClass = () => {
    if (props.inferenceTime < 20) return "excellent";
    if (props.inferenceTime < 50) return "good";
    if (props.inferenceTime < 100) return "fair";
    return "poor";
};

const getAlgorithmClass = () => {
    if (props.algorithmLatency < 50) return "excellent";
    if (props.algorithmLatency < 100) return "good";
    if (props.algorithmLatency < 200) return "fair";
    return "poor";
};

const getTotalLatencyClass = () => {
    if (props.totalLatency < 100) return "excellent";
    if (props.totalLatency < 200) return "good";
    if (props.totalLatency < 300) return "fair";
    return "poor";
};

// Trend calculation methods
const getTrend = (history, current) => {
    if (history.length < 2) return "→";
    const recent =
        history.slice(-5).reduce((a, b) => a + b, 0) /
        Math.min(5, history.length);
    const older = history.slice(-10, -5);
    if (older.length === 0) return "→";
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;

    if (recent > olderAvg * 1.1) return "↗";
    if (recent < olderAvg * 0.9) return "↘";
    return "→";
};

const getInferenceTrend = () =>
    getTrend(inferenceHistory.value, props.inferenceTime);
const getAlgorithmTrend = () =>
    getTrend(algorithmHistory.value, props.algorithmLatency);
const getTotalLatencyTrend = () =>
    getTrend(totalLatencyHistory.value, props.totalLatency);

// Chart drawing methods
const drawInferenceChart = () => {
    if (!inferenceCtx || inferenceHistory.value.length < 2) return;

    const canvas = inferenceChart.value;
    const width = canvas.width;
    const height = canvas.height;

    inferenceCtx.clearRect(0, 0, width, height);

    const data = inferenceHistory.value.slice(-50); // Last 50 data points
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = max - min || 1;

    // Draw grid
    inferenceCtx.strokeStyle = "#f0f0f0";
    inferenceCtx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = (height / 4) * i;
        inferenceCtx.beginPath();
        inferenceCtx.moveTo(0, y);
        inferenceCtx.lineTo(width, y);
        inferenceCtx.stroke();
    }

    // Draw line
    inferenceCtx.strokeStyle = "#2196F3";
    inferenceCtx.lineWidth = 2;
    inferenceCtx.beginPath();

    data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((value - min) / range) * height;

        if (index === 0) {
            inferenceCtx.moveTo(x, y);
        } else {
            inferenceCtx.lineTo(x, y);
        }
    });

    inferenceCtx.stroke();
};

const drawLatencyChart = () => {
    if (!latencyCtx || totalLatencyHistory.value.length < 2) return;

    const canvas = latencyChart.value;
    const width = canvas.width;
    const height = canvas.height;

    latencyCtx.clearRect(0, 0, width, height);

    const data = totalLatencyHistory.value.slice(-50);
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = max - min || 1;

    // Draw area chart
    const gradient = latencyCtx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, "rgba(255, 152, 0, 0.3)");
    gradient.addColorStop(1, "rgba(255, 152, 0, 0.1)");

    latencyCtx.fillStyle = gradient;
    latencyCtx.beginPath();
    latencyCtx.moveTo(0, height);

    data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((value - min) / range) * height;
        latencyCtx.lineTo(x, y);
    });

    latencyCtx.lineTo(width, height);
    latencyCtx.closePath();
    latencyCtx.fill();

    // Draw line
    latencyCtx.strokeStyle = "#FF9800";
    latencyCtx.lineWidth = 2;
    latencyCtx.beginPath();

    data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((value - min) / range) * height;

        if (index === 0) {
            latencyCtx.moveTo(x, y);
        } else {
            latencyCtx.lineTo(x, y);
        }
    });

    latencyCtx.stroke();
};

// Event handlers
const toggleExpanded = () => {
    isExpanded.value = !isExpanded.value;
    if (isExpanded.value) {
        nextTick(() => {
            initializeCharts();
        });
    }
};

const resetMetrics = () => {
    inferenceHistory.value = [];
    algorithmHistory.value = [];
    totalLatencyHistory.value = [];
    processedFrames.value = 0;
    droppedFrames.value = 0;
    emit("reset");
};

const initializeCharts = () => {
    if (inferenceChart.value && !inferenceCtx) {
        inferenceCtx = inferenceChart.value.getContext("2d");
    }
    if (latencyChart.value && !latencyCtx) {
        latencyCtx = latencyChart.value.getContext("2d");
    }
};

// Watch for metric updates
watch(
    () => props.inferenceTime,
    (newValue) => {
        if (newValue > 0 && props.isStreaming) {
            inferenceHistory.value.push(newValue);
            if (inferenceHistory.value.length > 100) {
                inferenceHistory.value.shift();
            }
            processedFrames.value++;

            if (isExpanded.value) {
                drawInferenceChart();
            }
        }
    },
);

watch(
    () => props.algorithmLatency,
    (newValue) => {
        if (newValue > 0 && props.isStreaming) {
            algorithmHistory.value.push(newValue);
            if (algorithmHistory.value.length > 100) {
                algorithmHistory.value.shift();
            }
        }
    },
);

watch(
    () => props.totalLatency,
    (newValue) => {
        if (newValue > 0 && props.isStreaming) {
            totalLatencyHistory.value.push(newValue);
            if (totalLatencyHistory.value.length > 100) {
                totalLatencyHistory.value.shift();
            }

            if (isExpanded.value) {
                drawLatencyChart();
            }
        }
    },
);

onMounted(() => {
    initializeCharts();
});

onUnmounted(() => {
    inferenceCtx = null;
    latencyCtx = null;
});
</script>

<style scoped>
.performance-monitor {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    background: white;
    margin: 8px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.monitor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.monitor-header h4 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #333;
}

.monitor-controls {
    display: flex;
    gap: 8px;
}

.expand-btn,
.reset-btn {
    padding: 4px 8px;
    font-size: 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    transition: all 0.2s;
}

.expand-btn:hover,
.reset-btn:hover {
    background: #f5f5f5;
}

.metrics-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.metrics-row {
    display: flex;
    gap: 12px;
    justify-content: space-between;
}

.metrics-row.primary {
    border-bottom: 1px solid #f0f0f0;
    padding-bottom: 12px;
}

.metric-item {
    flex: 1;
    text-align: center;
    padding: 8px;
    border-radius: 6px;
    background: #fafafa;
    min-width: 80px;
}

.metric-item.inference {
    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
}

.metric-item.algorithm {
    background: linear-gradient(135deg, #fff3e0 0%, #f3e5f5 100%);
}

.metric-item.total {
    background: linear-gradient(135deg, #e8f5e8 0%, #f3e5f5 100%);
}

.metric-label {
    font-size: 11px;
    color: #666;
    margin-bottom: 4px;
    font-weight: 500;
}

.metric-value {
    font-size: 16px;
    font-weight: bold;
    font-family: monospace;
    margin-bottom: 2px;
}

.metric-value.excellent {
    color: #4caf50;
}
.metric-value.good {
    color: #ff9800;
}
.metric-value.fair {
    color: #ff5722;
}
.metric-value.poor {
    color: #f44336;
}

.metric-trend {
    font-size: 12px;
}

.trend-icon {
    display: inline-block;
    font-weight: bold;
    color: #666;
}

.charts-row {
    display: flex;
    gap: 12px;
    margin-top: 8px;
}

.chart-container {
    flex: 1;
    text-align: center;
}

.chart-title {
    font-size: 12px;
    color: #666;
    margin-bottom: 4px;
}

.performance-chart {
    width: 100%;
    height: 60px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background: #fafafa;
}

.stats-row {
    display: flex;
    gap: 20px;
    margin-top: 8px;
}

.stat-group {
    flex: 1;
}

.stat-label {
    font-size: 12px;
    font-weight: 600;
    color: #333;
    margin-bottom: 8px;
}

.stat-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.stat-item {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #666;
}

.stat-item span:first-child {
    color: #888;
}

.stat-item span:last-child {
    font-weight: 500;
    color: #333;
    font-family: monospace;
}

/* Responsive design */
@media (max-width: 768px) {
    .metrics-row {
        flex-direction: column;
        gap: 8px;
    }

    .charts-row {
        flex-direction: column;
    }

    .stats-row {
        flex-direction: column;
        gap: 12px;
    }
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
    .performance-monitor {
        background: #2d2d2d;
        border-color: #555;
        color: #e0e0e0;
    }

    .monitor-header h4 {
        color: #e0e0e0;
    }

    .expand-btn,
    .reset-btn {
        background: #3d3d3d;
        border-color: #555;
        color: #e0e0e0;
    }

    .metric-item {
        background: #1e1e1e;
    }

    .performance-chart {
        background: #1e1e1e;
        border-color: #555;
    }
}
</style>
