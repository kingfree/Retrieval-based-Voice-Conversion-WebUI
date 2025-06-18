<template>
  <div class="waveform-container">
    <div class="waveform-header">
      <h3>{{ title }}</h3>
      <div class="waveform-controls">
        <span class="level-indicator" :class="{ active: isActive }">
          {{ isActive ? '🎵' : '🔇' }}
        </span>
        <span class="volume-level">{{ volumeLevel.toFixed(1) }}dB</span>
      </div>
    </div>
    <div
      :id="containerId"
      class="waveform-display"
      :style="{ height: height + 'px' }"
    ></div>
    <canvas
      ref="canvas"
      class="realtime-canvas"
      :width="canvasWidth"
      :height="canvasHeight"
      v-show="showRealtime"
    ></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import WaveSurfer from 'wavesurfer.js'

const props = defineProps({
  title: {
    type: String,
    default: '音频波形'
  },
  height: {
    type: Number,
    default: 128
  },
  waveColor: {
    type: String,
    default: '#1976d2'
  },
  progressColor: {
    type: String,
    default: '#42a5f5'
  },
  backgroundColor: {
    type: String,
    default: '#f5f5f5'
  },
  audioData: {
    type: Array,
    default: () => []
  },
  isRealtime: {
    type: Boolean,
    default: false
  },
  sampleRate: {
    type: Number,
    default: 44100
  }
})

const emit = defineEmits(['ready', 'finish', 'error'])

const containerId = ref(`waveform-${Math.random().toString(36).substr(2, 9)}`)
const canvas = ref(null)
const wavesurfer = ref(null)
const isActive = ref(false)
const volumeLevel = ref(-60)
const showRealtime = ref(true)
const canvasWidth = ref(800)
const canvasHeight = ref(128)

// Real-time waveform drawing
let animationFrame = null
let audioBuffer = []
const bufferSize = 2048
let canvasContext = null

onMounted(async () => {
  await nextTick()
  initWaveSurfer()
  if (props.isRealtime) {
    initRealtimeCanvas()
  }
})

onUnmounted(() => {
  if (wavesurfer.value) {
    wavesurfer.value.destroy()
  }
  if (animationFrame) {
    cancelAnimationFrame(animationFrame)
  }
})

function initWaveSurfer() {
  try {
    wavesurfer.value = WaveSurfer.create({
      container: `#${containerId.value}`,
      waveColor: props.waveColor,
      progressColor: props.progressColor,
      backgroundColor: props.backgroundColor,
      height: props.height,
      normalize: true,
      responsive: true,
      hideScrollbar: true,
      cursorWidth: 0,
      interact: false
    })

    wavesurfer.value.on('ready', () => {
      emit('ready')
    })

    wavesurfer.value.on('finish', () => {
      emit('finish')
    })

    wavesurfer.value.on('error', (error) => {
      console.error('WaveSurfer error:', error)
      emit('error', error)
    })

  } catch (error) {
    console.error('Failed to initialize WaveSurfer:', error)
    emit('error', error)
  }
}

function initRealtimeCanvas() {
  if (canvas.value) {
    canvasContext = canvas.value.getContext('2d')
    canvasWidth.value = canvas.value.offsetWidth || 800
    canvasHeight.value = props.height
    startRealtimeDrawing()
  }
}

function startRealtimeDrawing() {
  const draw = () => {
    if (!canvasContext || !canvas.value) return

    const width = canvasWidth.value
    const height = canvasHeight.value

    // Clear canvas
    canvasContext.fillStyle = props.backgroundColor
    canvasContext.fillRect(0, 0, width, height)

    if (audioBuffer.length > 0) {
      // Draw waveform
      canvasContext.strokeStyle = props.waveColor
      canvasContext.lineWidth = 1
      canvasContext.beginPath()

      const sliceWidth = width / audioBuffer.length
      let x = 0

      for (let i = 0; i < audioBuffer.length; i++) {
        const v = audioBuffer[i] * 0.5 + 0.5
        const y = v * height

        if (i === 0) {
          canvasContext.moveTo(x, y)
        } else {
          canvasContext.lineTo(x, y)
        }

        x += sliceWidth
      }

      canvasContext.stroke()

      // Calculate volume level
      const rms = Math.sqrt(
        audioBuffer.reduce((sum, val) => sum + val * val, 0) / audioBuffer.length
      )
      volumeLevel.value = rms > 0 ? 20 * Math.log10(rms) : -60
      isActive.value = volumeLevel.value > -50
    }

    animationFrame = requestAnimationFrame(draw)
  }

  draw()
}

// Watch for audio data changes
watch(() => props.audioData, (newData) => {
  if (props.isRealtime && newData && newData.length > 0) {
    // Update real-time buffer
    audioBuffer = [...newData]
    if (audioBuffer.length > bufferSize) {
      audioBuffer = audioBuffer.slice(-bufferSize)
    }
  } else if (!props.isRealtime && wavesurfer.value && newData && newData.length > 0) {
    // Load static audio data
    try {
      // Convert array to AudioBuffer for WaveSurfer
      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const buffer = audioContext.createBuffer(1, newData.length, props.sampleRate)
      const channelData = buffer.getChannelData(0)

      for (let i = 0; i < newData.length; i++) {
        channelData[i] = newData[i]
      }

      wavesurfer.value.loadDecodedBuffer(buffer)
    } catch (error) {
      console.error('Error loading audio data:', error)
    }
  }
}, { deep: true })

// Public methods
const loadAudioFile = (file) => {
  if (wavesurfer.value && file) {
    wavesurfer.value.loadBlob(file)
  }
}

const loadAudioUrl = (url) => {
  if (wavesurfer.value && url) {
    wavesurfer.value.load(url)
  }
}

const play = () => {
  if (wavesurfer.value) {
    wavesurfer.value.play()
  }
}

const pause = () => {
  if (wavesurfer.value) {
    wavesurfer.value.pause()
  }
}

const stop = () => {
  if (wavesurfer.value) {
    wavesurfer.value.stop()
  }
}

const clear = () => {
  if (wavesurfer.value) {
    wavesurfer.value.empty()
  }
  audioBuffer = []
  volumeLevel.value = -60
  isActive.value = false
}

// Expose methods
defineExpose({
  loadAudioFile,
  loadAudioUrl,
  play,
  pause,
  stop,
  clear
})
</script>

<style scoped>
.waveform-container {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 12px;
  margin: 8px 0;
  background: white;
}

.waveform-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.waveform-header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
  color: #333;
}

.waveform-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.level-indicator {
  font-size: 16px;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.level-indicator.active {
  opacity: 1;
  animation: pulse 1s infinite;
}

.volume-level {
  font-size: 12px;
  color: #666;
  font-family: monospace;
  min-width: 50px;
  text-align: right;
}

.waveform-display {
  border-radius: 4px;
  overflow: hidden;
  background: #f8f9fa;
}

.realtime-canvas {
  width: 100%;
  border-radius: 4px;
  background: #f8f9fa;
}

@keyframes pulse {
  0%, 50%, 100% {
    opacity: 1;
  }
  25%, 75% {
    opacity: 0.5;
  }
}

/* Dark theme support */
@media (prefers-color-scheme: dark) {
  .waveform-container {
    background: #2d2d2d;
    border-color: #555;
  }

  .waveform-header h3 {
    color: #e0e0e0;
  }

  .volume-level {
    color: #aaa;
  }

  .waveform-display,
  .realtime-canvas {
    background: #1e1e1e;
  }
}
</style>
