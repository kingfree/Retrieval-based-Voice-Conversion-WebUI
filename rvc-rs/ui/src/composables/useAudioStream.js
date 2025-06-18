import { ref, reactive, onUnmounted } from "vue";
import { listen } from "@tauri-apps/api/event";

export function useAudioStream() {
  // Audio data buffers
  const inputAudioData = ref([]);
  const outputAudioData = ref([]);

  // Stream state
  const isStreaming = ref(false);
  const inputVolume = ref(-60);
  const outputVolume = ref(-60);
  const lastError = ref(null);
  const connectionStatus = ref("disconnected");

  // Performance metrics
  const inferenceTime = ref(0);
  const algorithmLatency = ref(0);
  const bufferLatency = ref(0);
  const totalLatency = ref(0);

  // Audio processing stats
  const stats = reactive({
    inputSampleRate: 44100,
    outputSampleRate: 44100,
    bufferSize: 2048,
    processedSamples: 0,
    droppedFrames: 0,
    latency: 0,
  });

  // Event listeners
  let unlistenInputAudio = null;
  let unlistenOutputAudio = null;
  let unlistenStats = null;
  let unlistenPerformanceMetrics = null;

  // Buffer management
  const maxBufferSize = 8192;
  let inputBuffer = [];
  let outputBuffer = [];

  const initializeAudioStream = async () => {
    try {
      lastError.value = null;
      connectionStatus.value = "connecting";

      // Listen for input audio data
      unlistenInputAudio = await listen("input_audio_data", (event) => {
        try {
          const audioData = event.payload;
          if (audioData && Array.isArray(audioData.samples)) {
            updateInputBuffer(audioData.samples);
            calculateInputVolume(audioData.samples);
          } else {
            console.warn("Invalid input audio data received:", audioData);
          }
        } catch (error) {
          console.error("Error processing input audio data:", error);
          lastError.value = error.message;
        }
      });

      // Listen for output audio data
      unlistenOutputAudio = await listen("output_audio_data", (event) => {
        try {
          const audioData = event.payload;
          if (audioData && Array.isArray(audioData.samples)) {
            updateOutputBuffer(audioData.samples);
            calculateOutputVolume(audioData.samples);
          } else {
            console.warn("Invalid output audio data received:", audioData);
          }
        } catch (error) {
          console.error("Error processing output audio data:", error);
          lastError.value = error.message;
        }
      });

      // Listen for audio processing stats
      unlistenStats = await listen("audio_stats", (event) => {
        try {
          const newStats = event.payload;
          if (newStats && typeof newStats === "object") {
            Object.assign(stats, newStats);
          }
        } catch (error) {
          console.error("Error processing audio stats:", error);
          lastError.value = error.message;
        }
      });

      // Listen for stream state changes
      await listen("audio_stream_started", () => {
        console.log("Audio stream started");
        isStreaming.value = true;
        connectionStatus.value = "connected";
        clearBuffers();
      });

      await listen("audio_stream_stopped", () => {
        console.log("Audio stream stopped");
        isStreaming.value = false;
        connectionStatus.value = "disconnected";
        clearBuffers();
      });

      // Listen for performance metrics
      unlistenPerformanceMetrics = await listen(
        "performance_metrics",
        (event) => {
          try {
            const metrics = event.payload;
            if (metrics && typeof metrics === "object") {
              inferenceTime.value = metrics.inference_time_ms || 0;
              algorithmLatency.value = metrics.algorithm_latency_ms || 0;
              bufferLatency.value = metrics.buffer_latency_ms || 0;
              totalLatency.value = algorithmLatency.value + bufferLatency.value;
            }
          } catch (error) {
            console.error("Error processing performance metrics:", error);
            lastError.value = error.message;
          }
        },
      );

      // Listen for error events
      await listen("audio_stream_error", (event) => {
        console.error("Audio stream error:", event.payload);
        lastError.value =
          event.payload?.message || "Unknown audio stream error";
        connectionStatus.value = "error";
      });

      console.log("Audio stream listeners initialized successfully");
      connectionStatus.value = "connected";
    } catch (error) {
      console.error("Failed to initialize audio stream:", error);
      lastError.value = error.message;
      connectionStatus.value = "error";
      throw error;
    }
  };

  const updateInputBuffer = (samples) => {
    try {
      if (!Array.isArray(samples) || samples.length === 0) {
        console.warn("Invalid input buffer samples:", samples);
        return;
      }

      inputBuffer.push(...samples);

      // Keep buffer size manageable
      if (inputBuffer.length > maxBufferSize) {
        inputBuffer = inputBuffer.slice(-maxBufferSize);
      }

      // Update reactive data
      inputAudioData.value = [...inputBuffer];
    } catch (error) {
      console.error("Error updating input buffer:", error);
      lastError.value = error.message;
    }
  };

  const updateOutputBuffer = (samples) => {
    try {
      if (!Array.isArray(samples) || samples.length === 0) {
        console.warn("Invalid output buffer samples:", samples);
        return;
      }

      outputBuffer.push(...samples);

      // Keep buffer size manageable
      if (outputBuffer.length > maxBufferSize) {
        outputBuffer = outputBuffer.slice(-maxBufferSize);
      }

      // Update reactive data
      outputAudioData.value = [...outputBuffer];
    } catch (error) {
      console.error("Error updating output buffer:", error);
      lastError.value = error.message;
    }
  };

  const calculateInputVolume = (samples) => {
    if (samples.length === 0) {
      inputVolume.value = -60;
      return;
    }

    // Calculate RMS
    const rms = Math.sqrt(
      samples.reduce((sum, sample) => sum + sample * sample, 0) /
        samples.length,
    );

    // Convert to dB
    inputVolume.value = rms > 0 ? Math.max(-60, 20 * Math.log10(rms)) : -60;
  };

  const calculateOutputVolume = (samples) => {
    if (samples.length === 0) {
      outputVolume.value = -60;
      return;
    }

    // Calculate RMS
    const rms = Math.sqrt(
      samples.reduce((sum, sample) => sum + sample * sample, 0) /
        samples.length,
    );

    // Convert to dB
    outputVolume.value = rms > 0 ? Math.max(-60, 20 * Math.log10(rms)) : -60;
  };

  const clearBuffers = () => {
    inputBuffer = [];
    outputBuffer = [];
    inputAudioData.value = [];
    outputAudioData.value = [];
    inputVolume.value = -60;
    outputVolume.value = -60;
  };

  const getInputPeakLevel = () => {
    if (inputBuffer.length === 0) return 0;
    return Math.max(...inputBuffer.map(Math.abs));
  };

  const getOutputPeakLevel = () => {
    if (outputBuffer.length === 0) return 0;
    return Math.max(...outputBuffer.map(Math.abs));
  };

  const getInputFrequencyData = () => {
    if (inputBuffer.length < 512) return [];

    // Simple FFT approximation for frequency analysis
    // This is a simplified version - in production, you'd want a proper FFT
    const fftSize = Math.min(512, inputBuffer.length);
    const samples = inputBuffer.slice(-fftSize);

    // Basic frequency binning
    const bins = 32;
    const binSize = Math.floor(fftSize / bins);
    const freqData = [];

    for (let i = 0; i < bins; i++) {
      const start = i * binSize;
      const end = Math.min(start + binSize, samples.length);
      const binSamples = samples.slice(start, end);

      if (binSamples.length > 0) {
        const rms = Math.sqrt(
          binSamples.reduce((sum, sample) => sum + sample * sample, 0) /
            binSamples.length,
        );
        freqData.push(rms);
      } else {
        freqData.push(0);
      }
    }

    return freqData;
  };

  const getOutputFrequencyData = () => {
    if (outputBuffer.length < 512) return [];

    const fftSize = Math.min(512, outputBuffer.length);
    const samples = outputBuffer.slice(-fftSize);

    const bins = 32;
    const binSize = Math.floor(fftSize / bins);
    const freqData = [];

    for (let i = 0; i < bins; i++) {
      const start = i * binSize;
      const end = Math.min(start + binSize, samples.length);
      const binSamples = samples.slice(start, end);

      if (binSamples.length > 0) {
        const rms = Math.sqrt(
          binSamples.reduce((sum, sample) => sum + sample * sample, 0) /
            binSamples.length,
        );
        freqData.push(rms);
      } else {
        freqData.push(0);
      }
    }

    return freqData;
  };

  // Cleanup function
  const cleanup = async () => {
    try {
      if (unlistenInputAudio) {
        await unlistenInputAudio();
        unlistenInputAudio = null;
      }
      if (unlistenOutputAudio) {
        await unlistenOutputAudio();
        unlistenOutputAudio = null;
      }
      if (unlistenStats) {
        await unlistenStats();
        unlistenStats = null;
      }
      if (unlistenPerformanceMetrics) {
        await unlistenPerformanceMetrics();
        unlistenPerformanceMetrics = null;
      }
      clearBuffers();
    } catch (error) {
      console.error("Error during audio stream cleanup:", error);
    }
  };

  // Auto cleanup on unmount
  onUnmounted(() => {
    cleanup();
  });

  return {
    // Reactive data
    inputAudioData,
    outputAudioData,
    isStreaming,
    inputVolume,
    outputVolume,
    stats,
    lastError,
    connectionStatus,

    // Performance metrics
    inferenceTime,
    algorithmLatency,
    bufferLatency,
    totalLatency,

    // Methods
    initializeAudioStream,
    clearBuffers,
    getInputPeakLevel,
    getOutputPeakLevel,
    getInputFrequencyData,
    getOutputFrequencyData,
    cleanup,
  };
}
