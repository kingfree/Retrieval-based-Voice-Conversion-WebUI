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
      console.log("🎧 Initializing audio stream listeners...");
      lastError.value = null;
      connectionStatus.value = "connecting";

      // Listen for audio data (both input and output)
      unlistenInputAudio = await listen("audio_data", (event) => {
        try {
          console.log("🎵 Raw audio_data event received:", event);
          const { input_data, output_data, sample_rate } = event.payload;
          console.log("🎵 Received audio_data event:", {
            input_length: input_data?.length || 0,
            output_length: output_data?.length || 0,
            sample_rate,
            payload_keys: Object.keys(event.payload || {}),
          });

          if (input_data && Array.isArray(input_data)) {
            console.log(
              "🔍 Input data preview (first 5):",
              input_data.slice(0, 5),
            );
            updateInputBuffer(input_data);
            calculateInputVolume(input_data);
            inputAudioData.value = input_data;
            console.log(
              "📊 Input audio data updated, length:",
              input_data.length,
              "current inputAudioData.value length:",
              inputAudioData.value?.length,
            );
          } else {
            console.warn(
              "❌ Invalid input_data:",
              typeof input_data,
              input_data,
            );
          }

          if (output_data && Array.isArray(output_data)) {
            console.log(
              "🔍 Output data preview (first 5):",
              output_data.slice(0, 5),
            );
            updateOutputBuffer(output_data);
            calculateOutputVolume(output_data);
            outputAudioData.value = output_data;
            console.log(
              "📊 Output audio data updated, length:",
              output_data.length,
              "current outputAudioData.value length:",
              outputAudioData.value?.length,
            );
          } else {
            console.warn(
              "❌ Invalid output_data:",
              typeof output_data,
              output_data,
            );
          }

          if (sample_rate) {
            // Update sample rate if provided
            console.log("🔊 Sample rate updated:", sample_rate);
          }
        } catch (error) {
          console.error("❌ Error processing audio data:", error);
          lastError.value = error.message;
        }
      });

      // Listen for audio metrics/performance data
      unlistenOutputAudio = await listen("audio_metrics", (event) => {
        try {
          console.log("📈 Raw audio_metrics event received:", event);
          const {
            inference_time,
            total_processing_time,
            input_level,
            output_level,
            timestamp,
          } = event.payload;

          console.log("📈 Received audio_metrics event:", {
            inference_time,
            total_processing_time,
            input_level,
            output_level,
            timestamp,
            payload_keys: Object.keys(event.payload || {}),
          });

          // Update performance metrics
          if (inference_time !== undefined) {
            inferenceTime.value = inference_time;
            console.log("⏱️ Inference time updated:", inference_time, "ms");
          }
          if (total_processing_time !== undefined) {
            algorithmLatency.value = total_processing_time;
            totalLatency.value = total_processing_time;
            console.log(
              "⏱️ Total processing time updated:",
              total_processing_time,
              "ms",
            );
          }

          // Update audio levels if available
          if (input_level !== undefined) {
            const inputVolumeDb = Math.max(
              -60,
              20 * Math.log10(Math.max(input_level, 1e-6)),
            );
            inputVolume.value = inputVolumeDb;
            console.log(
              "🔊 Input volume updated:",
              inputVolumeDb,
              "dB",
              "from level:",
              input_level,
            );
          }
          if (output_level !== undefined) {
            const outputVolumeDb = Math.max(
              -60,
              20 * Math.log10(Math.max(output_level, 1e-6)),
            );
            outputVolume.value = outputVolumeDb;
            console.log(
              "🔊 Output volume updated:",
              outputVolumeDb,
              "dB",
              "from level:",
              output_level,
            );
          }
        } catch (error) {
          console.error("❌ Error processing audio metrics:", error);
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

      console.log("✅ Audio stream listeners initialized successfully");
      console.log("🔗 Connection status set to connected");
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
        console.warn("❌ Invalid input buffer samples:", samples);
        return;
      }

      console.log("📥 Updating input buffer with", samples.length, "samples");
      inputBuffer.push(...samples);

      // Keep buffer size manageable
      if (inputBuffer.length > maxBufferSize) {
        const oldLength = inputBuffer.length;
        inputBuffer = inputBuffer.slice(-maxBufferSize);
        console.log(
          `🗂️ Input buffer trimmed from ${oldLength} to ${inputBuffer.length} samples`,
        );
      }

      // Update reactive data
      inputAudioData.value = [...inputBuffer];
      console.log("✅ Input buffer updated, total length:", inputBuffer.length);
    } catch (error) {
      console.error("❌ Error updating input buffer:", error);
      lastError.value = error.message;
    }
  };

  const updateOutputBuffer = (samples) => {
    try {
      if (!Array.isArray(samples) || samples.length === 0) {
        console.warn("❌ Invalid output buffer samples:", samples);
        return;
      }

      console.log("📥 Updating output buffer with", samples.length, "samples");
      outputBuffer.push(...samples);

      // Keep buffer size manageable
      if (outputBuffer.length > maxBufferSize) {
        const oldLength = outputBuffer.length;
        outputBuffer = outputBuffer.slice(-maxBufferSize);
        console.log(
          `🗂️ Output buffer trimmed from ${oldLength} to ${outputBuffer.length} samples`,
        );
      }

      // Update reactive data
      outputAudioData.value = [...outputBuffer];
      console.log(
        "✅ Output buffer updated, total length:",
        outputBuffer.length,
      );
    } catch (error) {
      console.error("❌ Error updating output buffer:", error);
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

  const stopAudioStream = async () => {
    try {
      console.log("🛑 Stopping audio stream...");
      isStreaming.value = false;
      connectionStatus.value = "disconnected";
      await cleanup();
      console.log("✅ Audio stream stopped successfully");
    } catch (error) {
      console.error("❌ Error stopping audio stream:", error);
      lastError.value = error.message;
      connectionStatus.value = "error";
      throw error;
    }
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
    stopAudioStream,
    clearBuffers,

    // Cleanup
    cleanup,
  };
}
