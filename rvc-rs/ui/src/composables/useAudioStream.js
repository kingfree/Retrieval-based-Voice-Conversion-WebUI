import { ref, reactive, onUnmounted } from "vue";
import { listen } from "@tauri-apps/api/event";

export function useAudioStream(demoMode = false) {
  // Audio data buffers
  const inputAudioData = ref([]);
  const outputAudioData = ref([]);

  // Stream state
  const isStreaming = ref(false);
  const inputVolume = ref(-60);
  const outputVolume = ref(-60);
  const lastError = ref(null);
  const connectionStatus = ref("disconnected");

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

  // Buffer management
  const maxBufferSize = 8192;
  let inputBuffer = [];
  let outputBuffer = [];

  // Demo mode variables
  let demoInterval = null;
  let demoFrame = 0;

  const initializeAudioStream = async () => {
    try {
      lastError.value = null;

      if (demoMode) {
        console.log("Audio stream initialized in demo mode");
        connectionStatus.value = "connected";
        return;
      }

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

  const startDemoStream = () => {
    if (!demoMode || demoInterval) return;

    try {
      console.log("Starting demo audio stream");
      isStreaming.value = true;
      connectionStatus.value = "connected";
      demoFrame = 0;
      lastError.value = null;

      demoInterval = setInterval(() => {
        try {
          const sampleCount = 1024;
          const inputSamples = [];
          const outputSamples = [];

          for (let i = 0; i < sampleCount; i++) {
            const t = (demoFrame * sampleCount + i) / 44100.0;

            // Generate input audio with mixed frequencies and noise
            const inputSignal =
              0.3 * Math.sin(2 * Math.PI * 440 * t) + // A4 note
              0.2 * Math.sin(2 * Math.PI * 880 * t) + // A5 note
              0.1 * Math.sin(2 * Math.PI * 220 * t) + // A3 note
              0.05 * (Math.random() - 0.5); // White noise

            // Generate output audio (processed/converted)
            const outputSignal =
              0.4 * Math.sin(2 * Math.PI * 523.25 * t) + // C5 note
              0.25 * Math.sin(2 * Math.PI * 659.25 * t) + // E5 note
              0.15 * Math.sin(2 * Math.PI * 783.99 * t); // G5 note

            inputSamples.push(inputSignal);
            outputSamples.push(outputSignal);
          }

          updateInputBuffer(inputSamples);
          updateOutputBuffer(outputSamples);
          calculateInputVolume(inputSamples);
          calculateOutputVolume(outputSamples);

          // Update stats
          stats.processedSamples += sampleCount;
          stats.latency = 20 + Math.sin(demoFrame * 0.1) * 10; // Simulate 10-30ms latency

          demoFrame++;
        } catch (error) {
          console.error("Error in demo stream generation:", error);
          lastError.value = error.message;
          stopDemoStream();
        }
      }, 50); // 20 FPS updates
    } catch (error) {
      console.error("Failed to start demo stream:", error);
      lastError.value = error.message;
      connectionStatus.value = "error";
    }
  };

  const stopDemoStream = () => {
    try {
      console.log("Stopping demo audio stream");
      if (demoInterval) {
        clearInterval(demoInterval);
        demoInterval = null;
      }
      isStreaming.value = false;
      connectionStatus.value = "disconnected";
      clearBuffers();
    } catch (error) {
      console.error("Error stopping demo stream:", error);
      lastError.value = error.message;
    }
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
      if (demoMode) {
        stopDemoStream();
      } else {
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

    // Methods
    initializeAudioStream,
    clearBuffers,
    getInputPeakLevel,
    getOutputPeakLevel,
    getInputFrequencyData,
    getOutputFrequencyData,
    cleanup,

    // Demo mode methods
    startDemoStream,
    stopDemoStream,
  };
}
