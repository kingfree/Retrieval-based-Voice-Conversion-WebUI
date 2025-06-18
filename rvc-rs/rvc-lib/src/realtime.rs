use crate::phase_vocoder;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tch::Tensor;

use crate::{GUI, get_device_channels, rvc_for_realtime::RVC};

/// Callback function type for audio data with timing information
pub type AudioDataCallback = Arc<dyn Fn(&[f32], &[f32], f32, f32) + Send + Sync>;

/// Handle for an active voice conversion stream.
///
/// The streams are kept alive as long as this struct is alive.
pub struct VC {
    _input: cpal::Stream,
    _output: cpal::Stream,
    _buffer: Arc<Mutex<Vec<f32>>>,
    _rvc: Arc<Mutex<RVC>>,
    _audio_callback: Option<AudioDataCallback>,
}

impl VC {
    /// Update the pitch shift of the running RVC instance.
    pub fn change_key(&self, key: f32) {
        if let Ok(mut rvc) = self._rvc.lock() {
            rvc.change_key(key);
        }
    }

    /// Update the formant shift of the running RVC instance.
    pub fn change_formant(&self, formant: f32) {
        if let Ok(mut rvc) = self._rvc.lock() {
            rvc.change_formant(formant);
        }
    }

    /// Update the index rate of the running RVC instance.
    pub fn change_index_rate(&self, rate: f32) {
        if let Ok(mut rvc) = self._rvc.lock() {
            rvc.change_index_rate(rate);
        }
    }
}

/// Start realtime voice conversion using the previously selected devices.
///
/// This currently just copies audio from the input device to the output device.
pub fn start_vc() -> Result<VC, String> {
    start_vc_with_callback(None)
}

/// Start realtime voice conversion with an optional audio data callback.
pub fn start_vc_with_callback(audio_callback: Option<AudioDataCallback>) -> Result<VC, String> {
    let selected =
        crate::devices::selected_devices().ok_or_else(|| "devices not set".to_string())?;

    let cfg = GUI::load().map_err(|e| e.to_string())?;
    let rvc = Arc::new(Mutex::new(RVC::new(&cfg)));

    // Clone config for use in audio callback
    let cfg_clone = cfg.clone();

    // prepare crossfade windows for smoother output
    let crossfade_frames = (cfg.crossfade_length * selected.sample_rate as f32).round() as usize;
    let crossfade_frames = crossfade_frames.max(1);

    // Create fade windows as Vec<f32> to avoid thread safety issues with Tensor
    let mut fade_in_vec = Vec::with_capacity(crossfade_frames);
    let mut fade_out_vec = Vec::with_capacity(crossfade_frames);
    for i in 0..crossfade_frames {
        let pos = i as f32 / (crossfade_frames - 1) as f32;
        let fade_in_val = (pos * (0.5 * std::f32::consts::PI)).sin().powi(2);
        fade_in_vec.push(fade_in_val);
        fade_out_vec.push(1.0 - fade_in_val);
    }
    let fade_in_window = Arc::new(fade_in_vec);
    let fade_out_window = Arc::new(fade_out_vec);
    let prev_chunk: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    let host_id = cpal::available_hosts()
        .iter()
        .find(|id| id.name() == selected.hostapi)
        .copied()
        .ok_or_else(|| format!("hostapi '{}' not found", selected.hostapi))?;
    let host = cpal::host_from_id(host_id).map_err(|e| e.to_string())?;

    let input = host
        .input_devices()
        .map_err(|e| e.to_string())?
        .find(|d| {
            d.name()
                .map(|n| n == selected.input_device)
                .unwrap_or(false)
        })
        .ok_or_else(|| format!("input device '{}' not found", selected.input_device))?;
    let output = host
        .output_devices()
        .map_err(|e| e.to_string())?
        .find(|d| {
            d.name()
                .map(|n| n == selected.output_device)
                .unwrap_or(false)
        })
        .ok_or_else(|| format!("output device '{}' not found", selected.output_device))?;

    let channels = get_device_channels()?;
    let config = cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(selected.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buf_in = buffer.clone();
    let rvc_in = rvc.clone();
    let fade_in_c = fade_in_window.clone();
    let fade_out_c = fade_out_window.clone();
    let prev_in = prev_chunk.clone();
    let callback_clone = audio_callback.clone();
    let cfg_in = cfg_clone.clone();
    let selected_in = selected.clone();
    let err_fn = |e| eprintln!("stream error: {e}");
    let input_stream = input
        .build_input_stream(
            &config,
            move |data: &[f32], _| {
                let processing_start = Instant::now();

                // Store original input data for callback
                let input_data = data.to_vec();

                // Measure inference time
                let inference_start = Instant::now();
                let mut rvc = rvc_in.lock().unwrap();

                // Parameters for RVC::infer (matching Python implementation)
                // Calculate parameters based on GUI configuration like Python version
                let zc = selected_in.sample_rate / 100; // 1% of sample rate
                let block_frame =
                    (cfg_in.block_time * selected_in.sample_rate as f32).round() as usize;
                let block_frame_16k = 160 * block_frame / zc as usize; // Convert to 16kHz frame size
                let extra_frame =
                    (cfg_in.extra_time * selected_in.sample_rate as f32).round() as usize;
                let crossfade_frame =
                    (cfg_in.crossfade_length * selected_in.sample_rate as f32).round() as usize;
                let sola_buffer_frame = crossfade_frame.min(4 * zc as usize);
                let sola_search_frame = zc as usize;

                let skip_head = extra_frame / zc as usize; // Skip frames at the beginning
                let return_length =
                    (block_frame + sola_buffer_frame + sola_search_frame) / zc as usize; // Return length in frames
                let f0method = &cfg_in.f0method; // Read F0 method from configuration

                let mut processed =
                    match rvc.infer(data, block_frame_16k, skip_head, return_length, f0method) {
                        Ok(result) => result,
                        Err(e) => {
                            eprintln!(
                                "RVC inference failed: {}, falling back to original audio",
                                e
                            );
                            data.to_vec()
                        }
                    };
                let inference_time = inference_start.elapsed().as_secs_f32() * 1000.0; // Convert to ms
                drop(rvc); // Release RVC lock early

                // Measure crossfade processing time
                let _crossfade_start = Instant::now();
                let mut last = prev_in.lock().unwrap();
                if !last.is_empty() {
                    let len = crossfade_frames.min(last.len()).min(processed.len());
                    if len > 0 {
                        let a = Tensor::from_slice(&last[last.len() - len..]);
                        let b = Tensor::from_slice(&processed[..len]);
                        let fade_out_start = crossfade_frames - len;
                        let fade_out = Tensor::from_slice(&fade_out_c[fade_out_start..]);
                        let fade_in = Tensor::from_slice(&fade_in_c[fade_out_start..]);
                        let result = phase_vocoder(&a, &b, &fade_out, &fade_in);
                        let res_vec: Vec<f32> = Vec::<f32>::try_from(result).unwrap();
                        processed[..len].copy_from_slice(&res_vec);
                    }
                }
                *last = processed[processed.len().saturating_sub(crossfade_frames)..].to_vec();
                drop(last); // Release lock

                // Calculate total processing time (algorithm latency)
                let total_processing_time = processing_start.elapsed().as_secs_f32() * 1000.0;

                // Call the audio data callback with timing information
                if let Some(ref callback) = callback_clone {
                    callback(
                        &input_data,
                        &processed,
                        inference_time,
                        total_processing_time,
                    );
                }

                buf_in.lock().unwrap().extend_from_slice(&processed);
            },
            err_fn,
            None,
        )
        .map_err(|e| e.to_string())?;

    let buf_out = buffer.clone();
    let output_stream = output
        .build_output_stream(
            &config,
            move |out: &mut [f32], _| {
                let mut buf = buf_out.lock().unwrap();
                let len = out.len().min(buf.len());
                out[..len].copy_from_slice(&buf[..len]);
                if len < out.len() {
                    for o in &mut out[len..] {
                        *o = 0.0;
                    }
                }
                buf.drain(..len);
            },
            err_fn,
            None,
        )
        .map_err(|e| e.to_string())?;

    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;

    Ok(VC {
        _input: input_stream,
        _output: output_stream,
        _buffer: buffer,
        _rvc: rvc,
        _audio_callback: audio_callback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_vc_without_devices_fails() {
        assert!(start_vc().is_err());
    }
}
