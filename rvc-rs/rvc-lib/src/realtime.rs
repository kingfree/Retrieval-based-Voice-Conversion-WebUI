use crate::phase_vocoder;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use tch::Tensor;

use crate::{GUI, get_device_channels, rvc_for_realtime::RVC};

/// Handle for an active voice conversion stream.
///
/// The streams are kept alive as long as this struct is alive.
pub struct VC {
    _input: cpal::Stream,
    _output: cpal::Stream,
    _buffer: Arc<Mutex<Vec<f32>>>,
    _rvc: Arc<Mutex<RVC>>,
}

/// Start realtime voice conversion using the previously selected devices.
///
/// This currently just copies audio from the input device to the output device.
pub fn start_vc() -> Result<VC, String> {
    let selected =
        crate::devices::selected_devices().ok_or_else(|| "devices not set".to_string())?;

    let cfg = GUI::load().map_err(|e| e.to_string())?;
    let rvc = Arc::new(Mutex::new(RVC::new(&cfg)));

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
    let err_fn = |e| eprintln!("stream error: {e}");
    let input_stream = input
        .build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mut rvc = rvc_in.lock().unwrap();
                let mut processed = rvc.infer(data);
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
