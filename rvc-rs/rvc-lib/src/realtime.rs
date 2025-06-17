use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::{get_device_channels, GUI, rvc_for_realtime::RVC};

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
    let selected = crate::devices::selected_devices()
        .ok_or_else(|| "devices not set".to_string())?;

    let cfg = GUI::load().map_err(|e| e.to_string())?;
    let rvc = Arc::new(Mutex::new(RVC::from_config(&cfg)));

    let host_id = cpal::available_hosts()
        .iter()
        .find(|id| id.name() == selected.hostapi)
        .copied()
        .ok_or_else(|| format!("hostapi '{}' not found", selected.hostapi))?;
    let host = cpal::host_from_id(host_id).map_err(|e| e.to_string())?;

    let input = host
        .input_devices()
        .map_err(|e| e.to_string())?
        .find(|d| d.name().map(|n| n == selected.input_device).unwrap_or(false))
        .ok_or_else(|| format!("input device '{}' not found", selected.input_device))?;
    let output = host
        .output_devices()
        .map_err(|e| e.to_string())?
        .find(|d| d.name().map(|n| n == selected.output_device).unwrap_or(false))
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
    let err_fn = |e| eprintln!("stream error: {e}");
    let input_stream = input
        .build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mut rvc = rvc_in.lock().unwrap();
                let processed = rvc.infer(data);
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
