use cpal::traits::{DeviceTrait, HostTrait};
use once_cell::sync::Lazy;
use serde::Serialize;
use std::sync::Mutex;

#[derive(Serialize)]
pub struct DeviceInfo {
    pub hostapis: Vec<String>,
    pub input_devices: Vec<String>,
    pub output_devices: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedDevices {
    pub(crate) hostapi: String,
    pub(crate) input_device: String,
    pub(crate) output_device: String,
    pub(crate) sample_rate: u32,
}

pub(crate) static SELECTED: Lazy<Mutex<Option<SelectedDevices>>> = Lazy::new(|| Mutex::new(None));

pub(crate) fn selected_devices() -> Option<SelectedDevices> {
    SELECTED.lock().unwrap().clone()
}

/// Enumerate audio devices for the given host API.
pub fn update_devices(hostapi: Option<&str>) -> Result<DeviceInfo, String> {
    let host_ids = cpal::available_hosts();
    let host_names: Vec<String> = host_ids.iter().map(|id| id.name().to_string()).collect();

    let host = if let Some(name) = hostapi {
        if let Some(id) = host_ids.iter().find(|id| id.name() == name).copied() {
            cpal::host_from_id(id).map_err(|e| e.to_string())?
        } else {
            cpal::default_host()
        }
    } else {
        cpal::default_host()
    };

    let input_devices = host
        .input_devices()
        .map_err(|e| e.to_string())?
        .filter_map(|d| d.name().ok())
        .collect();
    let output_devices = host
        .output_devices()
        .map_err(|e| e.to_string())?
        .filter_map(|d| d.name().ok())
        .collect();

    Ok(DeviceInfo { hostapis: host_names, input_devices, output_devices })
}

/// Select the current input and output devices.
/// Returns the chosen sample rate on success.
pub fn set_devices(hostapi: &str, input: &str, output: &str) -> Result<u32, String> {
    let info = update_devices(Some(hostapi))?;
    if !info.input_devices.contains(&input.to_string()) {
        return Err(format!("input device '{input}' not found"));
    }
    if !info.output_devices.contains(&output.to_string()) {
        return Err(format!("output device '{output}' not found"));
    }

    let host_ids = cpal::available_hosts();
    let host_id = host_ids
        .iter()
        .find(|id| id.name() == hostapi)
        .copied()
        .ok_or_else(|| format!("hostapi '{hostapi}' not found"))?;
    let host = cpal::host_from_id(host_id).map_err(|e| e.to_string())?;

    let input_dev = host
        .input_devices()
        .map_err(|e| e.to_string())?
        .find(|d| d.name().map(|n| n == input).unwrap_or(false))
        .ok_or_else(|| format!("input device '{input}' not found"))?;
    let cfg = input_dev
        .default_input_config()
        .map_err(|e| e.to_string())?;
    let sr = cfg.sample_rate().0;

    let _output_dev = host
        .output_devices()
        .map_err(|e| e.to_string())?
        .find(|d| d.name().map(|n| n == output).unwrap_or(false))
        .ok_or_else(|| format!("output device '{output}' not found"))?;

    *SELECTED.lock().unwrap() = Some(SelectedDevices {
        hostapi: hostapi.to_string(),
        input_device: input.to_string(),
        output_device: output.to_string(),
        sample_rate: sr,
    });
    Ok(sr)
}

/// Retrieve the last selected sample rate if set.
pub fn selected_sample_rate() -> Option<u32> {
    SELECTED.lock().unwrap().as_ref().map(|s| s.sample_rate)
}

/// Query the default sample rate of the currently selected input device.
pub fn get_device_samplerate() -> Result<u32, String> {
    let selected = SELECTED
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "devices not set".to_string())?;

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

    let cfg = input.default_input_config().map_err(|e| e.to_string())?;
    Ok(cfg.sample_rate().0)
}

/// Determine the number of channels supported by both selected devices.
/// Returns the minimum of the two, capped at 2.
pub fn get_device_channels() -> Result<u16, String> {
    let selected = SELECTED
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "devices not set".to_string())?;

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
    let in_cfg = input.default_input_config().map_err(|e| e.to_string())?;

    let output = host
        .output_devices()
        .map_err(|e| e.to_string())?
        .find(|d| d.name().map(|n| n == selected.output_device).unwrap_or(false))
        .ok_or_else(|| format!("output device '{}' not found", selected.output_device))?;
    let out_cfg = output.default_output_config().map_err(|e| e.to_string())?;

    let channels = u16::min(u16::min(in_cfg.channels(), out_cfg.channels()), 2);
    Ok(channels)
}
