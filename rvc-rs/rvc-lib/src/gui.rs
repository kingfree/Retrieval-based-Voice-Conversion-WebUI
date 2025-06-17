use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct GUIConfig {
    pub pth_path: String,
    pub index_path: String,
    pub sg_hostapi: String,
    pub sg_wasapi_exclusive: bool,
    pub sg_input_device: String,
    pub sg_output_device: String,
    pub sr_type: String,
    #[serde(rename = "threhold")]
    pub threshold: f32,
    pub pitch: f32,
    pub formant: f32,
    pub rms_mix_rate: f32,
    pub index_rate: f32,
    pub block_time: f32,
    pub crossfade_length: f32,
    pub extra_time: f32,
    #[serde(deserialize_with = "de_u32_from_any")]
    pub n_cpu: u32,
    pub use_jit: bool,
    pub use_pv: bool,
    pub f0method: String,
    #[serde(rename = "I_noise_reduce")]
    pub i_noise_reduce: bool,
    #[serde(rename = "O_noise_reduce")]
    pub o_noise_reduce: bool,
}

impl Default for GUIConfig {
    fn default() -> Self {
        Self {
            pth_path: String::new(),
            index_path: String::new(),
            sg_hostapi: String::new(),
            sg_wasapi_exclusive: false,
            sg_input_device: String::new(),
            sg_output_device: String::new(),
            sr_type: "sr_model".to_string(),
            threshold: -60.0,
            pitch: 0.0,
            formant: 0.0,
            rms_mix_rate: 0.0,
            index_rate: 0.0,
            block_time: 0.25,
            crossfade_length: 0.05,
            extra_time: 2.5,
            n_cpu: 4,
            use_jit: false,
            use_pv: false,
            f0method: "rmvpe".to_string(),
            i_noise_reduce: false,
            o_noise_reduce: false,
        }
    }
}

fn de_u32_from_any<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let val = serde_json::Value::deserialize(deserializer)?;
    if let Some(u) = val.as_u64() {
        return Ok(u as u32);
    }
    if let Some(f) = val.as_f64() {
        return Ok(f as u32);
    }
    Err(de::Error::custom("invalid number for n_cpu"))
}

pub struct GUI;

impl GUI {
    fn config_paths() -> (PathBuf, PathBuf) {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inuse = base.join("configs/inuse/config.json");
        let default = base.join("configs/config.json");
        (inuse, default)
    }

    pub fn load() -> Result<GUIConfig, Box<dyn std::error::Error>> {
        let (inuse, default) = Self::config_paths();
        if !inuse.exists() {
            if let Some(parent) = inuse.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&default, &inuse)?;
        }
        let text = fs::read_to_string(&inuse)?;
        let cfg: GUIConfig = serde_json::from_str(&text).unwrap_or_default();
        Ok(cfg)
    }

    pub fn save(cfg: &GUIConfig) -> Result<(), Box<dyn std::error::Error>> {
        let (inuse, _) = Self::config_paths();
        if let Some(parent) = inuse.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(cfg)?;
        fs::write(inuse, text)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_flow_invalid_devices() {
        // load default config
        let mut cfg = GUI::load().expect("load config");
        cfg.pitch = 1.23;
        GUI::save(&cfg).expect("save config");

        // verify save
        let loaded = GUI::load().expect("reload");
        assert_eq!(loaded.pitch, 1.23);

        // try enumerating devices (may fail on CI)
        let _ = crate::update_devices(None);

        // selecting invalid devices should fail
        assert!(crate::set_devices("none", "in", "out").is_err());

        // without valid devices, starting VC fails
        assert!(crate::start_vc().is_err());
    }
}
