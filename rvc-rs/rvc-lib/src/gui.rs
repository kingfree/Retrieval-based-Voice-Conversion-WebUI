use serde::{Serialize, Deserialize};
use serde::de::{self, Deserializer};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GUIConfig {
    pub pth_path: String,
    pub index_path: String,
    pub sg_hostapi: String,
    #[serde(default)]
    pub sg_wasapi_exclusive: bool,
    pub sg_input_device: String,
    pub sg_output_device: String,
    pub sr_type: String,
    #[serde(rename = "threhold")]
    pub threshold: f32,
    pub pitch: f32,
    pub formant: f32,
    #[serde(default)]
    pub rms_mix_rate: f32,
    #[serde(default)]
    pub index_rate: f32,
    pub block_time: f32,
    pub crossfade_length: f32,
    pub extra_time: f32,
    #[serde(deserialize_with = "de_u32_from_any")]
    pub n_cpu: u32,
    #[serde(default)]
    pub use_jit: bool,
    #[serde(default)]
    pub use_pv: bool,
    pub f0method: String,
    #[serde(default)]
    pub I_noise_reduce: bool,
    #[serde(default)]
    pub O_noise_reduce: bool,
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
        let cfg: GUIConfig = serde_json::from_str(&text)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_default_config() {
        let cfg = GUI::load().unwrap();
        assert!(!cfg.pth_path.is_empty());
    }
}
