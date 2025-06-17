use crate::{GUIConfig, Harvest};

/// Minimal placeholder for realtime voice conversion logic.
///
/// The current implementation simply echoes the input audio. It
/// will be extended with model inference in the future.
pub struct RVC {
    pitch: f32,
    formant: f32,
    f0_min: f32,
    f0_max: f32,
    f0_mel_min: f32,
    f0_mel_max: f32,
}

impl RVC {
    /// Create a new `RVC` instance using values from the GUI configuration.
    pub fn from_config(cfg: &GUIConfig) -> Self {
        let f0_min = 50.0;
        let f0_max = 1100.0;
        let f0_mel_min = 1127.0 * ((1.0_f32 + f0_min / 700.0).ln());
        let f0_mel_max = 1127.0 * ((1.0_f32 + f0_max / 700.0).ln());
        Self {
            pitch: cfg.pitch,
            formant: cfg.formant,
            f0_min,
            f0_max,
            f0_mel_min,
            f0_mel_max,
        }
    }

    /// Process an input buffer and return converted audio.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        // TODO: call voice conversion models.
        let _ = (self.pitch, self.formant); // suppress unused warnings
        input.to_vec()
    }

    /// Extract F0 from an audio buffer using the specified method and
    /// convert it into coarse representation.
    ///
    /// Currently only the `harvest` method is implemented.
    pub fn get_f0(
        &self,
        x: &[f32],
        f0_up_key: f32,
        method: &str,
    ) -> (Vec<u8>, Vec<f32>) {
        let f0: Vec<f32> = match method {
            "harvest" | _ => {
                let extractor = Harvest::new(16000);
                extractor
                    .compute(x)
                    .into_iter()
                    .map(|v| v as f32)
                    .collect()
            }
        };
        let scale = (2.0f32).powf(f0_up_key / 12.0);
        let f0: Vec<f32> = f0.into_iter().map(|v| v * scale).collect();
        self.get_f0_post(&f0)
    }

    /// Convert raw F0 values into 8-bit coarse representation on the mel scale.
    pub fn get_f0_post(&self, f0: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let mut coarse = Vec::with_capacity(f0.len());
        let mut f0_out = Vec::with_capacity(f0.len());
        for &val in f0 {
            let mut mel = 1127.0 * ((1.0_f32 + val / 700.0).ln());
            if mel > 0.0 {
                mel = (mel - self.f0_mel_min) * 254.0 / (self.f0_mel_max - self.f0_mel_min) + 1.0;
            }
            if mel <= 1.0 {
                mel = 1.0;
            }
            if mel > 255.0 {
                mel = 255.0;
            }
            coarse.push(mel.round() as u8);
            f0_out.push(val);
        }
        (coarse, f0_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_f0_post_basic() {
        let cfg = GUIConfig::default();
        let rvc = RVC::from_config(&cfg);
        let input = vec![0.0, 50.0, 100.0, 1100.0];
        let (coarse, out) = rvc.get_f0_post(&input);
        assert_eq!(out, input);
        assert_eq!(coarse.len(), input.len());
        // Values below or equal to f0_min should map to 1
        assert_eq!(coarse[0], 1);
        assert_eq!(coarse[1], 1);
        // Maximum bound should clip to 255
        assert_eq!(coarse[3], 255);
        // Intermediate value should be in range
        assert!(coarse[2] > 1 && coarse[2] < 255);
    }

    #[test]
    fn test_get_f0_harvest_zero_signal() {
        let cfg = GUIConfig::default();
        let rvc = RVC::from_config(&cfg);
        let input = vec![0.0f32; 160];
        let (coarse, f0) = rvc.get_f0(&input, 0.0, "harvest");
        assert!(f0.iter().all(|&v| v == 0.0));
        assert!(coarse.iter().all(|&c| c == 1));
    }
}

