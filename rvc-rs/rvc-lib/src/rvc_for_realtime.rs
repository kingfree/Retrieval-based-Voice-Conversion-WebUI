use crate::{GUIConfig, Harvest};

/// Minimal placeholder for realtime voice conversion logic.
///
/// The current implementation simply echoes the input audio. It
/// will be extended with model inference in the future.
pub struct RVC {
    pitch: f32,
    formant: f32,
    pth_path: String,
    index_path: String,
    index_rate: f32,
    n_cpu: u32,
    f0_min: f32,
    f0_max: f32,
    f0_mel_min: f32,
    f0_mel_max: f32,
}

impl RVC {
    /// Create a new `RVC` instance using values from the GUI configuration.
    pub fn new(cfg: &GUIConfig) -> Self {
        let f0_min = 50.0;
        let f0_max = 1100.0;
        let f0_mel_min = 1127.0 * ((1.0_f32 + f0_min / 700.0).ln());
        let f0_mel_max = 1127.0 * ((1.0_f32 + f0_max / 700.0).ln());
        Self {
            pitch: cfg.pitch,
            formant: cfg.formant,
            pth_path: cfg.pth_path.clone(),
            index_path: cfg.index_path.clone(),
            index_rate: cfg.index_rate,
            n_cpu: cfg.n_cpu,
            f0_min,
            f0_max,
            f0_mel_min,
            f0_mel_max,
        }
    }

    /// Process an input buffer and return converted audio.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        // The real project will eventually call the trained models here.
        // For now we provide a very small pitch shifting implementation so
        // that the realtime pipeline does something audible.  The
        // `pitch` value stores a semitone offset.  We simply resample the
        // input using linear interpolation and ignore formant shifting.

        let ratio = (2.0f32).powf(self.pitch / 12.0);
        if ratio == 1.0 {
            // Fast path for the common "no-op" case.
            return input.to_vec();
        }

        let out_len = ((input.len() as f32) / ratio).round() as usize;
        let mut output = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = i as f32 * ratio;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f32;
            if idx + 1 < input.len() {
                let a = input[idx];
                let b = input[idx + 1];
                output.push(a * (1.0 - frac) + b * frac);
            } else if idx < input.len() {
                output.push(input[idx]);
            }
        }
        output
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
        let rvc = RVC::new(&cfg);
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
        let rvc = RVC::new(&cfg);
        let input = vec![0.0f32; 160];
        let (coarse, f0) = rvc.get_f0(&input, 0.0, "harvest");
        assert!(f0.iter().all(|&v| v == 0.0));
        assert!(coarse.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_infer_identity() {
        let mut cfg = GUIConfig::default();
        cfg.pitch = 0.0;
        let mut rvc = RVC::new(&cfg);
        let input: Vec<f32> = (0..320).map(|_| 0.5).collect();
        let output = rvc.infer(&input);
        assert_eq!(input, output);
    }

    #[test]
    fn test_infer_pitch_shift_up() {
        let mut cfg = GUIConfig::default();
        cfg.pitch = 12.0; // one octave up
        let mut rvc = RVC::new(&cfg);
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let output = rvc.infer(&input);

        // Manually computed linear resample at ratio=2
        let expected = vec![0.0, 1.0, 0.0, -1.0];
        assert_eq!(output, expected);
    }
}

