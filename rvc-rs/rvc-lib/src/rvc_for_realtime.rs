use crate::GUIConfig;

/// Minimal placeholder for realtime voice conversion logic.
///
/// The current implementation simply echoes the input audio. It
/// will be extended with model inference in the future.
pub struct RVC {
    pitch: f32,
    formant: f32,
}

impl RVC {
    /// Create a new `RVC` instance using values from the GUI configuration.
    pub fn from_config(cfg: &GUIConfig) -> Self {
        Self {
            pitch: cfg.pitch,
            formant: cfg.formant,
        }
    }

    /// Process an input buffer and return converted audio.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        // TODO: call voice conversion models.
        let _ = (self.pitch, self.formant); // suppress unused warnings
        input.to_vec()
    }
}

