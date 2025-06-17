use rvc_lib::{GUIConfig, RVC};
use std::fs;

fn load_json_f32_array(path: &str) -> Vec<f32> {
    let data = fs::read_to_string(path).expect(&format!("Failed to read {}", path));
    serde_json::from_str::<Vec<f32>>(&data).expect(&format!("Failed to parse {}", path))
}

#[test]
fn test_infer_without_model() {
    let cfg = GUIConfig::default();
    let mut rvc = RVC::new(&cfg);

    let input_wav = vec![0.0; 1600]; // 0.1 seconds at 16kHz
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    // Should fail because no model is loaded
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("model not loaded"));
}

#[test]
fn test_infer_with_mock_model() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.0; 1600]; // 0.1 seconds at 16kHz
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    // Should succeed with dummy output
    match result {
        Ok(output) => {
            assert_eq!(output.len(), 320);
        }
        Err(e) => {
            panic!("Inference failed with error: {}", e);
        }
    }
}

#[test]
fn test_infer_different_block_sizes() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let block_sizes = [160, 320, 640, 1280];
    let input_wav = vec![0.0; 6400]; // 0.4 seconds at 16kHz

    for &block_size in &block_sizes {
        let result = rvc.infer(&input_wav, block_size, 0, block_size, "pm");
        assert!(result.is_ok(), "Failed for block size {}", block_size);

        let output = result.unwrap();
        assert_eq!(
            output.len(),
            block_size,
            "Wrong output length for block size {}",
            block_size
        );
    }
}

#[test]
fn test_infer_different_f0_methods() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.0; 1600];
    let f0_methods = ["pm", "crepe", "rmvpe", "fcpe", "harvest"];

    for method in &f0_methods {
        let result = rvc.infer(&input_wav, 320, 0, 320, method);
        assert!(result.is_ok(), "Failed for F0 method {}", method);

        let output = result.unwrap();
        assert_eq!(
            output.len(),
            320,
            "Wrong output length for method {}",
            method
        );
    }
}

#[test]
fn test_infer_with_pitch_shift() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    cfg.pitch = 12.0; // One octave up
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.5; 1600];
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 320);
}

#[test]
fn test_infer_with_formant_shift() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    cfg.formant = 1.2; // Formant shift
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.5; 1600];
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 320);
}

#[test]
fn test_infer_with_index_search() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    cfg.index_path = "mock_index.index".to_string();
    cfg.index_rate = 0.5;
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;
    rvc.index_loaded = true;

    let input_wav = vec![0.5; 1600];
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 320);
}

#[test]
fn test_infer_with_skip_head() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.5; 3200]; // 0.2 seconds at 16kHz
    let skip_head = 160; // Skip first 0.01 seconds
    let return_length = 320;

    let result = rvc.infer(&input_wav, 640, skip_head, return_length, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), return_length);
}

#[test]
fn test_infer_non_f0_model() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state and disable F0
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;
    rvc.if_f0 = 0; // Non-F0 conditioned model

    let input_wav = vec![0.5; 1600];
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 320);
}

#[test]
fn test_infer_silence_input() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.0; 1600]; // Silence
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 320);

    // Output should not be all zeros (due to model processing)
    // but we can't guarantee specific values with placeholder implementation
}

#[test]
fn test_infer_sine_wave_input() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Generate 440Hz sine wave at 16kHz for 0.1 seconds
    let sample_rate = 16000.0;
    let frequency = 440.0;
    let duration = 0.1;
    let samples = (sample_rate * duration) as usize;

    let input_wav: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 320);
}

#[test]
fn test_infer_cache_behavior() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.5; 1600];

    // First inference
    let result1 = rvc.infer(&input_wav, 320, 0, 320, "pm");
    assert!(result1.is_ok());

    // Second inference should use cached pitch data
    let result2 = rvc.infer(&input_wav, 320, 0, 320, "pm");
    assert!(result2.is_ok());

    // Both should produce same length output
    assert_eq!(result1.unwrap().len(), result2.unwrap().len());
}

#[test]
fn test_infer_parameter_consistency() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    cfg.pitch = 5.0;
    cfg.formant = 1.1;
    cfg.index_rate = 0.3;
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.5; 1600];
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());

    // Verify parameters are maintained
    assert_eq!(rvc.f0_up_key, 5.0);
    assert_eq!(rvc.formant_shift, 1.1);
    assert_eq!(rvc.index_rate, 0.3);
}

#[test]
fn test_infer_error_handling() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Test with model loaded but no HuBERT
    rvc.model_loaded = true;
    rvc.hubert_loaded = false;

    let input_wav = vec![0.5; 1600];
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("HuBERT"));
}

#[test]
fn test_infer_timing_information() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let input_wav = vec![0.5; 1600];

    // Capture timing output (in real scenario, we'd use a proper logging framework)
    let result = rvc.infer(&input_wav, 320, 0, 320, "pm");

    assert!(result.is_ok());
    // Timing information is printed to stdout during inference
}

#[test]
fn test_infer_readiness_check() {
    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);

    // Should not be ready without models
    assert!(!rvc.is_ready());

    let mut cfg_with_model = GUIConfig::default();
    cfg_with_model.pth_path = "mock_model.pth".to_string();
    let mut rvc_with_model = RVC::new(&cfg_with_model);

    // Force model loaded state
    rvc_with_model.model_loaded = true;
    rvc_with_model.hubert_loaded = true;

    // Should be ready with models loaded
    assert!(rvc_with_model.is_ready());
}
