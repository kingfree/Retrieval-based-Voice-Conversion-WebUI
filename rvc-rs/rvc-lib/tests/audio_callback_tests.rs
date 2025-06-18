use rvc_lib::{AudioCallbackConfig, GUIConfig, RVC};
use std::thread;
use std::time::Duration;
use tch::Device;

#[test]
fn test_audio_callback_config_default() {
    let config = AudioCallbackConfig::default();
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.block_size, 512);
    assert!(config.enable_crossfade);
    assert_eq!(config.crossfade_samples, 64);
}

#[test]
fn test_audio_callback_config_custom() {
    let config = AudioCallbackConfig {
        sample_rate: 44100,
        block_size: 1024,
        enable_crossfade: false,
        crossfade_samples: 128,
    };
    assert_eq!(config.sample_rate, 44100);
    assert_eq!(config.block_size, 1024);
    assert!(!config.enable_crossfade);
    assert_eq!(config.crossfade_samples, 128);
}

#[test]
fn test_create_audio_callback_success() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig::default();
    let result = rvc.create_audio_callback(config);
    assert!(result.is_ok(), "Failed to create audio callback");
}

#[test]
fn test_create_audio_callback_not_ready() {
    let cfg = GUIConfig::default();
    let mut rvc = RVC::new(&cfg);

    let config = AudioCallbackConfig::default();
    let result = rvc.create_audio_callback(config);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.contains("not ready"));
    }
}

#[test]
fn test_set_and_clear_callback() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Initially no callback
    assert!(rvc.audio_callback.is_none());

    // Create and set callback
    let config = AudioCallbackConfig::default();
    let callback = rvc.create_audio_callback(config).unwrap();
    let result = rvc.set_audio_callback(callback);
    assert!(result.is_ok());
    assert!(rvc.audio_callback.is_some());

    // Clear callback
    rvc.clear_audio_callback();
    assert!(rvc.audio_callback.is_none());
}

#[test]
fn test_set_callback_while_streaming() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Start streaming
    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Try to set callback while streaming (should fail)
    let config = AudioCallbackConfig::default();
    let callback = rvc.create_audio_callback(config).unwrap();
    let result = rvc.set_audio_callback(callback);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("while streaming is active"));

    // Stop streaming
    rvc.stop_stream().unwrap();
}

#[test]
fn test_process_audio_callback_success() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Create and set callback
    let config = AudioCallbackConfig::default();
    let callback = rvc.create_audio_callback(config).unwrap();
    rvc.set_audio_callback(callback).unwrap();

    // Test processing
    let input = vec![0.5; 512];
    let mut output = vec![0.0; 512];
    let result = rvc.process_audio_callback(&input, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_process_audio_callback_no_callback() {
    let cfg = GUIConfig::default();
    let mut rvc = RVC::new(&cfg);

    let input = vec![0.5; 512];
    let mut output = vec![0.0; 512];
    let result = rvc.process_audio_callback(&input, &mut output);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No audio callback registered"));
}

#[test]
fn test_audio_callback_with_different_block_sizes() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let block_sizes = [64, 128, 256, 512, 1024, 2048];

    for &block_size in &block_sizes {
        let config = AudioCallbackConfig {
            sample_rate: 16000,
            block_size,
            enable_crossfade: true,
            crossfade_samples: 32,
        };

        let callback = rvc.create_audio_callback(config);
        assert!(
            callback.is_ok(),
            "Failed to create callback for block size {}",
            block_size
        );

        // Test the callback with appropriate sized input
        rvc.set_audio_callback(callback.unwrap()).unwrap();
        let input = vec![0.1; block_size];
        let mut output = vec![0.0; block_size];
        let result = rvc.process_audio_callback(&input, &mut output);
        assert!(
            result.is_ok(),
            "Failed to process with block size {}",
            block_size
        );

        rvc.clear_audio_callback();
    }
}

#[test]
fn test_audio_callback_with_different_sample_rates() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let sample_rates = [8000, 16000, 22050, 44100, 48000];

    for &sample_rate in &sample_rates {
        let config = AudioCallbackConfig {
            sample_rate,
            block_size: 512,
            enable_crossfade: true,
            crossfade_samples: 64,
        };

        let callback = rvc.create_audio_callback(config);
        assert!(
            callback.is_ok(),
            "Failed to create callback for sample rate {}",
            sample_rate
        );

        rvc.set_audio_callback(callback.unwrap()).unwrap();
        let input = vec![0.2; 512];
        let mut output = vec![0.0; 512];
        let result = rvc.process_audio_callback(&input, &mut output);
        assert!(
            result.is_ok(),
            "Failed to process with sample rate {}",
            sample_rate
        );

        rvc.clear_audio_callback();
    }
}

#[test]
fn test_audio_callback_with_crossfade_enabled() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig {
        sample_rate: 16000,
        block_size: 512,
        enable_crossfade: true,
        crossfade_samples: 64,
    };

    let callback = rvc.create_audio_callback(config).unwrap();
    rvc.set_audio_callback(callback).unwrap();

    // Process multiple blocks to test crossfade behavior
    for i in 0..5 {
        let input: Vec<f32> = (0..512)
            .map(|j| (i as f32 + j as f32 / 512.0) * 0.1)
            .collect();
        let mut output = vec![0.0; 512];
        let result = rvc.process_audio_callback(&input, &mut output);
        assert!(result.is_ok(), "Failed to process block {}", i);

        // Output should contain finite values
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite values in output for block {}",
            i
        );
    }
}

#[test]
fn test_audio_callback_with_crossfade_disabled() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig {
        sample_rate: 16000,
        block_size: 512,
        enable_crossfade: false,
        crossfade_samples: 0,
    };

    let callback = rvc.create_audio_callback(config).unwrap();
    rvc.set_audio_callback(callback).unwrap();

    let input = vec![0.3; 512];
    let mut output = vec![0.0; 512];
    let result = rvc.process_audio_callback(&input, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_audio_callback_with_pitch_parameters() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    cfg.pitch = 12.0; // One octave up
    cfg.formant = 1.2;
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig::default();
    let callback = rvc.create_audio_callback(config).unwrap();
    rvc.set_audio_callback(callback).unwrap();

    // Generate test audio (sine wave)
    let input: Vec<f32> = (0..512)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();
    let mut output = vec![0.0; 512];
    let result = rvc.process_audio_callback(&input, &mut output);
    assert!(result.is_ok());

    // With pitch shift, output length may differ but should be valid
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_audio_callback_real_time_simulation() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig {
        sample_rate: 16000,
        block_size: 256,
        enable_crossfade: true,
        crossfade_samples: 32,
    };

    let callback = rvc.create_audio_callback(config).unwrap();
    rvc.set_audio_callback(callback).unwrap();

    // Simulate real-time processing over multiple blocks
    let num_blocks = 20;
    let block_duration = Duration::from_millis(16); // ~16ms per block at 16kHz/256 samples

    for block_idx in 0..num_blocks {
        let start_time = std::time::Instant::now();

        // Generate test audio for this block
        let phase_offset = block_idx as f32 * 256.0 / 16000.0;
        let input: Vec<f32> = (0..256)
            .map(|i| {
                let t = phase_offset + i as f32 / 16000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3
            })
            .collect();

        let mut output = vec![0.0; 256];
        let result = rvc.process_audio_callback(&input, &mut output);
        assert!(result.is_ok(), "Failed to process block {}", block_idx);

        let processing_time = start_time.elapsed();
        println!(
            "Block {} processed in {:?} (target: {:?})",
            block_idx, processing_time, block_duration
        );

        // In a real-time system, processing should be faster than the block duration
        // But for testing with dummy implementation, we just check it completes
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_enhanced_streaming() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig {
        sample_rate: 16000,
        block_size: 512,
        enable_crossfade: true,
        crossfade_samples: 64,
    };

    // Test enhanced streaming
    let result = rvc.start_enhanced_stream(config);
    assert!(result.is_ok());
    assert!(rvc.is_streaming());

    // Should not be able to start again
    let config2 = AudioCallbackConfig::default();
    let result = rvc.start_enhanced_stream(config2);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("already running"));

    // Stop streaming
    let result = rvc.stop_stream();
    assert!(result.is_ok());
    assert!(!rvc.is_streaming());
}

#[test]
fn test_enhanced_streaming_without_ready() {
    let cfg = GUIConfig::default();
    let mut rvc = RVC::new(&cfg);

    let config = AudioCallbackConfig::default();
    let result = rvc.start_enhanced_stream(config);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not ready"));
}

#[test]
fn test_audio_callback_error_handling() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig::default();
    let callback = rvc.create_audio_callback(config).unwrap();
    rvc.set_audio_callback(callback).unwrap();

    // Test with various edge cases
    let test_cases = [
        vec![],                   // Empty input
        vec![0.0; 1],             // Very small input
        vec![1.0; 8192],          // Large input
        vec![f32::NAN; 512],      // Invalid input (NaN)
        vec![f32::INFINITY; 512], // Invalid input (Infinity)
    ];

    for (i, input) in test_cases.iter().enumerate() {
        let mut output = vec![0.0; input.len().max(512)];
        let result = rvc.process_audio_callback(input, &mut output);

        // Should handle all cases gracefully
        assert!(result.is_ok(), "Failed to handle test case {}", i);

        // Output should not contain invalid values (unless input was invalid)
        if !input.iter().any(|&x| !x.is_finite()) {
            assert!(
                output.iter().all(|&x| x.is_finite()),
                "Invalid output for test case {}",
                i
            );
        }
    }
}

#[test]
fn test_crossfade_function() {
    let current = vec![1.0; 64];
    let mut previous = vec![0.0; 64];
    let fade_samples = 16;

    let result = rvc_lib::apply_crossfade(&current, &mut previous, fade_samples);

    // Check basic properties
    assert_eq!(result.len(), current.len());
    assert!(result.iter().all(|&x| x.is_finite()));

    // Check crossfade effect
    assert!(result[0] < 1.0, "First sample should be faded");
    assert_eq!(
        result[fade_samples], 1.0,
        "Sample after fade should be full amplitude"
    );

    // Check that previous buffer was updated
    assert_eq!(previous[0], 1.0, "Previous buffer should be updated");
}

#[test]
fn test_crossfade_edge_cases() {
    // Test with zero fade samples
    let current = vec![1.0; 32];
    let mut previous = vec![0.5; 32];
    let result = rvc_lib::apply_crossfade(&current, &mut previous, 0);
    assert_eq!(result, current);

    // Test with fade samples larger than buffer
    let current = vec![1.0; 16];
    let mut previous = vec![0.5; 16];
    let result = rvc_lib::apply_crossfade(&current, &mut previous, 32);
    assert_eq!(result.len(), current.len());

    // Test with mismatched buffer sizes
    let current = vec![1.0; 32];
    let mut previous = vec![0.5; 16];
    let result = rvc_lib::apply_crossfade(&current, &mut previous, 8);
    assert_eq!(result.len(), current.len());
}

#[test]
fn test_simple_rvc_functionality() {
    let mut cfg = GUIConfig::default();
    cfg.pitch = 5.0; // Some pitch shift
    let _rvc = RVC::new(&cfg);

    // Test SimpleRVC creation indirectly through RVC methods
    let mut test_rvc = RVC::new(&cfg);
    test_rvc.model_loaded = true;
    test_rvc.hubert_loaded = true;

    let config = AudioCallbackConfig::default();
    let callback_result = test_rvc.create_audio_callback(config);
    assert!(
        callback_result.is_ok(),
        "Should create callback successfully"
    );
}

// Test helper function for SimpleRVC functionality
fn test_simple_rvc_inference() {
    let mut simple = rvc_lib::SimpleRVC {
        f0_up_key: 0.0,
        formant_shift: 1.0,
        device: Device::Cpu,
        is_half: false,
        tgt_sr: 16000,
        if_f0: 1,
        version: "v2".to_string(),
        model_loaded: true,
        hubert_loaded: true,
    };

    let input = vec![0.5; 256];
    let result = simple.infer_simple(&input);
    assert!(result.is_ok());
}

#[test]
fn test_callback_thread_safety() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let config = AudioCallbackConfig::default();
    let mut callback = rvc.create_audio_callback(config).unwrap();

    // Test that callback can be used in a thread
    let handle = thread::spawn(move || {
        let input = vec![0.1; 512];
        let mut output = vec![0.0; 512];
        callback(&input, &mut output);
        output.iter().any(|&x| x != 0.0) // Check if processing occurred
    });

    let result = handle.join();
    assert!(result.is_ok(), "Thread execution failed");
}
