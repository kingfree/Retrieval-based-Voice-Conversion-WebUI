use rvc_lib::{GUIConfig, RVC};

#[test]
fn test_streaming_lifecycle() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Initially should not be streaming
    assert!(!rvc.is_streaming());
    assert!(rvc.get_stream_info().is_none());

    // Start streaming
    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok(), "Failed to start stream: {:?}", result);

    // Should now be streaming
    assert!(rvc.is_streaming());
    let stream_info = rvc.get_stream_info();
    assert!(stream_info.is_some());
    assert!(stream_info.unwrap().contains("Streaming active"));

    // Stop streaming
    let result = rvc.stop_stream();
    assert!(result.is_ok(), "Failed to stop stream: {:?}", result);

    // Should no longer be streaming
    assert!(!rvc.is_streaming());
    assert!(rvc.get_stream_info().is_none());
}

#[test]
fn test_streaming_with_different_configurations() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Test different sample rates and block sizes
    let configs = [
        (16000, 256),
        (16000, 512),
        (16000, 1024),
        (22050, 512),
        (44100, 1024),
        (48000, 2048),
    ];

    for (sample_rate, block_size) in configs {
        let result = rvc.start_stream(sample_rate, block_size);
        assert!(
            result.is_ok(),
            "Failed to start stream with sr={}, bs={}: {:?}",
            sample_rate,
            block_size,
            result
        );

        assert!(rvc.is_streaming());

        let result = rvc.stop_stream();
        assert!(
            result.is_ok(),
            "Failed to stop stream with sr={}, bs={}: {:?}",
            sample_rate,
            block_size,
            result
        );

        assert!(!rvc.is_streaming());
    }
}

#[test]
fn test_streaming_audio_processing() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Test processing different types of audio data
    let test_cases = [
        // Silence
        vec![0.0; 512],
        // Sine wave (440 Hz at 16kHz for ~32ms)
        (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect::<Vec<f32>>(),
        // Random noise (using simple deterministic pattern for testing)
        (0..512)
            .map(|i| ((i as f32 * 17.0).sin() * 0.1))
            .collect::<Vec<f32>>(),
        // Chirp (frequency sweep)
        (0..512)
            .map(|i| {
                let t = i as f32 / 16000.0;
                let freq = 200.0 + 800.0 * t; // 200Hz to 1000Hz sweep
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3
            })
            .collect::<Vec<f32>>(),
    ];

    for (i, input_data) in test_cases.iter().enumerate() {
        let result = rvc.process_stream_chunk(input_data);
        assert!(
            result.is_ok(),
            "Failed to process chunk {}: {:?}",
            i,
            result
        );

        let output = result.unwrap();
        assert_eq!(
            output.len(),
            input_data.len(),
            "Output length mismatch for chunk {}",
            i
        );

        // Output should be finite numbers
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Non-finite values in output for chunk {}",
            i
        );
    }

    let result = rvc.stop_stream();
    assert!(result.is_ok());
}

#[test]
fn test_streaming_with_pitch_parameters() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    cfg.pitch = 12.0; // One octave up
    cfg.formant = 1.2; // Formant shift
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Generate test audio (440 Hz sine wave)
    let input_data: Vec<f32> = (0..512)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect();

    let result = rvc.process_stream_chunk(&input_data);
    assert!(result.is_ok());

    let output = result.unwrap();

    // With pitch shift, output length may differ from input due to resampling
    // but should still be a reasonable length and contain finite values
    assert!(output.len() > 0, "Output should not be empty");
    assert!(
        output.len() <= input_data.len() * 2,
        "Output should not be more than 2x input length"
    );
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "All output values should be finite"
    );

    let result = rvc.stop_stream();
    assert!(result.is_ok());
}

#[test]
fn test_streaming_error_conditions() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    // Test processing without starting stream
    let input_data = vec![0.0; 512];
    let result = rvc.process_stream_chunk(&input_data);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Streaming not active"));

    // Start stream
    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Try to start stream again (should fail)
    let result = rvc.start_stream(16000, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Stream already running"));

    // Stop stream
    let result = rvc.stop_stream();
    assert!(result.is_ok());

    // Try to stop stream again (should fail)
    let result = rvc.stop_stream();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No stream running"));
}

#[test]
fn test_streaming_without_models() {
    let cfg = GUIConfig::default();
    let mut rvc = RVC::new(&cfg);

    // Models are not loaded, should fail to start streaming
    let result = rvc.start_stream(16000, 512);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not ready"));
}

#[test]
fn test_streaming_parameter_updates_during_stream() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Test updating parameters during streaming
    let _original_pitch = rvc.f0_up_key;
    let _original_formant = rvc.formant_shift;
    let _original_index_rate = rvc.index_rate;

    rvc.change_key(5.0);
    assert_eq!(rvc.f0_up_key, 5.0);

    rvc.change_formant(1.5);
    assert_eq!(rvc.formant_shift, 1.5);

    rvc.change_index_rate(0.8);
    assert_eq!(rvc.index_rate, 0.8);

    // Should still be able to process audio with new parameters
    let input_data = vec![0.5; 512];
    let result = rvc.process_stream_chunk(&input_data);
    assert!(result.is_ok());

    let result = rvc.stop_stream();
    assert!(result.is_ok());
}

#[test]
fn test_streaming_buffer_management() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Process multiple chunks to test buffer management
    for i in 0..10 {
        let input_data: Vec<f32> = (0..512)
            .map(|j| {
                ((i * 512 + j) as f32 / 16000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.3
            })
            .collect();

        let result = rvc.process_stream_chunk(&input_data);
        assert!(result.is_ok(), "Failed to process chunk {}", i);

        let output = result.unwrap();
        assert_eq!(output.len(), 512);
    }

    // Check stream info (should show buffer activity)
    let stream_info = rvc.get_stream_info();
    assert!(stream_info.is_some());

    let result = rvc.stop_stream();
    assert!(result.is_ok());
}

#[test]
fn test_streaming_with_different_chunk_sizes() {
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "mock_model.pth".to_string();
    let mut rvc = RVC::new(&cfg);

    // Force model loaded state for testing
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    let result = rvc.start_stream(16000, 512);
    assert!(result.is_ok());

    // Test processing chunks of different sizes
    let chunk_sizes = [64, 128, 256, 512, 1024, 2048];

    for &chunk_size in &chunk_sizes {
        let input_data = vec![0.1; chunk_size];
        let result = rvc.process_stream_chunk(&input_data);
        assert!(
            result.is_ok(),
            "Failed to process chunk of size {}",
            chunk_size
        );

        let output = result.unwrap();
        assert_eq!(
            output.len(),
            chunk_size,
            "Output size mismatch for chunk size {}",
            chunk_size
        );
    }

    let result = rvc.stop_stream();
    assert!(result.is_ok());
}

#[test]
fn test_streaming_readiness_check() {
    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);

    // Without models loaded, should not be ready for streaming
    assert!(!rvc.is_ready());

    let mut cfg_with_model = GUIConfig::default();
    cfg_with_model.pth_path = "mock_model.pth".to_string();
    let mut rvc_with_model = RVC::new(&cfg_with_model);

    // Still not ready until models are actually loaded
    assert!(!rvc_with_model.is_ready());

    // Force model loaded state
    rvc_with_model.model_loaded = true;
    rvc_with_model.hubert_loaded = true;

    // Now should be ready
    assert!(rvc_with_model.is_ready());

    // Should be able to start streaming
    let result = rvc_with_model.start_stream(16000, 512);
    assert!(result.is_ok());

    let result = rvc_with_model.stop_stream();
    assert!(result.is_ok());
}
