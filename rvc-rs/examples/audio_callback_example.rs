//! Audio Callback Example for RVC Real-Time Voice Conversion
//!
//! This example demonstrates how to use the RVC audio callback system
//! for real-time voice conversion processing.

use rvc_lib::{AudioCallbackConfig, GUIConfig, RVC};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RVC Audio Callback Example");
    println!("==========================");

    // 1. Initialize RVC with configuration
    println!("1. Initializing RVC...");
    let mut cfg = GUIConfig::default();
    cfg.pth_path = "model.pth".to_string();
    cfg.pitch = 5.0; // Pitch shift up by 5 semitones
    cfg.formant = 1.1; // Slight formant shift

    let mut rvc = RVC::new(&cfg);

    // For this example, we'll simulate loaded models
    // In a real application, you would load actual model files
    rvc.model_loaded = true;
    rvc.hubert_loaded = true;

    println!(
        "   RVC initialized with pitch shift: {} semitones",
        cfg.pitch
    );

    // 2. Check if RVC is ready for processing
    if !rvc.is_ready() {
        return Err("RVC is not ready - models not loaded".into());
    }
    println!("   RVC is ready for processing");

    // 3. Create audio callback configuration
    println!("2. Creating audio callback...");
    let config = AudioCallbackConfig {
        sample_rate: 16000,
        block_size: 512,
        enable_crossfade: true,
        crossfade_samples: 64,
    };

    println!("   Configuration:");
    println!("     Sample rate: {} Hz", config.sample_rate);
    println!("     Block size: {} samples", config.block_size);
    println!(
        "     Crossfade: {} ({} samples)",
        if config.enable_crossfade {
            "enabled"
        } else {
            "disabled"
        },
        config.crossfade_samples
    );

    // 4. Create and set the audio callback
    let callback = rvc.create_audio_callback(config)?;
    rvc.set_audio_callback(callback)?;
    println!("   Audio callback created and registered");

    // 5. Simulate real-time audio processing
    println!("3. Simulating real-time audio processing...");

    let num_blocks = 10;
    let block_duration = Duration::from_millis(32); // ~32ms per block

    for block_idx in 0..num_blocks {
        let start_time = Instant::now();

        // Generate test audio (sine wave with varying frequency)
        let frequency = 220.0 + (block_idx as f32 * 50.0); // 220Hz to 670Hz
        let input_samples = generate_sine_wave(frequency, 16000, 512);

        // Prepare output buffer
        let mut output_samples = vec![0.0; 512];

        // Process audio through RVC callback
        rvc.process_audio_callback(&input_samples, &mut output_samples)?;

        let processing_time = start_time.elapsed();

        // Analyze output
        let input_rms = calculate_rms(&input_samples);
        let output_rms = calculate_rms(&output_samples);

        println!(
            "   Block {}: freq={:.0}Hz, processing_time={:.2}ms, input_rms={:.3}, output_rms={:.3}",
            block_idx + 1,
            frequency,
            processing_time.as_secs_f64() * 1000.0,
            input_rms,
            output_rms
        );

        // Simulate real-time constraints
        if processing_time < block_duration {
            std::thread::sleep(block_duration - processing_time);
        } else {
            println!("     Warning: Processing time exceeded block duration!");
        }
    }

    // 6. Test different configurations
    println!("4. Testing different configurations...");

    test_different_sample_rates(&mut rvc)?;
    test_different_block_sizes(&mut rvc)?;
    test_crossfade_settings(&mut rvc)?;

    // 7. Test enhanced streaming
    println!("5. Testing enhanced streaming...");
    test_enhanced_streaming(&mut rvc)?;

    // 8. Cleanup
    println!("6. Cleaning up...");
    rvc.clear_audio_callback();
    println!("   Audio callback cleared");

    println!("\nExample completed successfully!");
    println!("The RVC audio callback system is ready for integration with your audio application.");

    Ok(())
}

/// Generate a sine wave with specified parameters
fn generate_sine_wave(frequency: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect()
}

/// Calculate RMS (Root Mean Square) of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Test different sample rates
fn test_different_sample_rates(rvc: &mut RVC) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rates = [8000, 16000, 22050, 44100, 48000];

    for &sample_rate in &sample_rates {
        let config = AudioCallbackConfig {
            sample_rate,
            block_size: 512,
            enable_crossfade: true,
            crossfade_samples: 64,
        };

        let callback = rvc.create_audio_callback(config)?;
        rvc.set_audio_callback(callback)?;

        // Test processing
        let input = generate_sine_wave(440.0, sample_rate, 512);
        let mut output = vec![0.0; 512];
        rvc.process_audio_callback(&input, &mut output)?;

        println!("   Sample rate {} Hz: OK", sample_rate);
        rvc.clear_audio_callback();
    }

    Ok(())
}

/// Test different block sizes
fn test_different_block_sizes(rvc: &mut RVC) -> Result<(), Box<dyn std::error::Error>> {
    let block_sizes = [64, 128, 256, 512, 1024, 2048];

    for &block_size in &block_sizes {
        let config = AudioCallbackConfig {
            sample_rate: 16000,
            block_size,
            enable_crossfade: true,
            crossfade_samples: (block_size / 8).min(128),
        };

        let callback = rvc.create_audio_callback(config)?;
        rvc.set_audio_callback(callback)?;

        // Test processing
        let input = generate_sine_wave(440.0, 16000, block_size);
        let mut output = vec![0.0; block_size];
        rvc.process_audio_callback(&input, &mut output)?;

        println!("   Block size {} samples: OK", block_size);
        rvc.clear_audio_callback();
    }

    Ok(())
}

/// Test crossfade settings
fn test_crossfade_settings(rvc: &mut RVC) -> Result<(), Box<dyn std::error::Error>> {
    let crossfade_configs = [
        (false, 0),  // Disabled
        (true, 16),  // Small
        (true, 64),  // Medium
        (true, 128), // Large
    ];

    for &(enable_crossfade, crossfade_samples) in &crossfade_configs {
        let config = AudioCallbackConfig {
            sample_rate: 16000,
            block_size: 512,
            enable_crossfade,
            crossfade_samples,
        };

        let callback = rvc.create_audio_callback(config)?;
        rvc.set_audio_callback(callback)?;

        // Test processing multiple blocks to test crossfade
        for _ in 0..3 {
            let input = generate_sine_wave(440.0, 16000, 512);
            let mut output = vec![0.0; 512];
            rvc.process_audio_callback(&input, &mut output)?;
        }

        let status = if enable_crossfade {
            format!("enabled ({} samples)", crossfade_samples)
        } else {
            "disabled".to_string()
        };
        println!("   Crossfade {}: OK", status);
        rvc.clear_audio_callback();
    }

    Ok(())
}

/// Test enhanced streaming functionality
fn test_enhanced_streaming(rvc: &mut RVC) -> Result<(), Box<dyn std::error::Error>> {
    let config = AudioCallbackConfig {
        sample_rate: 16000,
        block_size: 256,
        enable_crossfade: true,
        crossfade_samples: 32,
    };

    // Start enhanced streaming
    rvc.start_enhanced_stream(config)?;
    println!("   Enhanced streaming started");

    // Process several chunks
    for i in 0..5 {
        let frequency = 330.0 + (i as f32 * 110.0);
        let chunk = generate_sine_wave(frequency, 16000, 256);
        let processed = rvc.process_stream_chunk(&chunk)?;

        let input_rms = calculate_rms(&chunk);
        let output_rms = calculate_rms(&processed);
        println!(
            "     Chunk {}: freq={:.0}Hz, input_rms={:.3}, output_rms={:.3}",
            i + 1,
            frequency,
            input_rms,
            output_rms
        );
    }

    // Stop streaming
    rvc.stop_stream()?;
    println!("   Enhanced streaming stopped");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave_generation() {
        let samples = generate_sine_wave(440.0, 16000, 16000); // 1 second
        assert_eq!(samples.len(), 16000);

        // Check that it's actually a sine wave (should have peaks)
        let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_val > 0.4 && max_val < 0.6); // Should be around 0.5
    }

    #[test]
    fn test_rms_calculation() {
        let silence = vec![0.0; 100];
        assert_eq!(calculate_rms(&silence), 0.0);

        let constant = vec![0.5; 100];
        assert!((calculate_rms(&constant) - 0.5).abs() < 0.001);

        let empty: Vec<f32> = vec![];
        assert_eq!(calculate_rms(&empty), 0.0);
    }
}
