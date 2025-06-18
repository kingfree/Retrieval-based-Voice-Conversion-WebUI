use rvc_lib::{GUIConfig, Harvest, RVC};
use std::fs;

#[test]
fn test_harvest_detailed_debug() {
    let harvest = Harvest::new(16000);

    // Test 1: Zero signal
    println!("=== Testing zero signal ===");
    let zero_signal = vec![0.0f32; 1600]; // 100ms of zeros at 16kHz
    let f0_zero = harvest.compute(&zero_signal);
    println!("Zero signal F0 length: {}", f0_zero.len());
    println!(
        "Zero signal F0 values: {:?}",
        &f0_zero[..f0_zero.len().min(10)]
    );
    assert!(
        f0_zero.iter().all(|&v| v == 0.0),
        "Zero signal should produce zero F0"
    );

    // Test 2: Simple 100Hz sine wave
    println!("\n=== Testing 100Hz sine wave ===");
    let sample_rate = 16000.0;
    let duration = 1.0; // 1 second
    let frequency = 100.0;
    let mut sine_wave = Vec::new();
    for i in 0..(sample_rate * duration) as usize {
        let t = i as f32 / sample_rate;
        sine_wave.push((2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5);
    }

    let f0_sine = harvest.compute(&sine_wave);
    println!("Sine wave F0 length: {}", f0_sine.len());
    println!(
        "Sine wave F0 values (first 10): {:?}",
        &f0_sine[..f0_sine.len().min(10)]
    );

    // Count non-zero values
    let non_zero_count = f0_sine.iter().filter(|&&v| v > 0.0).count();
    println!("Non-zero F0 values: {}/{}", non_zero_count, f0_sine.len());

    if non_zero_count > 0 {
        let detected_freqs: Vec<f64> = f0_sine.iter().filter(|&&v| v > 0.0).cloned().collect();
        let avg_freq = detected_freqs.iter().sum::<f64>() / detected_freqs.len() as f64;
        println!("Average detected frequency: {:.2} Hz", avg_freq);

        // Should be around 100 Hz
        assert!(
            avg_freq >= 80.0 && avg_freq <= 120.0,
            "Expected ~100 Hz, got {:.2} Hz",
            avg_freq
        );
    } else {
        println!("WARNING: No F0 detected for 100Hz sine wave!");
    }

    // Test 3: Compare with RVC implementation
    println!("\n=== Comparing with RVC PM method ===");
    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);
    let (_coarse, f0_pm) = rvc.get_f0(&sine_wave, 0.0, "pm");
    let pm_non_zero = f0_pm.iter().filter(|&&v| v > 0.0).count();
    println!("PM method non-zero values: {}/{}", pm_non_zero, f0_pm.len());

    if pm_non_zero > 0 {
        let pm_detected: Vec<f32> = f0_pm.iter().filter(|&&v| v > 0.0).cloned().collect();
        let pm_avg = pm_detected.iter().sum::<f32>() / pm_detected.len() as f32;
        println!("PM average frequency: {:.2} Hz", pm_avg);
    }
}

#[test]
fn test_harvest_with_test_data() {
    // Try to load existing test data
    let signal_path = "tests/data/sine100_signal.json";
    let expected_path = "tests/data/sine100_f0.json";

    if !std::path::Path::new(signal_path).exists() {
        println!("Test data file {} not found, skipping", signal_path);
        return;
    }

    println!("=== Testing with existing test data ===");
    let signal_data = fs::read_to_string(signal_path).expect("read signal");
    let signal: Vec<f32> = serde_json::from_str(&signal_data).expect("parse signal");

    let expected_data = fs::read_to_string(expected_path).expect("read expected");
    let expected: Vec<f32> = serde_json::from_str(&expected_data).expect("parse expected");

    println!("Signal length: {}", signal.len());
    println!("Expected F0 length: {}", expected.len());

    let harvest = Harvest::new(16000);
    let f0_harvest: Vec<f32> = harvest
        .compute(&signal)
        .into_iter()
        .map(|v| v as f32)
        .collect();

    println!("Harvest F0 length: {}", f0_harvest.len());
    println!(
        "First 10 harvest values: {:?}",
        &f0_harvest[..f0_harvest.len().min(10)]
    );
    println!(
        "First 10 expected values: {:?}",
        &expected[..expected.len().min(10)]
    );

    let harvest_non_zero = f0_harvest.iter().filter(|&&v| v > 0.0).count();
    let expected_non_zero = expected.iter().filter(|&&v| v > 0.0).count();

    println!(
        "Harvest non-zero: {}/{}",
        harvest_non_zero,
        f0_harvest.len()
    );
    println!(
        "Expected non-zero: {}/{}",
        expected_non_zero,
        expected.len()
    );

    // Compare RVC's harvest implementation
    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);
    let (_coarse, f0_rvc_harvest) = rvc.get_f0(&signal, 0.0, "harvest");
    let rvc_harvest_non_zero = f0_rvc_harvest.iter().filter(|&&v| v > 0.0).count();

    println!(
        "RVC Harvest non-zero: {}/{}",
        rvc_harvest_non_zero,
        f0_rvc_harvest.len()
    );
    println!(
        "First 10 RVC harvest values: {:?}",
        &f0_rvc_harvest[..f0_rvc_harvest.len().min(10)]
    );
}

#[test]
fn test_harvest_parameter_sensitivity() {
    println!("=== Testing Harvest parameter sensitivity ===");

    // Create a clear 200Hz sine wave
    let sample_rate = 16000.0;
    let duration = 0.5; // 500ms
    let frequency = 200.0;
    let mut sine_wave = Vec::new();
    for i in 0..(sample_rate * duration) as usize {
        let t = i as f32 / sample_rate;
        sine_wave.push((2.0 * std::f32::consts::PI * frequency * t).sin() * 0.8);
    }

    // Test different parameter combinations
    let test_params = [
        (50.0, 1100.0, 10.0), // Default
        (40.0, 1200.0, 10.0), // Wider range
        (80.0, 800.0, 10.0),  // Narrower range
        (50.0, 1100.0, 5.0),  // Shorter frame period
        (50.0, 1100.0, 20.0), // Longer frame period
    ];

    for (i, (f0_floor, f0_ceil, frame_period)) in test_params.iter().enumerate() {
        println!(
            "\nTest {}: f0_floor={}, f0_ceil={}, frame_period={}",
            i + 1,
            f0_floor,
            f0_ceil,
            frame_period
        );

        let harvest = Harvest::new(16000);
        // Note: We can't actually change parameters with current API
        // This would require extending the Harvest struct

        let f0 = harvest.compute(&sine_wave);
        let non_zero_count = f0.iter().filter(|&&v| v > 0.0).count();

        if non_zero_count > 0 {
            let detected: Vec<f64> = f0.iter().filter(|&&v| v > 0.0).cloned().collect();
            let avg = detected.iter().sum::<f64>() / detected.len() as f64;
            println!(
                "  Detected {}/{} frames, avg freq: {:.2} Hz",
                non_zero_count,
                f0.len(),
                avg
            );
        } else {
            println!("  No F0 detected");
        }
    }
}

#[tokio::test]
async fn test_harvest_async() {
    println!("=== Testing Harvest async functionality ===");

    let harvest = Harvest::new(16000);

    // Test with a longer signal
    let sample_rate = 16000.0;
    let duration = 2.0; // 2 seconds
    let frequency = 440.0; // A4 note
    let mut sine_wave = Vec::new();
    for i in 0..(sample_rate * duration) as usize {
        let t = i as f32 / sample_rate;
        sine_wave.push((2.0 * std::f32::consts::PI * frequency * t).sin() * 0.7);
    }

    println!("Processing {} samples asynchronously...", sine_wave.len());

    let start_time = std::time::Instant::now();
    let f0_async = harvest.compute_async(sine_wave.clone()).await;
    let async_duration = start_time.elapsed();

    let start_time = std::time::Instant::now();
    let f0_sync = harvest.compute(&sine_wave);
    let sync_duration = start_time.elapsed();

    println!("Async duration: {:?}", async_duration);
    println!("Sync duration: {:?}", sync_duration);

    let async_f0_f32: Vec<f32> = f0_async.into_iter().map(|v| v as f32).collect();
    let sync_f0_f32: Vec<f32> = f0_sync.into_iter().map(|v| v as f32).collect();

    // Results should be identical or very close
    assert_eq!(async_f0_f32.len(), sync_f0_f32.len());

    let mut diff_count = 0;
    for (i, (a, s)) in async_f0_f32.iter().zip(sync_f0_f32.iter()).enumerate() {
        if (a - s).abs() > 1e-6 {
            if diff_count < 5 {
                println!("Difference at index {}: async={}, sync={}", i, a, s);
            }
            diff_count += 1;
        }
    }

    if diff_count == 0 {
        println!("Async and sync results are identical");
    } else {
        println!(
            "Found {} differences between async and sync results",
            diff_count
        );
    }
}
