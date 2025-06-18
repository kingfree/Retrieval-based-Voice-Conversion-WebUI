use rvc_lib::{GUIConfig, RVC, Harvest};
use std::collections::HashMap;

fn generate_test_signal(frequency: f32, duration: f32, sample_rate: f32) -> Vec<f32> {
    let mut signal = Vec::new();
    let samples = (duration * sample_rate) as usize;

    for i in 0..samples {
        let t = i as f32 / sample_rate;
        signal.push((2.0 * std::f32::consts::PI * frequency * t).sin() * 0.7);
    }

    signal
}

fn generate_harmonic_signal(_fundamental: f32, harmonics: &[f32], duration: f32, sample_rate: f32) -> Vec<f32> {
    let mut signal = vec![0.0; (duration * sample_rate) as usize];

    for (i, &harmonic_freq) in harmonics.iter().enumerate() {
        let amplitude = 1.0 / (i + 1) as f32; // Decreasing amplitude for higher harmonics
        for (j, sample) in signal.iter_mut().enumerate() {
            let t = j as f32 / sample_rate;
            *sample += amplitude * (2.0 * std::f32::consts::PI * harmonic_freq * t).sin();
        }
    }

    // Normalize
    let max_val = signal.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_val > 0.0 {
        for sample in &mut signal {
            *sample = (*sample / max_val) * 0.7;
        }
    }

    signal
}

fn analyze_f0_accuracy(detected: &[f32], expected_freq: f32, tolerance: f32) -> (f32, f32, usize) {
    let non_zero: Vec<f32> = detected.iter().filter(|&&x| x > 0.0).cloned().collect();

    if non_zero.is_empty() {
        return (0.0, 0.0, 0);
    }

    let avg_freq = non_zero.iter().sum::<f32>() / non_zero.len() as f32;
    let accuracy = if (avg_freq - expected_freq).abs() <= tolerance {
        1.0
    } else {
        1.0 - ((avg_freq - expected_freq).abs() / expected_freq).min(1.0)
    };

    (avg_freq, accuracy, non_zero.len())
}

#[test]
fn test_f0_method_comparison_sine_waves() {
    println!("=== F0 Method Comparison: Pure Sine Waves ===");

    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);
    let harvest = Harvest::new(16000);

    let test_frequencies = [100.0, 200.0, 440.0, 880.0];
    let methods = ["harvest", "pm", "crepe", "rmvpe", "fcpe"];
    let mut results: HashMap<String, Vec<(f32, f32, f32)>> = HashMap::new();

    for method in &methods {
        results.insert(method.to_string(), Vec::new());
    }

    for &freq in &test_frequencies {
        println!("\n--- Testing {}Hz sine wave ---", freq);

        let signal = generate_test_signal(freq, 1.0, 16000.0);

        for &method in &methods {
            let (_coarse, f0) = if method == "harvest" {
                let f0_harvest: Vec<f32> = harvest.compute(&signal).into_iter().map(|x| x as f32).collect();
                (vec![1u8; f0_harvest.len()], f0_harvest)
            } else {
                rvc.get_f0(&signal, 0.0, method)
            };

            let (avg_freq, accuracy, detection_count) = analyze_f0_accuracy(&f0, freq, freq * 0.1);
            results.get_mut(method).unwrap().push((avg_freq, accuracy, detection_count as f32));

            println!("  {}: avg={:.2}Hz, accuracy={:.2}%, detected={}/{}",
                     method, avg_freq, accuracy * 100.0, detection_count, f0.len());
        }
    }

    // Calculate overall scores
    println!("\n=== Overall Performance Summary ===");
    for method in &methods {
        let method_results = results.get(*method).unwrap();
        let avg_accuracy = method_results.iter().map(|x| x.1).sum::<f32>() / method_results.len() as f32;
        let avg_detection_rate = method_results.iter().map(|x| x.2).sum::<f32>() / method_results.len() as f32 / 100.0; // Assume 100 frames typical

        println!("{}: Average accuracy = {:.1}%, Average detection rate = {:.1}%",
                 method, avg_accuracy * 100.0, avg_detection_rate * 100.0);
    }
}

#[test]
fn test_f0_method_comparison_harmonic_signals() {
    println!("=== F0 Method Comparison: Harmonic Signals ===");

    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);

    let test_cases = [
        (110.0, vec![110.0, 220.0, 330.0]), // A2 with harmonics
        (220.0, vec![220.0, 440.0, 660.0, 880.0]), // A3 with harmonics
    ];

    let methods = ["harvest", "pm", "crepe", "rmvpe", "fcpe"];

    for (fundamental, harmonics) in &test_cases {
        println!("\n--- Testing {}Hz with harmonics {:?} ---", fundamental, harmonics);

        let signal = generate_harmonic_signal(*fundamental, harmonics, 1.0, 16000.0);

        for &method in &methods {
            let (_coarse, f0) = rvc.get_f0(&signal, 0.0, method);
            let (avg_freq, accuracy, detection_count) = analyze_f0_accuracy(&f0, *fundamental, fundamental * 0.15);

            println!("  {}: avg={:.2}Hz, accuracy={:.2}%, detected={}/{}",
                     method, avg_freq, accuracy * 100.0, detection_count, f0.len());
        }
    }
}

#[test]
fn test_f0_method_noise_robustness() {
    println!("=== F0 Method Noise Robustness Test ===");

    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);

    let base_freq = 220.0;
    let signal = generate_test_signal(base_freq, 1.0, 16000.0);
    let methods = ["harvest", "pm", "crepe", "rmvpe", "fcpe"];
    let noise_levels = [0.0, 0.1, 0.2, 0.3]; // SNR levels

    for &noise_level in &noise_levels {
        println!("\n--- Noise level: {:.1} ---", noise_level);

        // Add noise to signal
        let mut noisy_signal = signal.clone();
        if noise_level > 0.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            (noise_level as f32).to_bits().hash(&mut hasher);
            let seed = hasher.finish();

            // Simple pseudo-random noise generation
            for (i, sample) in noisy_signal.iter_mut().enumerate() {
                let noise_val = ((seed.wrapping_add(i as u64).wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f32 / 32768.0 - 1.0;
                *sample += noise_val * noise_level;
            }
        }

        for &method in &methods {
            let (_coarse, f0) = rvc.get_f0(&noisy_signal, 0.0, method);
            let (avg_freq, accuracy, detection_count) = analyze_f0_accuracy(&f0, base_freq, base_freq * 0.2);

            println!("  {}: avg={:.2}Hz, accuracy={:.2}%, detected={}/{}",
                     method, avg_freq, accuracy * 100.0, detection_count, f0.len());
        }
    }
}

#[test]
fn test_f0_method_pitch_shift_accuracy() {
    println!("=== F0 Method Pitch Shift Accuracy Test ===");

    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);

    let base_freq = 220.0;
    let signal = generate_test_signal(base_freq, 1.0, 16000.0);
    let methods = ["harvest", "pm", "crepe", "rmvpe", "fcpe"];
    let pitch_shifts = [0.0, 12.0, -12.0, 7.0, -7.0]; // semitones

    for &pitch_shift in &pitch_shifts {
        let expected_freq = base_freq * (2.0f32).powf(pitch_shift / 12.0);
        println!("\n--- Pitch shift: {} semitones (expected: {:.2}Hz) ---", pitch_shift, expected_freq);

        for &method in &methods {
            let (_coarse, f0) = rvc.get_f0(&signal, pitch_shift, method);
            let (avg_freq, accuracy, detection_count) = analyze_f0_accuracy(&f0, expected_freq, expected_freq * 0.15);

            println!("  {}: avg={:.2}Hz, accuracy={:.2}%, detected={}/{}",
                     method, avg_freq, accuracy * 100.0, detection_count, f0.len());
        }
    }
}

#[test]
fn test_f0_method_performance() {
    println!("=== F0 Method Performance Test ===");

    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);
    let harvest = Harvest::new(16000);

    // Create a longer signal for performance testing
    let signal = generate_test_signal(440.0, 5.0, 16000.0); // 5 seconds
    let methods = ["harvest", "pm", "crepe", "rmvpe", "fcpe"];

    println!("Testing with {:.1}s signal ({} samples)", 5.0, signal.len());

    for &method in &methods {
        let start_time = std::time::Instant::now();

        let (_coarse, f0) = if method == "harvest" {
            let f0_harvest: Vec<f32> = harvest.compute(&signal).into_iter().map(|x| x as f32).collect();
            (vec![1u8; f0_harvest.len()], f0_harvest)
        } else {
            rvc.get_f0(&signal, 0.0, method)
        };

        let elapsed = start_time.elapsed();
        let (avg_freq, accuracy, detection_count) = analyze_f0_accuracy(&f0, 440.0, 44.0);

        println!("  {}: {:.2?}, avg={:.2}Hz, accuracy={:.2}, detected={}/{}",
                 method, elapsed, avg_freq, accuracy, detection_count, f0.len());
    }
}

#[test]
fn test_f0_edge_cases() {
    println!("=== F0 Method Edge Cases Test ===");

    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);

    let methods = ["harvest", "pm", "crepe", "rmvpe", "fcpe"];

    // Test case 1: Very short signal
    println!("\n--- Very short signal (50ms) ---");
    let short_signal = generate_test_signal(200.0, 0.05, 16000.0);
    for &method in &methods {
        let (_coarse, f0) = rvc.get_f0(&short_signal, 0.0, method);
        println!("  {}: {} frames", method, f0.len());
    }

    // Test case 2: Very low frequency
    println!("\n--- Very low frequency (60Hz) ---");
    let low_freq_signal = generate_test_signal(60.0, 1.0, 16000.0);
    for &method in &methods {
        let (_coarse, f0) = rvc.get_f0(&low_freq_signal, 0.0, method);
        let (avg_freq, _, detection_count) = analyze_f0_accuracy(&f0, 60.0, 12.0);
        println!("  {}: avg={:.2}Hz, detected={}/{}", method, avg_freq, detection_count, f0.len());
    }

    // Test case 3: Very high frequency
    println!("\n--- High frequency (1000Hz) ---");
    let high_freq_signal = generate_test_signal(1000.0, 1.0, 16000.0);
    for &method in &methods {
        let (_coarse, f0) = rvc.get_f0(&high_freq_signal, 0.0, method);
        let (avg_freq, _, detection_count) = analyze_f0_accuracy(&f0, 1000.0, 100.0);
        println!("  {}: avg={:.2}Hz, detected={}/{}", method, avg_freq, detection_count, f0.len());
    }

    // Test case 4: Empty signal
    println!("\n--- Empty signal ---");
    let empty_signal: Vec<f32> = vec![];
    for &method in &methods {
        let (_coarse, f0) = rvc.get_f0(&empty_signal, 0.0, method);
        println!("  {}: {} frames", method, f0.len());
        assert!(f0.is_empty() || f0.iter().all(|&x| x == 0.0));
    }
}

#[tokio::test]
async fn test_harvest_async_performance() {
    println!("=== Harvest Async vs Sync Performance ===");

    let harvest = Harvest::new(16000);
    let signal = generate_test_signal(440.0, 2.0, 16000.0);

    // Sync test
    let start_time = std::time::Instant::now();
    let f0_sync = harvest.compute(&signal);
    let sync_duration = start_time.elapsed();

    // Async test
    let start_time = std::time::Instant::now();
    let f0_async = harvest.compute_async(signal.clone()).await;
    let async_duration = start_time.elapsed();

    println!("Sync:  {:.2?}, {} frames", sync_duration, f0_sync.len());
    println!("Async: {:.2?}, {} frames", async_duration, f0_async.len());

    // Results should be identical
    assert_eq!(f0_sync.len(), f0_async.len());
    for (i, (s, a)) in f0_sync.iter().zip(f0_async.iter()).enumerate() {
        assert!((s - a).abs() < 1e-10, "Mismatch at index {}: {} vs {}", i, s, a);
    }

    println!("Async and sync results are identical");
}
