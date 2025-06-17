#!/usr/bin/env python3
"""
Generate comprehensive test cases for F0 detection algorithms.

This script generates various audio signals and their expected F0 values
for testing the Rust implementation of F0 extraction methods.
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any

def save_test_case(name: str, signal: np.ndarray, f0_expected: np.ndarray,
                   sample_rate: int = 16000, metadata: Dict[str, Any] = None):
    """Save a test case as JSON files."""
    # Convert to float32 for consistency with Rust
    signal_f32 = signal.astype(np.float32).tolist()
    f0_f32 = f0_expected.astype(np.float32).tolist()

    # Save signal
    with open(f"{name}_signal.json", "w") as f:
        json.dump(signal_f32, f, indent=2)

    # Save expected F0
    with open(f"{name}_f0.json", "w") as f:
        json.dump(f0_f32, f, indent=2)

    # Save metadata
    meta = {
        "sample_rate": sample_rate,
        "signal_length": len(signal),
        "f0_length": len(f0_expected),
        "description": metadata.get("description", ""),
        "frequency": metadata.get("frequency", None),
        "duration": metadata.get("duration", len(signal) / sample_rate),
    }
    if metadata:
        meta.update(metadata)

    with open(f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Generated test case: {name}")
    print(f"  Signal length: {len(signal)} samples")
    print(f"  F0 frames: {len(f0_expected)}")
    print(f"  Expected F0 range: {f0_expected.min():.2f} - {f0_expected.max():.2f} Hz")

def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000,
                      amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_complex_harmonic(fundamental: float, duration: float, sample_rate: int = 16000,
                            harmonics: List[Tuple[int, float]] = None) -> np.ndarray:
    """Generate a complex harmonic signal with multiple harmonics."""
    if harmonics is None:
        harmonics = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)]

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.zeros_like(t)

    for harmonic_num, amplitude in harmonics:
        signal += amplitude * np.sin(2 * np.pi * fundamental * harmonic_num * t)

    # Normalize
    return signal / np.max(np.abs(signal)) * 0.7

def generate_chirp(f_start: float, f_end: float, duration: float,
                  sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a linear chirp signal with varying frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Linear frequency sweep
    instantaneous_freq = f_start + (f_end - f_start) * t / duration
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
    signal = 0.5 * np.sin(phase)

    # Generate expected F0 (frame-based)
    hop_length = int(sample_rate * 0.01)  # 10ms frames
    frame_count = len(t) // hop_length
    f0_expected = np.zeros(frame_count)

    for i in range(frame_count):
        frame_center = (i * hop_length + hop_length // 2) / sample_rate
        if frame_center <= duration:
            f0_expected[i] = f_start + (f_end - f_start) * frame_center / duration

    return signal, f0_expected

def generate_noisy_sine(frequency: float, duration: float, noise_level: float = 0.1,
                       sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave with additive noise."""
    clean_signal = generate_sine_wave(frequency, duration, sample_rate)
    noise = np.random.normal(0, noise_level, len(clean_signal))
    return clean_signal + noise

def generate_voiced_unvoiced_segments(frequencies: List[float], durations: List[float],
                                    silence_duration: float = 0.02,
                                    sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate signal with alternating voiced and unvoiced segments."""
    segments = []
    f0_segments = []

    for i, (freq, dur) in enumerate(zip(frequencies, durations)):
        # Voiced segment
        if freq > 0:
            signal_seg = generate_sine_wave(freq, dur, sample_rate)
            hop_length = int(sample_rate * 0.01)
            frame_count = len(signal_seg) // hop_length
            f0_seg = np.full(frame_count, freq)
        else:
            # Unvoiced segment (noise)
            signal_seg = np.random.normal(0, 0.1, int(sample_rate * dur))
            hop_length = int(sample_rate * 0.01)
            frame_count = len(signal_seg) // hop_length
            f0_seg = np.zeros(frame_count)

        segments.append(signal_seg)
        f0_segments.append(f0_seg)

        # Add silence between segments
        if i < len(frequencies) - 1:
            silence_samples = int(sample_rate * silence_duration)
            segments.append(np.zeros(silence_samples))
            silence_frames = silence_samples // hop_length
            f0_segments.append(np.zeros(silence_frames))

    signal = np.concatenate(segments)
    f0_expected = np.concatenate(f0_segments)

    return signal, f0_expected

def compute_expected_f0_frames(signal: np.ndarray, true_frequency: float,
                             sample_rate: int = 16000) -> np.ndarray:
    """Compute expected F0 values for frame-based analysis."""
    hop_length = int(sample_rate * 0.01)  # 10ms frame period
    frame_count = (len(signal) + hop_length - 1) // hop_length

    # For a pure sine wave, all frames should detect the same frequency
    # except for very short signals or edge effects
    f0_expected = np.full(frame_count, true_frequency)

    # Set edge frames to 0 if the signal is too short
    if len(signal) < hop_length * 2:
        f0_expected[:] = 0.0

    return f0_expected

def main():
    """Generate all test cases."""
    print("Generating F0 detection test cases...")

    # Test case 1: Simple sine waves at different frequencies
    frequencies = [50, 100, 200, 440, 880, 1000]
    for freq in frequencies:
        signal = generate_sine_wave(freq, 0.1)  # 100ms
        f0_expected = compute_expected_f0_frames(signal, freq)
        save_test_case(
            f"sine_{freq}hz",
            signal,
            f0_expected,
            metadata={
                "description": f"Pure sine wave at {freq} Hz",
                "frequency": freq,
                "signal_type": "sine"
            }
        )

    # Test case 2: Zero signal (silence)
    silence = np.zeros(1600)  # 100ms of silence
    f0_silence = np.zeros(10)  # 10 frames of 0 Hz
    save_test_case(
        "silence",
        silence,
        f0_silence,
        metadata={
            "description": "Silent signal (all zeros)",
            "signal_type": "silence"
        }
    )

    # Test case 3: Complex harmonic signals
    harmonics_test = [
        (110, [(1, 1.0), (2, 0.5), (3, 0.25)]),  # Guitar-like
        (220, [(1, 1.0), (2, 0.3), (3, 0.1), (4, 0.05)]),  # Voice-like
    ]

    for fundamental, harmonics in harmonics_test:
        signal = generate_complex_harmonic(fundamental, 0.15, harmonics=harmonics)
        f0_expected = compute_expected_f0_frames(signal, fundamental)
        save_test_case(
            f"harmonic_{fundamental}hz",
            signal,
            f0_expected,
            metadata={
                "description": f"Complex harmonic signal with fundamental at {fundamental} Hz",
                "frequency": fundamental,
                "signal_type": "harmonic",
                "harmonics": harmonics
            }
        )

    # Test case 4: Frequency sweep (chirp)
    chirp_signal, chirp_f0 = generate_chirp(100, 500, 0.2)
    save_test_case(
        "chirp_100_500hz",
        chirp_signal,
        chirp_f0,
        metadata={
            "description": "Linear frequency sweep from 100 Hz to 500 Hz",
            "signal_type": "chirp",
            "f_start": 100,
            "f_end": 500
        }
    )

    # Test case 5: Noisy signals
    noise_levels = [0.05, 0.1, 0.2]
    for noise_level in noise_levels:
        np.random.seed(42)  # For reproducible noise
        signal = generate_noisy_sine(220, 0.1, noise_level)
        f0_expected = compute_expected_f0_frames(signal, 220)
        save_test_case(
            f"noisy_sine_220hz_noise{int(noise_level*100)}",
            signal,
            f0_expected,
            metadata={
                "description": f"220 Hz sine wave with {noise_level*100}% noise",
                "frequency": 220,
                "signal_type": "noisy_sine",
                "noise_level": noise_level
            }
        )

    # Test case 6: Voiced/unvoiced segments
    voices_unvoiced_signal, voices_unvoiced_f0 = generate_voiced_unvoiced_segments(
        [200, 0, 300, 0, 150],  # frequencies (0 = unvoiced)
        [0.05, 0.02, 0.05, 0.02, 0.05]  # durations
    )
    save_test_case(
        "voiced_unvoiced_segments",
        voices_unvoiced_signal,
        voices_unvoiced_f0,
        metadata={
            "description": "Alternating voiced and unvoiced segments",
            "signal_type": "voiced_unvoiced",
            "frequencies": [200, 0, 300, 0, 150],
            "durations": [0.05, 0.02, 0.05, 0.02, 0.05]
        }
    )

    # Test case 7: Very short signals
    short_signal = generate_sine_wave(440, 0.01)  # 10ms
    short_f0 = compute_expected_f0_frames(short_signal, 440)
    save_test_case(
        "short_sine_440hz",
        short_signal,
        short_f0,
        metadata={
            "description": "Very short 440 Hz sine wave (10ms)",
            "frequency": 440,
            "signal_type": "short_sine"
        }
    )

    # Test case 8: Edge case frequencies
    for freq in [51, 1099]:  # Approximating f0_min=50, f0_max=1100
        signal = generate_sine_wave(freq, 0.1)
        f0_expected = compute_expected_f0_frames(signal, freq)
        save_test_case(
            f"edge_freq_{freq}hz",
            signal,
            f0_expected,
            metadata={
                "description": f"Edge case frequency at {freq} Hz",
                "frequency": freq,
                "signal_type": "edge_case"
            }
        )

    # Test case 9: Amplitude variations
    freq = 220
    t = np.linspace(0, 0.1, 1600, False)
    # Amplitude modulation
    amplitude_envelope = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * 5 * t))  # 5 Hz AM
    signal = amplitude_envelope * np.sin(2 * np.pi * freq * t)
    f0_expected = compute_expected_f0_frames(signal, freq)
    save_test_case(
        "amplitude_modulated_220hz",
        signal,
        f0_expected,
        metadata={
            "description": "220 Hz sine wave with amplitude modulation",
            "frequency": freq,
            "signal_type": "amplitude_modulated"
        }
    )

    print(f"\nGenerated {len(os.listdir('.')) // 3} test cases successfully!")
    print("Each test case consists of three files:")
    print("  - *_signal.json: Input audio signal")
    print("  - *_f0.json: Expected F0 values")
    print("  - *_meta.json: Test case metadata")

if __name__ == "__main__":
    main()
