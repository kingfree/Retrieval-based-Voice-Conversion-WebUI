/// Core functionality placeholder
pub fn greet(name: &str) -> String {
    format!("Hello, {name} from rvc-lib!")
}

/// Blend two overlapping audio buffers using a phase vocoder crossfade.
///
/// `a` and `b` are the buffers to be blended, typically the previous and
/// current segments. `fade_out` and `fade_in` are the respective fade
/// envelopes. All slices must have the same length.
pub fn phase_vocoder(
    a: &[f32],
    b: &[f32],
    fade_out: &[f32],
    fade_in: &[f32],
) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), fade_out.len());
    assert_eq!(a.len(), fade_in.len());

    let n = a.len();

    // Pre-compute the sqrt window used for analysis and synthesis.
    let window: Vec<f32> = (0..n)
        .map(|i| (fade_out[i] * fade_in[i]).sqrt())
        .collect();

    // Setup FFT planner.
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let rfft = planner.plan_fft_forward(n);

    // Forward FFT for the two windows.
    let mut buf_a: Vec<f32> = a
        .iter()
        .zip(window.iter())
        .map(|(&x, &w)| x * w)
        .collect();
    let mut buf_b: Vec<f32> = b
        .iter()
        .zip(window.iter())
        .map(|(&x, &w)| x * w)
        .collect();
    let mut fa = rfft.make_output_vec();
    let mut fb = rfft.make_output_vec();
    rfft.process(&mut buf_a, &mut fa).unwrap();
    rfft.process(&mut buf_b, &mut fb).unwrap();

    let bins = fa.len(); // n/2 + 1
    let mut absab = Vec::with_capacity(bins);
    for i in 0..bins {
        absab.push(fa[i].norm() + fb[i].norm());
    }
    if n % 2 == 0 {
        for v in &mut absab[1..bins - 1] {
            *v *= 2.0;
        }
    } else {
        for v in &mut absab[1..] {
            *v *= 2.0;
        }
    }

    // Phase calculations.
    let phia: Vec<f32> = fa.iter().map(|c| c.arg()).collect();
    let phib: Vec<f32> = fb.iter().map(|c| c.arg()).collect();

    let mut deltaphase = Vec::with_capacity(bins);
    for (&phi_a, &phi_b) in phia.iter().zip(phib.iter()) {
        let mut dp = phi_b - phi_a;
        dp -= 2.0 * std::f32::consts::PI * ((dp / (2.0 * std::f32::consts::PI) + 0.5).floor());
        deltaphase.push(dp);
    }

    let mut w = Vec::with_capacity(bins);
    for (i, dp) in deltaphase.iter().enumerate() {
        w.push(2.0 * std::f32::consts::PI * i as f32 + *dp);
    }

    // Synthesis
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let mut sum = 0.0f32;
        for k in 0..bins {
            sum += absab[k] * (w[k] * t + phia[k]).cos();
        }
        result.push(
            a[i] * fade_out[i].powi(2)
                + b[i] * fade_in[i].powi(2)
                + sum * window[i] / n as f32,
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet("World"), "Hello, World from rvc-lib!");
    }

    #[test]
    fn test_phase_vocoder() {
        let a = [
            0.84442186, 0.7579544, 0.42057157, 0.25891674,
            0.5112747, 0.40493414, 0.7837986, 0.30331272,
        ];
        let b = [
            0.47659695, 0.58338207, 0.9081129, 0.50468683,
            0.28183785, 0.7558042, 0.618369, 0.25050634,
        ];
        let fade_out = [
            0.9097463, 0.98278546, 0.81021726, 0.90216595,
            0.31014758, 0.72983176, 0.8988383, 0.6839839,
        ];
        let fade_in = [
            0.4721427, 0.10070121, 0.43417183, 0.61088693,
            0.9130111, 0.9666064, 0.47700977, 0.86530995,
        ];
        let expected = [
            1.55919146, 0.92036238, 0.7466557, 0.78565353,
            0.5473819, 1.8781089, 1.3730892, 0.5757979,
        ];

        let result = phase_vocoder(&a, &b, &fade_out, &fade_in);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }
}
