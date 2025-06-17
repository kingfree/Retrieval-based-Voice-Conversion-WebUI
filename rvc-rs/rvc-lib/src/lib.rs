/// Core functionality placeholder
pub fn greet(name: &str) -> String {
    format!("Hello, {name} from rvc-lib!")
}

mod harvest;
pub use harvest::Harvest;

mod gui;
pub use gui::{GUIConfig, GUI};

mod devices;
pub use devices::{
    DeviceInfo,
    update_devices,
    set_devices,
    selected_sample_rate,
    get_device_samplerate,
    get_device_channels,
};

mod realtime;
pub use realtime::{start_vc, VC};

mod rvc_for_realtime;
pub use rvc_for_realtime::RVC;

use tch::{Kind, Tensor};

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

    let n = a.len() as i64;

    let a_t = Tensor::of_slice(a);
    let b_t = Tensor::of_slice(b);
    let fade_out_t = Tensor::of_slice(fade_out);
    let fade_in_t = Tensor::of_slice(fade_in);

    let window = (&fade_out_t * &fade_in_t).sqrt();
    let fa = (&a_t * &window).rfft(1, false, true);
    let fb = (&b_t * &window).rfft(1, false, true);

    let mut absab = fa.abs() + fb.abs();
    if n % 2 == 0 {
        let len = absab.size()[0];
        absab.i((1..len - 1)).mul_(2.0);
    } else {
        let len = absab.size()[0];
        absab.i((1..len)).mul_(2.0);
    }

    let phia = fa.angle();
    let phib = fb.angle();
    let mut deltaphase = &phib - &phia;
    deltaphase -= (deltaphase / (2.0 * std::f64::consts::PI) + 0.5).floor() * 2.0 * std::f64::consts::PI;
    let w = Tensor::arange(absab.size()[0], (Kind::Float, tch::Device::Cpu)) * 2.0 * std::f64::consts::PI + &deltaphase;

    let t = Tensor::arange(n, (Kind::Float, tch::Device::Cpu)) / (n as f64);
    let cos_arg = w.unsqueeze(0) * t.unsqueeze(1) + phia.unsqueeze(0);
    let sum = (absab.unsqueeze(0) * cos_arg.cos()).sum_dim_intlist(&[1], false, Kind::Float);

    let result = a_t * fade_out_t.pow(2.0) + b_t * fade_in_t.pow(2.0) + &sum * &window / (n as f64);
    Vec::<f32>::from(result)
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
