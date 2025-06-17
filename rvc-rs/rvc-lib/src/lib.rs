/// Core functionality placeholder
pub fn greet(name: &str) -> String {
    format!("Hello, {name} from rvc-lib!")
}

mod harvest;
pub use harvest::Harvest;

mod gui;
pub use gui::{GUI, GUIConfig};

mod devices;
pub use devices::{
    DeviceInfo, get_device_channels, get_device_samplerate, selected_sample_rate, set_devices,
    update_devices,
};

mod realtime;
pub use realtime::{VC, start_vc};

mod rvc_for_realtime;
pub use rvc_for_realtime::RVC;

use std::f64::consts::PI;
use tch::{IndexOp, Kind, Tensor};

pub fn phase_vocoder(a: &Tensor, b: &Tensor, fade_out: &Tensor, fade_in: &Tensor) -> Tensor {
    let window = (fade_out * fade_in).sqrt();
    let fa = (a * &window).fft_rfft(None, -1, "backward");
    let fb = (b * &window).fft_rfft(None, -1, "backward");

    let absab = fa.abs() + fb.abs();
    let absab = absab.shallow_clone();

    let n = a.size()[0] as i64;
    let n_half = n / 2 + 1;

    if n % 2 == 0 {
        // even length: [1, ..., n/2 - 1]
        let idx = 1..(n_half - 1);
        let _ = absab.i(idx).f_mul_(&Tensor::from(2.0));
    } else {
        // odd length: [1, ..., n/2]
        let idx = 1..n_half;
        let _ = absab.i(idx).f_mul_(&Tensor::from(2.0));
    }

    let phia = fa.angle();
    let phib = fb.angle();

    let mut deltaphase = &phib - &phia;
    deltaphase = &deltaphase - 2.0 * PI * (&deltaphase / (2.0 * PI) + 0.5).floor();

    let w = Tensor::arange(n_half, (Kind::Float, a.device())) * 2.0 * PI + &deltaphase;

    let t = Tensor::arange(n, (Kind::Float, a.device())).unsqueeze(-1) / (n as f64);

    let cos_term =
        (&absab * (w * &t + &phia).cos()).sum_dim_intlist(&[-1i64][..], false, Kind::Float);
    let result = a * fade_out.pow_tensor_scalar(2.0)
        + b * fade_in.pow_tensor_scalar(2.0)
        + &cos_term * &window / (n as f64);

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
        let a_data = [
            0.84442186, 0.7579544, 0.42057157, 0.25891674, 0.5112747, 0.40493414, 0.7837986,
            0.30331272,
        ];
        let b_data = [
            0.47659695, 0.58338207, 0.9081129, 0.50468683, 0.28183785, 0.7558042, 0.618369,
            0.25050634,
        ];
        let fade_out_data = [
            0.9097463, 0.98278546, 0.81021726, 0.90216595, 0.31014758, 0.72983176, 0.8988383,
            0.6839839,
        ];
        let fade_in_data = [
            0.4721427, 0.10070121, 0.43417183, 0.61088693, 0.9130111, 0.9666064, 0.47700977,
            0.86530995,
        ];
        let expected = [
            1.55919146, 0.92036238, 0.7466557, 0.78565353, 0.5473819, 1.8781089, 1.3730892,
            0.5757979,
        ];

        let a = Tensor::from_slice(&a_data);
        let b = Tensor::from_slice(&b_data);
        let fade_out = Tensor::from_slice(&fade_out_data);
        let fade_in = Tensor::from_slice(&fade_in_data);

        let result = phase_vocoder(&a, &b, &fade_out, &fade_in);
        let result_vec: Vec<f64> = result.try_into().unwrap();
        for (r, e) in result_vec.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }
}
