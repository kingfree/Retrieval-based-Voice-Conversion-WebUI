use rsworld_sys::{Harvest as WorldHarvest, HarvestOption, GetSamplesForHarvest};
use tokio::task;

/// Pitch extraction using the WORLD Harvest algorithm.
///
/// The `Harvest` struct provides a minimal wrapper around the C
/// implementation of the algorithm bundled via `rsworld-sys`.
pub struct Harvest {
    option: HarvestOption,
    fs: i32,
}

impl Harvest {
    /// Create a new `Harvest` extractor for the given sample rate.
    pub fn new(fs: i32) -> Self {
        let mut option = HarvestOption::new();
        option.f0_floor = 50.0;
        option.f0_ceil = 1100.0;
        option.frame_period = 10.0;
        Self { option, fs }
    }

    /// Compute F0 values for the provided audio buffer synchronously.
    pub fn compute(&self, x: &[f32]) -> Vec<f64> {
        let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let option = HarvestOption {
            f0_floor: self.option.f0_floor,
            f0_ceil: self.option.f0_ceil,
            frame_period: self.option.frame_period,
        };
        compute_inner(self.fs, option, &x)
    }

    /// Compute F0 values asynchronously on a dedicated thread.
    pub async fn compute_async(&self, x: Vec<f32>) -> Vec<f64> {
        let fs = self.fs;
        let option = HarvestOption {
            f0_floor: self.option.f0_floor,
            f0_ceil: self.option.f0_ceil,
            frame_period: self.option.frame_period,
        };
        task::spawn_blocking(move || {
            let x_d: Vec<f64> = x.into_iter().map(|v| v as f64).collect();
            compute_inner(fs, option, &x_d)
        })
        .await
        .unwrap()
    }
}

fn compute_inner(fs: i32, option: HarvestOption, x: &[f64]) -> Vec<f64> {
    let x_length = x.len() as i32;
    let f0_len = unsafe { GetSamplesForHarvest(fs, x_length, option.frame_period) } as usize;
    let mut temporal_positions = vec![0.0f64; f0_len];
    let mut f0 = vec![0.0f64; f0_len];
    unsafe {
        WorldHarvest(
            x.as_ptr(),
            x_length,
            fs,
            &option as *const _,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }
    f0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harvest_zero_signal() {
        let extractor = Harvest::new(16000);
        let x = vec![0.0f32; 160];
        let f0 = extractor.compute(&x);
        assert!(f0.iter().all(|&v| v == 0.0));
    }

    #[tokio::test]
    async fn test_harvest_async_zero_signal() {
        let extractor = Harvest::new(16000);
        let x = vec![0.0f32; 160];
        let f0 = extractor.compute_async(x).await;
        assert!(f0.iter().all(|&v| v == 0.0));
    }
}
