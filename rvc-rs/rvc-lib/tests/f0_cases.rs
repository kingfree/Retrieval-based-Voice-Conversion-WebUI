use rvc_lib::{GUIConfig, RVC};
use std::fs;

fn load_json(path: &str) -> Vec<f32> {
    let data = fs::read_to_string(path).expect("read json");
    serde_json::from_str::<Vec<f32>>(&data).unwrap()
}

#[test]
fn test_f0_zero_signal() {
    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);
    let signal = load_json("tests/data/zero_signal.json");
    let expected = load_json("tests/data/zero_f0.json");
    let (_coarse, f0) = rvc.get_f0(&signal, 0.0, "harvest");
    assert_eq!(f0.len(), expected.len());
    for (a,b) in f0.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_f0_sine100() {
    let cfg = GUIConfig::default();
    let rvc = RVC::new(&cfg);
    let signal = load_json("tests/data/sine100_signal.json");
    let expected = load_json("tests/data/sine100_f0.json");
    let (_coarse, f0) = rvc.get_f0(&signal, 0.0, "harvest");
    assert_eq!(f0.len(), expected.len());
    for (a,b) in f0.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}
