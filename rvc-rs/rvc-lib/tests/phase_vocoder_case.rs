use rvc_lib::phase_vocoder;
use std::fs;

#[derive(serde::Deserialize)]
struct Case {
    a: Vec<f32>,
    b: Vec<f32>,
    fade_out: Vec<f32>,
    fade_in: Vec<f32>,
    expected: Vec<f32>,
}

#[test]
fn test_phase_vocoder_case() {
    use tch::Tensor;

    let data = fs::read_to_string("tests/data/phase_vocoder_case.json").unwrap();
    let case: Case = serde_json::from_str(&data).unwrap();

    let a = Tensor::from_slice(&case.a);
    let b = Tensor::from_slice(&case.b);
    let fade_out = Tensor::from_slice(&case.fade_out);
    let fade_in = Tensor::from_slice(&case.fade_in);

    let result = phase_vocoder(&a, &b, &fade_out, &fade_in);
    let result: Vec<f32> = result.try_into().unwrap();
    assert_eq!(result.len(), case.expected.len());
    for (&r, &e) in result.iter().zip(case.expected.iter()) {
        assert!((r - e).abs() < 1e-4);
    }
}
