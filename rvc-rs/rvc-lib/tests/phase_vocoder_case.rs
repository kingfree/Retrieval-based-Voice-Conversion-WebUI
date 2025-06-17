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
    let data = fs::read_to_string("tests/data/phase_vocoder_case.json").unwrap();
    let case: Case = serde_json::from_str(&data).unwrap();
    let result = phase_vocoder(&case.a, &case.b, &case.fade_out, &case.fade_in);
    assert_eq!(result.len(), case.expected.len());
    for (r, e) in result.iter().zip(case.expected.iter()) {
        assert!((r - e).abs() < 1e-4);
    }
}
