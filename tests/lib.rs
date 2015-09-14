extern crate num;
extern crate vox_box;

use num::complex::Complex64;
use vox_box::complex::{SquareRoot, ToComplexVec};
use vox_box::*;
use vox_box::polynomial::Polynomial;
use vox_box::waves::Osc;

#[test]
fn complex_sqrt() {
    let c = Complex64::new(2.0, -3.0);
    let root = Complex64::new(1.6741492280355, -0.89597747612984);
    let delta = 0.000001;
    let result: Complex64 = c.sqrt() - &root;
    println!("Complex is {:?}", result);
    assert!(result.re.abs() < delta);
    assert!(result.im.abs() < delta);
}

#[test]
fn sine_resonances() {
    let wave: Vec<f64> = Vec::<f64>::sine(64).iter()
        .cycle()
        .take(512)
        .map(|v| (v * 0.75))
        .collect();
    let mut auto = wave.autocorrelate(32);
    auto.normalize();
    let lpc = auto.lpc(2);
    let roots = lpc.to_complex_vec().find_roots().unwrap();
    let resonances: Vec<f64> = roots.resonances(44100);
    println!("Resonances: {:?}", resonances);
    assert!((resonances[0] - 689f64).abs() < 1f64);
}
