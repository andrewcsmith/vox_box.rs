extern crate num;
extern crate vox_box;

use num::complex::Complex64;
use vox_box::complex::{SquareRoot, ToComplexVec};
use vox_box::*;
use vox_box::polynomial::Polynomial;
use vox_box::waves::*;
use vox_box::periodic::*;
use vox_box::spectrum::*;

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
    println!("LPC: {:?}", &lpc);
    let roots = lpc.to_complex_vec().find_roots().unwrap();
    println!("Roots: {:?}", &roots);
    let resonances: Vec<Resonance<f64>> = roots.to_resonance(44100.0);
    println!("Resonances: {:?}", resonances);
    assert!((resonances[0].frequency - 689f64).abs() < 1f64);
    assert!((resonances[0].amplitude - 1.0019589).abs() < 1.0e-6);
}
