extern crate num;
use num::complex::{Complex, Complex32, Complex64};

pub mod complex;
pub mod polynomial;

use std::iter::{Iterator};
use std::f64::consts::PI;
// use std::i16;
// use std::fs::File;
use std::ops::*;
// use std::path::Path;
use std::cmp::Ordering::Equal;
use std::cmp::PartialOrd;

use complex::ToComplex;
use polynomial::Polynomial;

pub trait Osc {
    fn sine(size: usize) -> Vec<f64>;
    fn saw(size: usize) -> Vec<f64>;
}

impl Osc for Vec<f64> {
    fn sine(size: usize) -> Vec<f64> {
        let mut sine: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = (i as f64) * 2.0 * PI / ((size) as f64);
            sine.push(phase.sin());
        }
        sine
    }

    fn saw(size: usize) -> Vec<f64> {
        let mut saw: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = (i as f64) / (size) as f64;
            saw.push((phase - 0.5) * -2.0);
        }
        saw
    }
}

pub trait Windowing {
    fn hanning(size: usize) -> Vec<f64>;
    fn hamming(size: usize) -> Vec<f64>;
}

impl Windowing for Vec<f64> {
    fn hanning(size: usize) -> Vec<f64> {
        let mut win: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(0.5 * (1.0 - phase.cos()));
        }
        win
    }
    
    fn hamming(size: usize) -> Vec<f64> {
        let mut win: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(0.54 - (0.46 * phase.cos()));
        }
        win
    }
}

pub trait Filter<T> {
    fn preemphasis(&mut self, factor: f64) -> &mut Vec<T>;
}

/// Factor is center frequency / sample_rate
impl<T> Filter<T> for Vec<T> where T: Mul<f64, Output=T> + Sub<T, Output=T> + Copy {
    fn preemphasis<'a>(&'a mut self, factor: f64) -> &'a mut Vec<T> {
        let filter = (-2.0 * PI * factor).exp();
        for i in (1..self.len()).rev() {
            self[i] = self[i] - (self[i-1] * filter);
        };
        self
    }
}

pub trait Autocorrelates<T> {
    fn autocorrelate(&self, n_coeffs: i32) -> Vec<T>;
    fn normalize(&mut self);
}

impl<T> Autocorrelates<T> for Vec<T> where T: Mul<T, Output=T> + Add<T, Output=T> + Copy + std::cmp::PartialOrd + Div<T, Output=T> {
    fn autocorrelate(&self, n_coeffs: i32) -> Vec<T> {
        let mut coeffs: Vec<T> = Vec::with_capacity(n_coeffs as usize);
        for lag in 0..n_coeffs {
            let mut accum = self[0];
            for i in 1..(self.len() - (lag as usize)) {
                accum = accum + (self[i as usize] * self[((i as i32) + lag) as usize]);
            }
            coeffs.push(accum);
        }
        coeffs
    }

    fn normalize(&mut self) {
        let mut sorted = self.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Equal));
        let max = sorted[0];
        for i in (0..self.len()) {
            self[i] = self[i] / max;
        }
    }
}

pub trait LPC<T> {
    fn lpc(&self, n_coeffs: usize) -> Vec<T>;
}

impl LPC<f64> for Vec<f64> {
    fn lpc(&self, n_coeffs: usize) -> Vec<f64> {
        let mut ac: Vec<f64> = vec![0f64; n_coeffs + 1];
        let mut kc: Vec<f64> = vec![0f64; n_coeffs];
        let mut tmp: Vec<f64> = vec![0f64; n_coeffs];

        /* order 0 */
        let mut err = self[0];
        ac[0] = 1.0;

        /* order >= 1 */
        for i in (1..n_coeffs+1) {
            let mut acc = self[i];
            for j in (1..i) {
                acc = acc + (ac[j] * self[i-j]);
            }
            kc[i-1] = -1.0 * acc/err;
            ac[i] = kc[i-1];
            for j in (0..n_coeffs) {
                tmp[j] = ac[j];
            }
            for j in (1..i) {
                ac[j] = ac[j] + (kc[i-1] * tmp[i-j]);
            }
            err = err * (1f64 - (kc[i-1] * kc[i-1]));
        };
        ac
    }
}

pub trait Resonance<T> {
    fn resonances(self, sample_rate: u32) -> Vec<T>;
}

impl Resonance<f64> for Vec<Complex<f64>> {
    // Give it some roots, it'll find the resonances
    fn resonances(self, sample_rate: u32) -> Vec<f64> {
        let mut res: Vec<f64> = self.iter()
            .filter(|v| v.im >= 0f64)
            .map(|v| v.im.atan2(v.re))
            .map(|v| v * ((sample_rate as f64) / (2f64 * PI)))
            .filter(|v| *v > 1f64)
            .collect();
        res.sort_by(|a, b| (a.partial_cmp(b)).unwrap());
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    use num::complex::Complex64;

    #[test]
    fn test_ac() { 
        let sine = Vec::<f64>::sine(16);
        sine.autocorrelate(16);
    }

    #[test]
    fn test_resonances() {
        let roots = vec![Complex64::new( -0.5, 0.86602540378444 ), Complex64::new( -0.5, -0.86602540378444 )];
        let res = roots.resonances(300);
        println!("Resonances: {:?}", res);
        assert!((res[0] - 100.0).abs() < 1e-8);
    }

    #[test]
    fn test_lpc() {
        let sine = Vec::<f64>::sine(8);
        let mut auto = sine.autocorrelate(8);
        auto.normalize();       
        let auto_exp = vec![1.0, 0.7071, 0.1250, -0.3536, -0.5, -0.3536, -0.1250, 0.0];
        // Rust output:
        let lpc_exp = vec![1.0, -1.3122, 0.8660, -0.0875, -0.0103];
        let lpc = auto.lpc(4);
        // println!("LPC coeffs: {:?}", &lpc);
        for (a, b) in auto.iter().zip(auto_exp.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
        for (a, b) in lpc.iter().zip(lpc_exp.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
    }

    #[test]
    fn test_pe() {
        let mut saw = Vec::<f64>::saw(32);
        let mut sine = Vec::<f64>::sine(32);
        saw.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
        sine.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
    }
}

