extern crate num;
extern crate sample;
use num::complex::Complex;
use num::{Float, FromPrimitive};
use sample::Sample;

pub mod complex;
pub mod polynomial;
pub mod waves;
pub mod resample;

use std::iter::Iterator;
use std::f64::consts::PI;
use std::ops::*;
use std::cmp::Ordering::Equal;
use std::cmp::PartialOrd;

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

impl<T> LPC<T> for Vec<T> where T: Float { 
    fn lpc(&self, n_coeffs: usize) -> Vec<T> {
        let mut ac: Vec<T> = vec![T::zero(); n_coeffs + 1];
        let mut kc: Vec<T> = vec![T::zero(); n_coeffs];
        let mut tmp: Vec<T> = vec![T::zero(); n_coeffs];

        /* order 0 */
        let mut err = self[0];
        ac[0] = T::one();

        /* order >= 1 */
        for i in (1..n_coeffs+1) {
            let mut acc = self[i];
            for j in (1..i) {
                acc = acc + (ac[j] * self[i-j]);
            }
            kc[i-1] = acc.neg() / err;
            ac[i] = kc[i-1];
            for j in (0..n_coeffs) {
                tmp[j] = ac[j];
            }
            for j in (1..i) {
                ac[j] = ac[j] + (kc[i-1] * tmp[i-j]);
            }
            err = err * (T::one() - (kc[i-1] * kc[i-1]));
        };
        ac
    }
}

pub trait Resonance<T> {
    fn resonances(self, sample_rate: u32) -> Vec<T>;
}

impl<T> Resonance<T> for Vec<Complex<T>> where T: Float + FromPrimitive {
    // Give it some roots, it'll find the resonances
    fn resonances(self, sample_rate: u32) -> Vec<T> {
        let freq_mul: T = T::from_f64((sample_rate as f64) / (PI * 2f64)).unwrap();
        let mut res: Vec<T> = self.iter()
            .filter(|v| v.im >= T::zero())
            .map(|v| v.im.atan2(v.re) * freq_mul)
            .filter(|v| *v > T::one())
            .collect();
        res.sort_by(|a, b| (a.partial_cmp(b)).unwrap());
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    use num::complex::Complex64;
    use super::waves::*;

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
    fn test_ac() { 
        let sine = Vec::<f64>::sine(16);
        sine.autocorrelate(16);
    }
}

