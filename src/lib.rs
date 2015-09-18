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

pub struct FormantFrame<T: Float> {
    frequency: T,
}

pub struct FormantExtractor<'a, T: 'a + Float> {
    num_formants: usize,
    frame_index: usize,
    resonances: &'a Vec<Vec<T>>,
    estimates: Vec<T>
}

impl<'a, T: 'a + Float + PartialEq> FormantExtractor<'a, T> {
    pub fn new(
        num_formants: usize, 
        resonances: &'a Vec<Vec<T>>, 
        starting_estimates: Vec<T>) -> Self {
        FormantExtractor { 
            num_formants: num_formants, 
            resonances: resonances, 
            frame_index: 0,
            estimates: starting_estimates 
        }
    }
}

impl<'a, T: 'a + Float + PartialEq + FromPrimitive> Iterator for FormantExtractor<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = self.resonances[self.frame_index].clone();
        let mut slots: Vec<Option<T>> = self.estimates.iter()
        .enumerate()
        .map(|(index, estimate)| {
            let mut indices: Vec<usize> = (0..frame.len()).collect();
            indices.sort_by(|a, b| {
                (frame[*a] - *estimate).abs().partial_cmp(&(frame[*b] - *estimate).abs()).unwrap()
            });
            let win = indices.first().unwrap().clone();
            Some(frame[win]) // (resonance_index, resonance_freq)
        }).collect();

        // Step 3: Remove duplicates. If the same peak p_j fills more than one slots S_i keep it
        // only in the slot S_k which corresponds to the estimate EST_k that it is closest to in
        // frequency, and remove it from any other slots.
        let mut w: usize = 0;
        let mut has_unassigned: bool = false;
        for r in (1..slots.len()) {
            match slots[r] {
                Some(v) => { 
                    if v == slots[w].unwrap() {
                        if (v - self.estimates[r]).abs() < (v - self.estimates[w]).abs() {
                            slots[w] = None;
                            has_unassigned = true;
                            w = r;
                        } else {
                            slots[r] = None;
                            has_unassigned = true;
                        }
                    } else {
                        w = r;
                    }
                },
                None => { }
            }
        }

        if has_unassigned {
            // Step 4: Deal with unassigned peaks. If there are no unassigned peaks p_j, go to Step 5.
            // Otherwise, try to fill empty slots with peaks not assigned in Step 2 as follows.
            for j in 0..frame.len() {
                let peak = Some(frame[0]);
                if slots.contains(&peak) { continue; }
                match slots[j] {
                    Some(_) => { },
                    None => { slots[j] = peak; continue; }
                }
                match slots[j-1] {
                    Some(_) => { },
                    None => { slots.swap(j, j-1); slots[j] = peak; continue; }
                }
                match slots[j+1] {
                    Some(_) => { },
                    None => { slots.swap(j, j+1); slots[j] = peak; continue; }
                }
            }
        }

        let winners: Vec<T> = slots.iter().map(|v| v.unwrap_or(T::from_f32(0.0).unwrap())).collect();
        self.estimates = winners.clone();
        self.frame_index += 1;
        Some(winners)
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

    #[test]
    fn test_formant_extractor() {
        let resonances = vec![
            vec![100.0, 150.0, 200.0, 240.0, 300.0], 
            vec![110.0, 180.0, 210.0, 230.0, 310.0],
            vec![200.0, 270.0, 290.0, 350.0, 360.0]
        ];
        let mut extractor = FormantExtractor::new( 3, &resonances, vec![140.0, 230.0, 320.0]);
        // First cycle has initial guesses
        match extractor.next() {
            Some(r) => { assert_eq!(r, vec![150.0, 240.0, 300.0]) },
            None => { panic!() }
        };

        // Second cycle should be different
        match extractor.next() {
            Some(r) => { assert_eq!(r, vec![180.0, 230.0, 310.0]) },
            None => { panic!() }
        };

        // Third cycle should have removed duplicates
        match extractor.next() {
            Some(r) => { assert_eq!(r, vec![200.0, 0.0, 290.0]) },
            None => { panic!() }
        };
    }
}

