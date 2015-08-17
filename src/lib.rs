extern crate nalgebra as na;

use na::{DVec, Vec3};
use std::iter::{Iterator, IntoIterator};
use std::f64::consts::PI;
use std::i16;
use std::fs::File;
use std::path::Path;

pub trait Osc {
    fn sine(size: usize) -> DVec<f64>;
    fn saw(size: usize) -> DVec<f64>;
}

impl Osc for DVec<f64> {
    fn sine(size: usize) -> DVec<f64> {
        DVec::<f64>::from_fn(size, |i| {
            let phase: f64 = (i as f64) * 2.0 * PI / ((size - 1) as f64);
            phase.sin()
        })
    }

    fn saw(size: usize) -> DVec<f64> {
        DVec::<f64>::from_fn(size, |i| {
            let phase: f64 = (i as f64) / (size - 1) as f64;
            (phase - 0.5) * -2.0
        })
    }
}

pub trait Windowing {
    fn hanning(size: usize) -> DVec<f64>;
    fn hamming(size: usize) -> DVec<f64>;
}

impl Windowing for DVec<f64> {
    fn hanning(size: usize) -> DVec<f64> {
        DVec::<f64>::from_fn(size, |i| {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            0.5 * (1.0 - phase.cos())
        })
    }
    
    fn hamming(size: usize) -> DVec<f64> {
        DVec::<f64>::from_fn(size, |i| {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            0.54 - (0.46 * phase.cos())
        })
    }
}

pub trait Filter {
    fn preemphasis(&mut self, freq: f64, sample_rate: f64) -> &mut DVec<f64>;
}

impl Filter for DVec<f64> {
    fn preemphasis<'a>(&'a mut self, freq: f64, sample_rate: f64) -> &'a mut DVec<f64> {
        let filter = (-2.0 * PI * freq / sample_rate).exp();
        for i in (1..self.len()).rev() {
            self[i] -= self[i-1] * filter;
        };
        self
    }
}

pub trait Autocorrelate {
    fn autocorrelate(&self, n_coeffs: i32, max_freq: f64, sample_rate: f64) -> Vec<f64>;
}

impl Autocorrelate for Vec<f64> {
    fn autocorrelate(&self, n_coeffs: i32, max_freq: f64, sample_rate: f64) -> Vec<f64> {
        let mut coeffs: Vec<f64> = Vec::with_capacity(n_coeffs as usize);
        for lag in 0..n_coeffs {
            let len: i32 = self.len() as i32;
            let mut correlations: Vec<f64> = Vec::with_capacity((len - (lag as i32)) as usize);
            let mut accum = 0f64;
            for i in 0..(correlations.len()) {
                accum += self[i as usize] * self[((i as i32) + lag) as usize];
            }
            coeffs.push(accum);
        }
        coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    use na::{DVec};

    #[test]
    fn test_ac() { 
        let sine = DVec::<f64>::sine(16);
        let auto = sine.autocorrelate(4, 1200, 44100);
        println!("sine: {:?}\nauto: {:?}", sine, auto);
        // println!("sine: {:?}\nauto: ", sine);
    }
}

