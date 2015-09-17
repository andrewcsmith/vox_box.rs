extern crate num;

use std::f64::consts::PI;
use std::ops::*;
use num::Float;
use num::traits::{FromPrimitive, ToPrimitive};

pub trait Osc<T> {
    fn sine(size: usize) -> Vec<T>;
    fn saw(size: usize) -> Vec<T>;
}

impl<T: Float + FromPrimitive> Osc<T> for Vec<T> {
    fn sine(size: usize) -> Vec<T> {
        let mut sine: Vec<T> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: T = T::from_usize(i).unwrap() * T::from_f32(2.0).unwrap() * T::from_f64(PI).unwrap() / T::from_f32(size as f32).unwrap();
            sine.push(phase.sin());
        }
        sine
    }

    fn saw(size: usize) -> Vec<T> {
        let mut saw: Vec<T> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: T = T::from_f64((i as f64) / (size as f64)).unwrap();
            saw.push((phase - T::from_f64(0.5f64).unwrap()) * T::from_f64(-2.0f64).unwrap());
        }
        saw
    }
}

pub trait Windowing<T> {
    fn hanning(size: usize) -> Vec<T>;
    fn hamming(size: usize) -> Vec<T>;
}

impl<T: Float + FromPrimitive> Windowing<T> for Vec<T> {
    fn hanning(size: usize) -> Vec<T> {
        let mut win: Vec<T> = Vec::<T>::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(T::from_f64(0.5 * (1.0 - phase.cos())).unwrap());
        }
        win
    }
    
    fn hamming(size: usize) -> Vec<T> {
        let mut win: Vec<T> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(T::from_f64(0.54 - (0.46 * phase.cos())).unwrap());
        }
        win
    }
}

pub trait Filter<T> {
    fn preemphasis(&mut self, factor: T) -> &mut Vec<T>;
}

/// Factor is center frequency / sample_rate
impl<T> Filter<T> for Vec<T> where T: Mul<T, Output=T> + Sub<T, Output=T> + Copy + ToPrimitive + FromPrimitive {
    fn preemphasis<'a>(&'a mut self, factor: T) -> &'a mut Vec<T> {
        let filter = T::from_f64((-2.0 * PI * factor.to_f64().unwrap()).exp()).unwrap();
        for i in (1..self.len()).rev() {
            self[i] = self[i] - (self[i-1] * filter);
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pe() {
        let mut saw = Vec::<f64>::saw(32);
        let mut sine = Vec::<f64>::sine(32);
        saw.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
        sine.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
    }
}
