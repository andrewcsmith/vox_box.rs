extern crate num;

use std::f64::consts::PI;
use std::iter::Iterator;
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

pub enum Window {
    Hanning,
    Hamming
}

pub trait Windowing<T> {
    fn window(&mut self, Window);
    fn hanning(size: usize) -> Vec<T>;
    fn hanning_autocor(size: usize) -> Vec<T>;
    fn hamming(size: usize) -> Vec<T>;
}

impl<T: Float + FromPrimitive> Windowing<T> for Vec<T> {
    fn window(&mut self, window_type: Window) {
        let window = match window_type {
            Window::Hanning => { Vec::<T>::hanning(self.len()) },
            Window::Hamming => { Vec::<T>::hamming(self.len()) }
        };

        for i in 0..self.len() {
            self[i] = self[i] * window[i];
        }
    }

    fn hanning(size: usize) -> Vec<T> {
        let mut win: Vec<T> = Vec::<T>::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(T::from_f64(0.5 * (1.0 - phase.cos())).unwrap());
        }
        win
    }

    fn hanning_autocor(size: usize) -> Vec<T> {
        let mut win: Vec<T> = Vec::<T>::with_capacity(size);
        for i in 0..size {
            let phase = (i as f64) / (size as f64);
            win.push(T::from_f64(
                ((1.0 - phase) *
                (2.0/3.0 + ((1.0/3.0) * (2.0 * PI * phase).cos()))) +
                (0.5 / PI) * (2.0 * PI * phase).sin()).unwrap()
            )
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

// Struct that iterates over each window of a block of data
pub struct Windower<T> {
    window: Vec<T>,
    data: Vec<T>,
    hop_size: usize,
    bin_size: usize,
    current_index: usize
}

impl<T: Float + FromPrimitive> Windower<T> {
    pub fn new(window_type: Window, data: Vec<T>, hop_size: usize, bin_size: usize) -> Windower<T> {
        let window = match window_type {
            Window::Hanning => { Vec::<T>::hanning(bin_size) },
            Window::Hamming => { Vec::<T>::hamming(bin_size) },
        };
        Windower { window: window, data: data, hop_size: hop_size, bin_size: bin_size, current_index: 0 }
    }

    pub fn len(&self) -> usize {
        ((self.data.len() - self.bin_size) / self.hop_size) + 1
    }
}

impl<T: Float + FromPrimitive> Iterator for Windower<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < (self.len() - 1) {
            let start = self.current_index * self.hop_size;
            let end = start + self.bin_size;
            self.current_index += 1;
            let mut output = Vec::<T>::with_capacity(self.bin_size);
            for (i, v) in self.data[start..end].iter().enumerate() {
                output.push(*v * self.window[i]);
            }
            Some(output)
        } else {
            None
        }
    }
}

pub trait Filter<T> {
    fn preemphasis(&mut self, factor: T) -> &mut Vec<T>;
}

// Factor is center frequency / sample_rate
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
    use super::super::*;

    #[test]
    fn test_pe() {
        let mut saw = Vec::<f64>::saw(32);
        let mut sine = Vec::<f64>::sine(32);
        saw.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
        sine.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
    }

    #[test]
    fn test_windower() {
        let data = Vec::<f64>::sine(64);
        let windower = Windower::new(Window::Hanning, data, 16, 32);
        assert_eq!(windower.len(), 3);
    }

    #[test]
    fn test_window_autocorr() {
        let data = Vec::<f64>::hanning_autocor(16);
        println!("window autocorr: {:?}", data);
        let mut manual = Vec::<f64>::hanning(16).autocorrelate(16);
        manual.normalize();
        println!("manual autocorr: {:?}", manual);
        for i in 0..16 {
            let diff = (manual[i] - data[i]).abs();
            assert!(diff < 1e-1);
        }
    }
}
