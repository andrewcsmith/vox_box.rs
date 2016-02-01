extern crate num;

use std::f64::consts::PI;
use std::iter::{Iterator, ExactSizeIterator};
use std::ops::*;
use std::cmp::{PartialOrd};
use std::cmp::Ordering::*;
use num::Float;
use num::traits::{FromPrimitive, ToPrimitive, Zero};
use std::marker::PhantomData;

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

pub trait Resample<T> {
    fn resample_linear(self, usize) -> Self;
}

impl<T: Float + FromPrimitive> Resample<T> for Vec<T> {
    fn resample_linear(self, new_size: usize) -> Self {
        let mut resampled = Vec::<T>::with_capacity(new_size);
        for i in 0..new_size {
            let phase = (i as f64) / ((new_size-1) as f64);
            let index = phase * ((self.len()-1) as f64);
            let a = self[index.floor() as usize];
            let b = self[index.ceil() as usize];
            let t = T::from_f64(index - index.floor()).unwrap();
            resampled.push(a + (b - a) * t);
        }
        resampled
    }
}

pub trait Max<T> {
    fn max(&self) -> T;
}

impl<T: Copy + PartialOrd<T>> Max<T> for [T] {
    fn max(&self) -> T {
        let mut max = self[0];
        for i in 0..self.len() {
            let elem = self[i];
            max = match elem.partial_cmp(&max).unwrap_or(Equal) {
                Less => { max }
                Equal => { max }
                Greater => { elem }
            };
        }
        max
    }
}

pub trait Normalize<T> {
    fn normalize(&mut self);
}

impl<T: Float> Normalize<T> for [T] {
    fn normalize(&mut self) {
        let max = self.max();
        for i in 0..self.len() {
            self[i] = self[i] / max;
        }
    }
}

/// Enum to select the type of window desired
///
/// { Hanning, Hamming }
#[derive(Clone, Copy)]
pub enum WindowType {
    Hanning,
    HanningAutocorrelation,
    Hamming
}

/// Trait that defines something that can be windowed
///
/// window: should mutate self without allocating an additional window
pub trait Windowable<T> {
    fn window(&mut self, WindowType);
}

impl<T: Float + FromPrimitive> Windowable<T> for [T] {
    fn window(&mut self, wtype: WindowType) {
        let len = self.len();
        for (v, w) in self.iter_mut().zip(Window::<T>::new(wtype, len)) {
            *v = *v * w;
        }
    }
}

pub struct Window<T> {
    resource_type: PhantomData<T>,
    window_type: WindowType,
    len: usize,
    idx: usize
}

impl<T> Window<T> {
    pub fn new(window_type: WindowType, len: usize) -> Window<T> {
        Window {
            resource_type: PhantomData::<T>,
            window_type: window_type,
            len: len,
            idx: 0
        }
    }
}

impl<T: Zero + Float + FromPrimitive> Window<T> {
    fn val_at(&self, idx: usize) -> T {
        match self.window_type {
            WindowType::Hanning => {
                let phase: f64 = (idx as f64 / (self.len as f64 - 1.)) * 2.0 * PI;
                T::from_f64(0.5 * (1. - phase.cos())).unwrap_or(T::zero())
            },
            WindowType::HanningAutocorrelation => {
                let phase: f64 = (idx as f64 / (self.len as f64));
                T::from_f64(
                    ((1.0 - phase) *
                    (2.0/3.0 + ((1.0/3.0) * (2.0 * PI * phase).cos()))) +
                    (0.5 / PI) * (2.0 * PI * phase).sin()).unwrap_or(T::zero())
            },
            WindowType::Hamming => {
                let phase: f64 = (idx as f64 / (self.len as f64 - 1.)) * 2.0 * PI;
                T::from_f64(0.54 - (0.46 * phase.cos())).unwrap_or(T::zero())
            }
        }
    }
}

impl<T: Float + FromPrimitive> Iterator for Window<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.idx < self.len {
            let ret = self.val_at(self.idx);
            self.idx += 1;
            Some(ret)
        } else {
            None
        }
    }
}

/// Iterates over a block of data
///
/// Returns a Vec<T> at each iteration, which will be owned by the receiving context
pub struct Windower<'a, T: 'a> {
    window_type: WindowType,
    hop_size: usize,
    bin_size: usize,
    current_index: usize,
    data: &'a [T]
}

impl<'a, T> Windower<'a, T> {
    pub fn new(window_type: WindowType, data: &'a [T], hop_size: usize, bin_size: usize) -> Windower<'a, T> {
        Windower { 
            window_type: window_type, 
            hop_size: hop_size, 
            bin_size: bin_size, 
            current_index: 0,
            data: data
        }
    }

    pub fn len(&self) -> usize {
        ((self.data.len() - self.bin_size) / self.hop_size) + 1
    }

    fn window(&self) -> Window<T> {
        Window { 
            resource_type: PhantomData::<T>, 
            window_type: self.window_type,
            len: self.bin_size,
            idx: 0
        }
    }
}

impl<'a, T: Copy + Float + FromPrimitive> Iterator for Windower<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < (self.len() - 1) {
            let start = self.current_index * self.hop_size;
            let end = start + self.bin_size;
            self.current_index += 1;
            let window = self.window();
            let data_iter = self.data[start..end].iter();

            let out = window.zip(data_iter).map(|val| {
                val.0 * *val.1
            }).collect();

            Some(out)
        } else {
            None
        }
    }
}

/// Filter
///
/// Preemphasis gives a 6db/oct boost above a particular center frequency
/// Factor is center frequency / sample_rate
pub trait Filter<T> {
    fn preemphasis(&mut self, factor: T) -> &mut [T];
}

impl<T> Filter<T> for [T] where T: Mul<T, Output=T> + Sub<T, Output=T> + Copy + ToPrimitive + FromPrimitive {
    fn preemphasis<'a>(&'a mut self, factor: T) -> &'a mut [T] {
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

    #[test]
    fn test_resample_vec32() {
        let upsamp: Vec<f32> = vec![1f32, 2.0, 3.0, 4.0];
        let exp: Vec<f32> = vec![1f32, 4.0];
        let res: Vec<f32> = upsamp.resample_linear(2);
        assert_eq!(res, exp);
    }

    #[test]
    fn test_resample_vec64() {
        let upsamp: Vec<f64> = vec![1f64, 2.0, 3.0, 4.0];
        let exp: Vec<f64> = vec![1f64, 2.5, 4.0];
        let res: Vec<f64> = upsamp.resample_linear(3);
        assert_eq!(res, exp);
    }
}
