extern crate num;
extern crate sample;

use std::f64::consts::PI;
use std::iter::Iterator;
use std::ops::*;
use std::cmp::Ordering::*;

use sample::{Sample, FloatSample, FromSample};

pub trait RMS<S> {
    fn rms(&self) -> S;
}

impl<S: Sample> RMS<S> for [S] {
    fn rms(&self) -> S {
        let sum = self.iter()
            .fold(S::equilibrium(), |acc, &item: &S| {
                acc.add_amp(item.mul_amp(item.to_float_sample()).to_signed_sample())
            });
        (sum.to_float_sample() / (self.len() as f64).to_sample::<S::Float>())
            .sample_sqrt().to_sample::<S>()
    }
}

pub trait Amplitude<S> {
    fn amplitude(self) -> S;
}

impl<S: Sample> Amplitude<S> for S {
    fn amplitude(self) -> S {
        if self < S::equilibrium() {
            self.mul_amp(S::Float::from_sample(-1.0))
        } else {
            self
        }
    }
}

pub trait MaxAmplitude<S> {
    fn max_amplitude(&self) -> S;
}

/// Returns the maximum peak amplitude in a given slice of samples
impl<S: Sample> MaxAmplitude<S> for [S] {
    fn max_amplitude(&self) -> S {
        assert!(self.len() > 0);
        if self.len() == 1 {
            self[0].amplitude()
        } else {
            self[1..].iter().fold(self[0].amplitude(), |acc, elem| {
                let amp = elem.amplitude();
                match amp.partial_cmp(&acc) {
                    Some(Greater) => amp,
                    _ => acc
                }
            })
        }
    }
}

pub trait Normalize<S> {
    fn normalize_with_max(&mut self, max: Option<S>);
    fn normalize(&mut self) {
        self.normalize_with_max(None);
    }
}

impl<S: Sample> Normalize<S> for [S] {
    fn normalize_with_max(&mut self, max: Option<S>) {
        let scale_factor: <S as Sample>::Float = <S as Sample>::identity() / 
            max.unwrap_or(self.max_amplitude()).to_float_sample();
        for elem in self.iter_mut() {
            *elem = elem.mul_amp(scale_factor);
        }
    }
}

/// Filter
///
/// Preemphasis should give a 6db/oct boost above a particular center frequency
/// Factor is center `frequency / sample_rate`
pub trait Filter {
    fn preemphasis(&mut self, factor: f64) -> &mut Self; 
}

impl<S: Sample + FromSample<f64>> Filter for [S] {
    fn preemphasis<'a>(&'a mut self, factor: f64) -> &'a mut [S] {
        let mut last = self[self.len()-1];
        let filter = 2.0 * PI * factor;
        for x in self.iter_mut().rev().skip(1) {
            *x = x.add_amp(last.mul_amp(filter.to_sample::<S::Float>()).to_signed_sample());
            last = *x;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    extern crate sample;

    use super::*;
    use super::super::*;
    use super::super::periodic::*;

    use sample::conv::ToSampleSlice;
    use sample::window::Window;

    fn sine(len: usize) -> Vec<f64> {
        let rate = sample::signal::rate(len as f64).const_hz(1.0);
        rate.clone().sine().take(len).collect::<Vec<[f64; 1]>>().to_sample_slice().to_vec()
    }

    #[test]
    fn test_pe() {
        let mut sine = sine(32);
        sine.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
    }

    #[test]
    fn test_window_autocorr() {
        let lag_window: Window<[f64; 1], HanningLag> = Window::new(16);
        let data: Vec<[f64; 1]> = lag_window.take(16).collect();
        println!("window autocorr: {:?}", &data);
        let window: Window<[f64; 1], Hanning> = Window::new(16);
        let mut manual: Vec<f64> = {
            let mut d: Vec<[f64; 1]> = window.take(16).collect();
            d.to_sample_slice().autocorrelate(16)
        };
        manual.normalize();
        println!("manual autocorr: {:?}", &manual);
        for i in 0..16 {
            let diff = (manual[i] - data.to_sample_slice()[i]).abs();
            assert!(diff < 1e-1);
        }
    }

    #[test]
    fn test_rms() {
        let sine = sine(64);
        let rms = sine.rms();
        println!("rms is {:?}", rms);
        assert!((rms - 0.707).abs() < 0.001);
    }
}
