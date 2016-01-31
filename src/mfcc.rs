extern crate num;
extern crate rand;
extern crate rustfft as fft;

use num::{Float, ToPrimitive, FromPrimitive};
use std::f64::consts::PI;
use num::traits::{Zero, Signed};
use complex::{ToComplex, ToComplexVec};
use waves::{Filter};
use num::complex::*;
use std::fmt::Debug;

const FFT_SIZE: usize = 512;

pub trait MFCC<T> {
    fn mfcc(&self, num_coeffs: usize, freq_bounds: (f64, f64), sample_rate: f64) -> Vec<T>;
}

pub fn hz_to_mel(hz: f64) -> f64 {
    1125. * (hz / 700.).ln_1p()
}

pub fn mel_to_hz(mel: f64) -> f64 {
    700. * ((mel / 1125.).exp() - 1.)
}

pub fn dct<T: FromPrimitive + ToPrimitive + Float>(signal: &[T]) -> Vec<T> {
    signal.iter().enumerate().map(|(k, val)| {
        T::from_f64(2. * (0..signal.len()).fold(0., |acc, n| {
            acc + signal[n].to_f64().unwrap() * (PI * k as f64 * (2. * n as f64 + 1.) / (2. * signal.len() as f64)).cos()
        })).unwrap()
    }).collect()
}

/// MFCC assumes that it is a windowed signal
impl<T: Debug + Float + ToPrimitive + FromPrimitive + ToComplex<T> + Zero + Signed> MFCC<T> for [T] {
    fn mfcc(&self, num_coeffs: usize, freq_bounds: (f64, f64), sample_rate: f64) -> Vec<T> {
        let mel_range = hz_to_mel(freq_bounds.1) - hz_to_mel(freq_bounds.0);
        // Still an iterator
        let points = (0..(num_coeffs + 2)).map(|i| (i as f64 / num_coeffs as f64) * mel_range + hz_to_mel(freq_bounds.0));
        let bins: Vec<usize> = points.map(|point| ((FFT_SIZE + 1) as f64 * mel_to_hz(point) / sample_rate).floor() as usize).collect();

        let mut spectrum = vec![T::zero().to_complex(); FFT_SIZE];
        let mut fft = fft::FFT::new(FFT_SIZE, false);
        let signal = self.to_vec().to_complex_vec();
        fft.process(&signal, &mut spectrum);

        let energies: Vec<T> = bins.windows(3).map(|window| {
            let up = window[1] - window[0];

            let up_sum = (window[0]..window[1]).enumerate().fold(0f64, |acc, (i, bin)| {
                let multiplier = i as f64 / up as f64;
                acc + spectrum[bin].norm_sqr().to_f64().unwrap().abs() * multiplier
            });

            let down = window[2] - window[1];
            let down_sum = (window[1]..window[2]).enumerate().fold(0f64, |acc, (i, bin)| {
                let multiplier = i as f64 / down as f64;
                acc + spectrum[bin].norm().to_f64().unwrap().abs() * multiplier
            });
            T::from_f64((up_sum + down_sum).log10()).unwrap_or(T::from_f32(1.0e-10).unwrap())
        }).collect();

        dct(&energies[..])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{thread_rng, Rng};
    use waves::{Filter, Windowing, Window};

    #[test]
    fn test_hz_to_mel() {
        assert!(hz_to_mel(300.) - 401.25 < 1.0e-2);
    }

    #[test]
    fn test_mel_to_hz() {
        assert!(mel_to_hz(401.25) - 300. < 1.0e-2);
    }

    #[test]
    fn test_mfcc() {
        let mut rng = thread_rng();
        let mut vec: Vec<f64> = (0..super::FFT_SIZE).map(|_| rng.gen_range::<f64>(-1., 1.)).collect();
        vec.preemphasis(0.1f64 * 22_050.).window(Window::Hanning);
        let mfccs = vec.mfcc(26, (133., 6855.), 22_050.);
        println!("mfccs: {:?}", mfccs);
    }

    #[test]
    fn test_dct() {
        let signal = [0.2, 0.3, 0.4, 0.3];
        let dcts = dct(&signal[..]);
        let exp = [2.4, -0.26131, -0.28284, 0.10823];
        println!("dcts: {:?}", &dcts);
        for pair in dcts.iter().zip(exp.iter()) {
            assert!(pair.0 - pair.1 < 1.0e-5);
        }
    }
}
