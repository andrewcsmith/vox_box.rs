extern crate num;
extern crate rand;
extern crate rustfft as fft;

use num::{Float, ToPrimitive, FromPrimitive};
use num::traits::{Zero, Signed};
use complex::{ToComplex, ToComplexVec};
use waves::{Filter};
use num::complex::*;
use std::fmt::Debug;

const FFT_SIZE: usize = 512;

pub trait MFCC<T> {
    fn mfcc(&self, num_coeffs: usize, freq_bounds: (f64, f64), sample_rate: f64) -> Vec<Complex<f64>>;
}

pub fn hz_to_mel(hz: f64) -> f64 {
    1125. * (hz / 700.).ln_1p()
}

pub fn mel_to_hz(mel: f64) -> f64 {
    700. * ((mel / 1125.).exp() - 1.)
}

/// MFCC assumes that it is a windowed signal
impl<T: Debug + Float + ToPrimitive + FromPrimitive + ToComplex<T> + Zero + Signed> MFCC<T> for [T] {
    fn mfcc(&self, num_coeffs: usize, freq_bounds: (f64, f64), sample_rate: f64) -> Vec<Complex<f64>> {
        let mel_range = hz_to_mel(freq_bounds.1) - hz_to_mel(freq_bounds.0);
        // Still an iterator
        let points = (0..(num_coeffs + 2)).map(|i| (i as f64 / num_coeffs as f64) * mel_range + hz_to_mel(freq_bounds.0));
        let bins: Vec<usize> = points.map(|point| ((FFT_SIZE + 1) as f64 * mel_to_hz(point) / sample_rate).floor() as usize).collect();

        let mut spectrum = vec![T::zero().to_complex(); FFT_SIZE];
        let mut fft = fft::FFT::new(FFT_SIZE, false);
        let signal = self.to_vec().to_complex_vec();
        fft.process(&signal, &mut spectrum);

        let energies: Vec<Complex<f64>> = bins.windows(3).map(|window| {
            let up = window[1] - window[0];

            let up_sum = (window[0]..window[1]).enumerate().fold(0f64, |acc, (i, bin)| {
                let multiplier = i as f64 / up as f64;
                acc + spectrum[bin].norm_sqr().to_f64().unwrap().abs() * multiplier
            });

            let down = window[2] - window[1];
            let down_sum = (window[1]..window[2]).enumerate().fold(0f64, |acc, (i, bin)| {
                let multiplier = i as f64 / down as f64;
                acc + spectrum[bin].norm_sqr().to_f64().unwrap().abs() * multiplier
            });
            Complex::<f64> { re: (up_sum + down_sum).log10(), im: 0. }
        }).collect();
        
        let mut mfccs = energies.clone();
        let mut mfcc_fft = fft::FFT::new(num_coeffs, false);
        mfcc_fft.process(&energies, &mut mfccs[..]); mfccs[1..(1 + (num_coeffs as f64 / 2.).floor() as usize)].to_vec() 
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
        panic!()
    }
}
