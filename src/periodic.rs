extern crate num;

use num::{Float, ToPrimitive, FromPrimitive};
use super::waves::Normalize;

use sample;
use sample::window::Window;
use sample::{Sample, ToSampleSlice, FromSample};

use std::f64::consts::PI;
use std;

/// Lifted from Praat code
fn interpolate_sinc<S: Sample>(y: &[S], offset: isize, nx: usize, x: S, max_depth: usize) -> f64 {
    let x = x.to_float_sample().to_sample::<f64>();
    let nl = x.floor() as usize;
    let nr = nl + 1;
    let phil = x - nl as f64;
    let phir = 1. - phil;
    let mut result = 0.;

    // simple cases
    if nx < 1 { return std::f64::NAN }
    if x > nx as f64 { return y[nx as usize].to_float_sample().to_sample::<f64>() }
    if x < 1. { return y[1].to_float_sample().to_sample::<f64>() }
    if x == nl as f64 { return y[nl].to_float_sample().to_sample::<f64>() };

    for n in 0..max_depth {
        result += {
            let a = PI * (phil + n as f64);
            let r_lag = y[(offset as i32 + nr as i32 - n as i32) as usize].to_float_sample().to_sample::<f64>();
            let first = a.sin() / a;
            let second = 0.5 + 0.5 * (a / (phil + max_depth as f64)).cos();
            r_lag * first * second
        };
        result += {
            let a = PI * (phir + n as f64);
            let r_lag = y[(offset as i32 + nl as i32 + n as i32) as usize].to_float_sample().to_sample::<f64>();
            let first = a.sin() / a;
            let second = 0.5 + 0.5 * (a / (phir + max_depth as f64)).cos();
            r_lag * first * second
        };
    }

    result
}

pub trait LagType: sample::window::Type {
    type Lag: sample::window::Type;
}

pub struct HanningLag;

impl sample::window::Type for HanningLag {
    fn at_phase<S: Sample>(phase: S) -> S {
        let pi_2 = (PI * 2.).to_sample();
        let v: f64 = (phase.to_float_sample() * pi_2).to_sample::<f64>();
        let one_third: S::Float = (1.0 / 3.0).to_sample();
        let two_thirds: S::Float = (2.0 / 3.0).to_sample();
        let one: S::Float = 1.0.to_sample();
        ((one - phase.to_float_sample()) * (two_thirds + (one_third * (sample::ops::f64::cos(v)).to_sample())) 
            + (one / pi_2) * (sample::ops::f64::sin(v)).to_sample()).to_sample::<S>()
    }
}

impl LagType for sample::window::Hanning {
    type Lag = HanningLag;
}

/// Trait for things that can Autocorrelate. Implement the mutable version,
/// which takes a slice of coefficients, and receive a version that allocates
/// its own vector for free.
///
/// ```
/// extern crate vox_box;
/// use vox_box::periodic::Autocorrelate;
/// 
/// let some_values = [1.0, 0.5, 0.0, -0.5, -1.0];
/// assert_eq!(some_values.autocorrelate(2), vec![-1.0, -1.0]);
/// ```
pub trait Autocorrelate<T> 
    where T: Sample
{
    fn autocorrelate_mut(&self, n_coeffs: usize, coeffs: &mut [T]);
    fn autocorrelate(&self, n_coeffs: usize) -> Vec<T> {
        let mut coeffs: Vec<T> = vec![T::equilibrium(); n_coeffs];
        self.autocorrelate_mut(n_coeffs, &mut coeffs);
        coeffs
    }
}

impl<T> Autocorrelate<T> for [T] 
    where T: Sample
{
    fn autocorrelate_mut(&self, n_coeffs: usize, coeffs: &mut [T]) {
        assert!(n_coeffs <= coeffs.len());
        for lag in 0..n_coeffs {
            let mut accum: T = self[0];
            for i in 1..(self.len() - lag) {
                accum = accum.add_amp(self[i].mul_amp(self[(i + lag) as usize].to_float_sample()).to_signed_sample());
            }
            coeffs[lag] = accum;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pitch<T: Float> {
    pub frequency: T,
    pub strength: T
}

impl<T> Pitch<T> 
    where T: Float
{
    pub fn new(frequency: T, strength: T) -> Self {
        Pitch { frequency: frequency, strength: strength }
    }
}

pub struct PitchExtractor<'a, T: 'a + Float> {
    voiced_unvoiced_cost: T,
    voicing_threshold: T,
    candidates: &'a [&'a [Pitch<T>]]
}

impl<'a, T: 'a + Float> PitchExtractor<'a, T> {
    pub fn new(candidates: &'a [&'a [Pitch<T>]], voiced_unvoiced_cost: T, voicing_threshold: T) -> Self {
        PitchExtractor {
            voiced_unvoiced_cost: voiced_unvoiced_cost,
            voicing_threshold: voicing_threshold,
            candidates: candidates
        }
    }
}

impl<'a, T: 'a + Float> Iterator for PitchExtractor<'a, T> {
    type Item = Pitch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_candidates = self.candidates.len();
        if n_candidates == 0 { 
            return None 
        }

        let candidate = self.candidates[0][0];
        self.candidates = if n_candidates > 1 { 
            &self.candidates[1..] 
        } else { 
            &[] 
        };
        Some(candidate)
    }
}

pub trait Pitched<S, T: Float> {
    fn pitch<W: LagType>(&self, sample_rate: T, threshold: T, silence_threshold: S, local_peak: S, global_peak: S, octave_cost: T, min: T, max: T) -> Vec<Pitch<T>>;
}

trait LocalMaxima<S: Sample> {
    fn local_maxima(&self) -> Vec<(usize, S)>;
}

impl<S> LocalMaxima<S> for [S]
    where S: Sample 
{
    /// Find the local maxima for a vector. Skips the one at index 0.
    fn local_maxima(&self) -> Vec<(usize, S)> {
        self.windows(3).enumerate().filter(|x| {
            x.1[0] < x.1[1] && x.1[2] < x.1[1]
        }).map(|x| ((x.0 + 1), x.1[1])).collect()
    }
}

impl<S, T> Pitched<S, T> for [S]
    where S: Sample + FromSample<f64> + std::fmt::Debug, 
          S::Float: ToPrimitive,
          T: Float + FromPrimitive
{
    /// Assumes that it's being passed a windowed signal
    fn pitch<W: LagType>(&self, sample_rate: T, threshold: T, silence_threshold: S, local_peak: S, global_peak: S, octave_cost: T, min: T, max: T) -> Vec<Pitch<T>> {
        let window_lag: Vec<S> = Window::<[S; 1], W::Lag>::new(self.len()).take(self.len()).map(|x| x.to_sample_slice()[0]).collect();
        let mut self_lag = self.autocorrelate(self.len());
        self_lag.normalize();

        for (s, w) in self_lag.iter_mut().zip(window_lag.iter()) {
            *s = (s.to_float_sample() / w.to_float_sample()).to_sample::<S>();
        }

        self_lag.resize(self.len() * 2, S::from_sample(0.));

        let voiceless = (T::zero(), threshold + T::zero().max(T::one() + T::one() - (T::from(local_peak.to_float_sample()).unwrap() / T::from(global_peak.to_float_sample()).unwrap()) / (T::from(silence_threshold.to_float_sample()).unwrap() / T::one() + threshold)));

        let interpolation_depth = 0.5;
        let brent_ixmax = (interpolation_depth * self.len() as f64).floor() as usize;

        let mut maxima: Vec<Pitch<T>> = self_lag[0..brent_ixmax as usize].local_maxima().iter()
            .map(|x| {
                // Calculate the frequency using parabolic interpolation
                let peak: S::Float = self_lag[x.0].to_float_sample();
                let peak_rev: S::Float = self_lag[x.0-1].to_float_sample();
                let peak_fwd: S::Float = self_lag[x.0+1].to_float_sample();
                let dr = 0.5 * (peak_fwd - peak_rev).to_f64().unwrap();
                let d2r = 2. * peak.to_f64().unwrap() - (peak_rev - peak_fwd).to_f64().unwrap();
                let freq = sample_rate / T::from_f64(x.0 as f64 + dr / d2r).unwrap();

                // Calculate the strength using sin(x)/x interpolation
                // let offset = -(brent_ixmax as isize) - 1;
                let offset = 0;
                let nx = (brent_ixmax as isize - offset) as usize;
                let n = (sample_rate / freq - T::from(offset).unwrap()).to_f64().unwrap().to_sample::<S>();
                let mut strn = interpolate_sinc(&self_lag[..], offset, nx, n, 30);

                // Reflect high values due to short sampling periods around 1.0
                if strn > 1. { strn = 1. / strn; }

                println!("x: {:?} to {:?}", x, self_lag[x.0+1]);
                println!("freq: {}\t\tstrn: {}", freq.to_f64().unwrap(), strn.to_f64().unwrap());

                Pitch { frequency: freq, strength: T::from_f64(strn).unwrap() }
            })
            .filter(|x| ((x.frequency) == T::from_f64(0f64).unwrap()) || (x.frequency > min && x.frequency < max))
            .collect();
        maxima.push(Pitch::new(T::from_usize(0).unwrap(), threshold)); // Index of 0 == no pitch
        maxima.sort_by(|a, b| (b.strength).partial_cmp(&a.strength).unwrap());
        maxima
    }
}

#[cfg(test)]
mod tests {
    extern crate sample;

    use super::*;
    use super::super::waves::*;

    use sample::{window, ToSampleSlice};
    use sample::signal::Sine;
    use std::cmp::Ordering;
    use std::f64::consts::PI;

    fn sine(len: usize) -> Vec<f64> {
        let rate = sample::signal::rate(len as f64).const_hz(1.0);
        rate.clone().sine().take(len).collect::<Vec<[f64; 1]>>().to_sample_slice().to_vec()
    }

    #[test]
    fn test_ac() { 
        let sine = sine(16);
        let mut coeffs: Vec<f64> = vec![0.; 16];
        sine.autocorrelate_mut(16, &mut coeffs[..]);
        let out = sine.autocorrelate(16);
        assert_eq!(coeffs, out);
    }

    #[test]
    fn test_pitch() {
        let exp_freq = 200.0;
        let mut signal = sample::signal::rate(44100.).const_hz(exp_freq).sine();
        let vector: Vec<[f64; 1]> = signal.take(4096).collect();
        let mut maxima: f64 = vector.to_sample_slice().max();
        for chunk in window::Windower::hanning(&vector[..], 1024, 512) {
            let chunk_data: Vec<[f64; 1]> = chunk.collect();
            let pitch = chunk_data.to_sample_slice().pitch::<window::Hanning>(44100., 0.2, 0.05, maxima, maxima, 0.01, 100., 500.);
            println!("pitch: {:?}", pitch);
            assert!((pitch[0].frequency - exp_freq).abs() < 1.0);
        }
    }
}
