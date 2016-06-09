extern crate num;

use num::{Float, ToPrimitive, FromPrimitive};
use super::waves::Normalize;

use sample;
use sample::window::Window;
use sample::{Sample, ToSampleSlice};


pub trait LagType: sample::window::Type {
    type Lag: sample::window::Type;
}

pub struct HanningLag;

impl sample::window::Type for HanningLag {
    fn at_phase<S: Sample>(phase: S) -> S {
        let pi_2 = <S::Float as sample::Float>::pi() * 2.0.to_sample();
        let v = phase.to_float_sample() * pi_2;
        let one_third: S::Float = (1.0 / 3.0).to_sample();
        let two_thirds: S::Float = (2.0 / 3.0).to_sample();
        let one: S::Float = 1.0.to_sample();
        ((one - phase.to_float_sample()) * (two_thirds + (one_third * sample::Float::cos(v))) 
            + (one / pi_2) * sample::Float::sin(v)).to_sample::<S>()
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
    where S: Sample, 
          S::Float: ToPrimitive,
          T: Float + FromPrimitive
{
    /// Assumes that it's being passed a windowed signal
    fn pitch<W: LagType>(&self, sample_rate: T, threshold: T, silence_threshold: S, local_peak: S, global_peak: S, octave_cost: T, min: T, max: T) -> Vec<Pitch<T>> {
        let window_lag: Vec<S> = Window::<[S; 1], W>::new(self.len()).take(self.len()).map(|x| x.to_sample_slice()[0]).collect();

        let mut self_lag = self.autocorrelate(self.len());
        self_lag.normalize();

        for i in 0..self.len() {
            self_lag[i] = (self_lag[i].to_float_sample() / window_lag[i].to_float_sample()).to_sample::<S>();
        };

        let voiceless = (T::zero(), threshold + T::zero().max(T::one() + T::one() - (T::from(local_peak.to_float_sample()).unwrap() / T::from(global_peak.to_float_sample()).unwrap()) / (T::from(silence_threshold.to_float_sample()).unwrap() / T::one() + threshold)));

        let mut maxima: Vec<Pitch<T>> = self_lag.local_maxima().iter()
            .map(|x| {
                let freq = sample_rate / T::from_usize(x.0).unwrap();
                let mut strn = T::from(x.1.to_float_sample()).unwrap() - (octave_cost.powi(2) * (min * freq).log2());
                if strn > T::from_f64(1.).unwrap() { strn = T::from_f64(1.).unwrap() / strn; }
                Pitch { frequency: freq, strength: strn }
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
    use super::*;
    use super::super::waves::*;
    use std::cmp::Ordering;
    use std::f64::consts::PI;

    #[test]
    fn test_ac() { 
        let sine = Vec::<f64>::sine(16);
        let mut coeffs: Vec<f64> = vec![0.; 16];
        sine.autocorrelate_mut(16, &mut coeffs[..]);
        let out = sine.autocorrelate(16);
        assert_eq!(coeffs, out);
    }

    #[test]
    fn test_pitch() {
        let mut signal = Vec::<f64>::with_capacity(128);
        for i in 0..128 {
            let phase = (i as f64) / 128f64;
            signal.push(
                (1f64 + 0.3f64 * (140f64 * 2f64 * PI * phase).sin()) *
                (280f64 * 2f64 * PI * phase).sin()
            );
        };

        let mut vector = Vec::<f64>::with_capacity(512);
        for _ in 0..4 { vector.extend(signal.iter().cloned()); }
        let mut sortable = vector.clone();
        sortable.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let global_peak = sortable[0];
        let local_peak = global_peak.clone();
        let mut auto = vector.autocorrelate(512);
        auto.normalize();
        let maxima = auto.local_maxima();
        assert_eq!(maxima.len(), 95);
        let len = vector.len();
        for (v, w) in vector.iter_mut().zip(Window::<f64>::new(WindowType::Hanning, len)) {
            *v *= w;
        }
        let pitch = vector.pitch(512f64, 0.2f64, 0.05, local_peak, global_peak, 0.01, 10f64, 100f64, WindowType::Hanning);
        assert_eq!(pitch[0].frequency, 16f64);
    }

}
