extern crate num;

use num::{Float, FromPrimitive, Zero};

use std::f64::consts::PI;
use std::ops::*;
use std::cmp::PartialOrd;
use std::collections::VecDeque;
use std::marker::Sized;

use super::waves::{WindowType, Window, Windower, Normalize};

pub trait HasLength {
    fn len(&self) -> usize;
}

impl<T> HasLength for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<'a, T> HasLength for &'a [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> HasLength for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> HasLength for VecDeque<T> {
    fn len(&self) -> usize {
        VecDeque::len(self)
    }
}

pub trait Autocorrelate<'a, T> {
    fn autocorrelate(&self, n_coeffs: usize) -> Vec<T>;
    fn autocorrelate_mut(&self, n_coeffs: usize, coeffs: &'a mut [T]) -> &'a mut [T];
}

impl<'a, V, T> Autocorrelate<'a, T> for V
    where T: Mul<T, Output=T> + 
             Add<T, Output=T> + 
             Div<T, Output=T> +
             Zero +
             Copy + 
             PartialOrd,
          V: Index<usize, Output=T> +
             HasLength 
{ 
    fn autocorrelate(&self, n_coeffs: usize) -> Vec<T> {
        let mut coeffs: Vec<T> = vec![T::zero(); n_coeffs];
        self.autocorrelate_mut(n_coeffs, &mut coeffs);
        coeffs
    }

    fn autocorrelate_mut(&self, n_coeffs: usize, coeffs: &'a mut [T]) -> &'a mut [T] {
        for lag in 0..n_coeffs {
            let mut accum = self[0];
            for i in 1..(self.len() - (lag)) {
                accum = accum + (self[i] * self[(i + lag) as usize]);
            }
            coeffs[lag] = accum;
        }
        coeffs
    }
}

#[derive(Clone, Debug)]
pub struct Pitch<T> {
    pub frequency: T,
    pub strength: T
}

impl<T> Pitch<T> {
    pub fn new(frequency: T, strength: T) -> Self {
        Pitch { frequency: frequency, strength: strength }
    }
}

pub struct PitchExtractor<'a, T: 'a + Float> {
    frame_index: usize,
    voiced_unvoiced_cost: T,
    voicing_threshold: T,
    candidates: &'a Vec<Vec<Pitch<T>>>,
    current_cost: T
}

impl<'a, T: 'a + Float> PitchExtractor<'a, T> {
    pub fn new(candidates: &'a Vec<Vec<Pitch<T>>>, voiced_unvoiced_cost: T, voicing_threshold: T) -> Self {
        PitchExtractor {
            frame_index: 0,
            voiced_unvoiced_cost: voiced_unvoiced_cost,
            voicing_threshold: voicing_threshold,
            candidates: candidates,
            current_cost: T::zero()
        }
    }
}

impl<'a, T: 'a + Float> Iterator for PitchExtractor<'a, T> {
    type Item = Pitch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.candidates.len() == self.frame_index {
            return None;
        }

        self.frame_index += 1;
        Some(self.candidates[self.frame_index][0].clone())
    }
}

pub trait HasPitch<T> {
    fn pitch(&self, sample_rate: T, threshold: T, silence_threshold: T, local_peak: T, global_peak: T, octave_cost: T, min: T, max: T, window_type: WindowType) -> Vec<Pitch<T>>;
    fn local_maxima(&self) -> Vec<(usize, T)>;
}

impl<T: Float + PartialOrd + FromPrimitive> HasPitch<T> for Vec<T> {
    // Assumes that it's being passed a windowed signal
    // Returns (frequency, strength)
    fn pitch(&self, sample_rate: T, threshold: T, silence_threshold: T, local_peak: T, global_peak: T, octave_cost: T, min: T, max: T, window_type: WindowType) -> Vec<Pitch<T>> {
        let window_lag: Vec<T> = match window_type {
            WindowType::Hanning => { Window::<T>::new(WindowType::HanningAutocorrelation, self.len()).collect() },
            _ => { panic!() }
        };
        let mut self_lag = self.autocorrelate(self.len());
        self_lag.normalize();
        for i in 0..self.len() {
            self_lag[i] = self_lag[i] / window_lag[i];
        };

        let voiceless = (T::zero(), threshold + T::zero().max(T::from_i32(2).unwrap() - (local_peak / global_peak) / (silence_threshold / T::one() + threshold)));

        let local_max: Vec<(usize, T)> = self_lag.local_maxima();
        let mut maxima: Vec<Pitch<T>> = local_max.iter()
            .map(|x| {
                let freq = sample_rate / T::from_usize(x.0).unwrap();
                let strn = x.1 - (octave_cost.powi(2) * (min * freq).log2());
                Pitch { frequency: freq, strength: strn }
            })
            .filter(|x| ((x.frequency) == T::from_f64(0f64).unwrap()) || (x.frequency > min && x.frequency < max))
            .collect();
        maxima.push(Pitch::new(T::from_usize(0).unwrap(), threshold)); // Index of 0 == no pitch
        maxima.sort_by(|a, b| (b.strength).partial_cmp(&a.strength).unwrap());
        maxima
    }

    // Find the local maxima for a vector. Skips the one at index 0.
    fn local_maxima(&self) -> Vec<(usize, T)> {
        self.windows(3).enumerate().filter(|x| {
            x.1[0] < x.1[1] && x.1[2] < x.1[1]
        }).map(|x| ((x.0 + 1), x.1[1])).collect()
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
        sine.autocorrelate(16);
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
