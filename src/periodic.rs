extern crate num;

use num::{Float, ToPrimitive, FromPrimitive};
use super::waves::Normalize;

use dasp::Sample;
use dasp::sample::FromSample;
use dasp::signal::window::Window;
use dasp::slice::ToSampleSlice;
use dasp::window::Window as WindowType;

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::f64::EPSILON;
use std;

/// Interpolate a given point x within a sampled slice y, using sinc interpolation.
///
/// ```rust
/// sin(x) / x
/// ```
///
/// This windows around the target point `x` using a Hanning window. 
///
/// [Source](http://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf): 
/// Boersma, Paul. "Accurate short-term analysis of the fundamental frequency and the
/// harmonics-to-noise ratio of a sampled sound." Institute of Phonetic Sciences, University of
/// Amsterdam, Proceedings 17 (1993), 97-110.
pub fn interpolate_sinc<S: Sample>(y: &[S], offset: isize, nx: usize, x: S, mut max_depth: usize) -> f64 {
    let x = x.to_float_sample().to_sample::<f64>();
    let nl = x.floor() as usize;
    let nr = nl + 1;
    let phil = x - nl as f64;
    let phir = 1. - phil;
    let mut result = 0.;

    // simple cases
    if nx < 1 { return std::f64::NAN }
    if x > nx as f64 { return y[offset as usize + nx as usize - 1].to_float_sample().to_sample::<f64>() }
    if x < 0. { return y[0].to_float_sample().to_sample::<f64>() }
    if (x - nl as f64).abs() < 1.0e-10 { return y[offset as usize + nl].to_float_sample().to_sample::<f64>() };
    if (x - nr as f64).abs() < 1.0e-10 { return y[offset as usize + nr].to_float_sample().to_sample::<f64>() };

    // Protect against usize underflow in indexing the lag vector
    // Clip max_depth to offset + nr at the lowest point
    if (offset + nr as isize) < max_depth as isize {
        if (offset + nr as isize) < 0 {
            max_depth = 0;
        } else {
            max_depth = (offset + nr as isize) as usize;
        }
    }

    // Clip max_depth to nx - offset + nl - 1 at the highest point
    if (offset + nl as isize + max_depth as isize) >= nx as isize {
        max_depth = (nx as isize - offset + nl as isize - 1) as usize;
    }

    for n in 0..(max_depth+1) {
        // Sum the values to the left of the sample
        result += {
            // a is PI * (the scalar + nsamp away from the source)
            let a = PI * (phil + n as f64);
            let mut lag_val = offset as i32 + nr as i32 - n as i32;
            if lag_val < 0 { lag_val = 0; }
            // each element
            let r_lag = y[lag_val as usize].to_float_sample().to_sample::<f64>();
            // this is sinc
            let first = a.sin() / a;
            let second = 0.5 + 0.5 * (a / (phil + max_depth as f64)).cos();
            r_lag * first * second
        };
        // Sum the values to the right of the sample
        result += {
            let a = PI * (phir + n as f64);
            let mut lag_val = offset as i32 + nl as i32 + n as i32;
            if lag_val < 0 { lag_val = 0; }
            if lag_val >= y.len() as i32 { lag_val = y.len() as i32 - 1; }
            let r_lag = y[lag_val as usize].to_float_sample().to_sample::<f64>();
            let first = a.sin() / a;
            let second = 0.5 + 0.5 * (a / (phir + max_depth as f64)).cos();
            r_lag * first * second
        };
    }

    result
}

pub enum Interpolation {
    None,
    Parabolic,
    Sinc(usize),
}

struct BrentParams<'a, S: Sample + 'a> {
    y: &'a [S],
    offset: isize,
    depth: usize,
    ixmax: usize,
    is_max: bool
}

fn brent_maximize<'a, S: Sample>(f: &Fn(f64, &BrentParams<'a, S>) -> f64, 
                             bounds: (f64, f64),
                             params: &'a BrentParams<S>,
                             tol: f64, fx: &mut f64) -> f64 {
    let (mut a, mut b) = bounds;
    let golden = 1. - 0.6180339887498948482045868343656381177203091798057628621;
    let sqrt_epsilon = EPSILON.sqrt();
    let itermax = 60;

    assert!(tol > 0.);
    assert!(a < b);
    let mut v = a + golden * (b - a);
    let mut fv = f(v, params);
    let mut x = v;
    let mut w = v;
    *fx = fv;
    let mut fw = fv;

    for _ in 1..(itermax+1) {
        let range = b - a;
        let middle_range = (a + b) * 0.5;
        let tol_act = sqrt_epsilon * x.abs() + tol / 3.;

        if (x - middle_range).abs() + range * 0.5 <= 2. * tol_act {
            return x;
        }

        let mut new_step = if x < middle_range {
            golden * (b - x)
        } else {
            golden * (a - x)
        };

        if (x - w).abs() >= tol_act {
            let t = (x - w) * (*fx - fv);
            let mut q = (x - v) * (*fx - fw);
            let mut p = (x - v) * q - (x - w) * t;
            q = 2. * q - t;

            if q > 0. {
                p = -p;
            } else {
                q = -q;
            }
            if p.abs() < (new_step * q).abs() &&
            p > q * (a - x + 2. * tol_act) &&
            p < q * (b - x - 2. * tol_act) {
                new_step = p / q;
            }
        }

        if new_step.abs() < tol_act {
            new_step = if new_step > 0. { tol_act } else { -tol_act };
        }

        {
            let t = x + new_step;
            let ft = f(t, params);

            if ft <= *fx {
                if t < x {
                    b = x;
                } else {
                    a = x;
                }
                v = w; w = x; x = t;
                fv = fw; fw = *fx; *fx = ft;
            } else {
                if t < x {
                    a = t;
                } else { 
                    b = t;
                }

                if ft <= fw || (w - x).abs() < EPSILON {
                    v = w; w = t;
                    fv = fw; fw = ft;
                } else if ft <= fv || (v - x).abs() < EPSILON || (v - w).abs() < EPSILON {
                    v = t;
                    fv = ft;
                }
            }
        }
    }
    x
}


/// Returns (xmid, ymid) for the maximum sample index and sample value
pub fn improve_extremum<S: Sample + FromSample<f64>>(y: &[S], offset: isize, nx: usize, ixmid: f64, interp: Interpolation, is_max: bool) -> (f64, f64) {
    if ixmid == 0. { return (0., y[0].to_float_sample().to_sample::<f64>()) }
    if ixmid >= nx as f64 { return (nx as f64, y[nx-1].to_float_sample().to_sample::<f64>()) }

    match interp {
        Interpolation::None => {
            (0., y[0].to_float_sample().to_sample::<f64>())
        },
        Interpolation::Parabolic => {
            let diff = y[ixmid.floor() as usize + 1].to_float_sample().to_sample::<f64>() - y[ixmid.floor() as usize - 1].to_float_sample().to_sample::<f64>();
            let mid = y[ixmid.floor() as usize].to_float_sample().to_sample::<f64>();
            let dy = 0.5 * diff;
            let d2y = 2.0 * mid - diff;
            let ixmid_real = ixmid as f64 + dy / d2y;
            (ixmid_real, mid + 0.5 * dy * dy / d2y)
        },
        Interpolation::Sinc(max_depth) => {
            let params = BrentParams {
                y: y,
                offset: offset,
                depth: max_depth,
                ixmax: nx,
                is_max: is_max
            };
            let f = |x: f64, params: &BrentParams<S>| -> f64 {
                let out = interpolate_sinc(params.y, params.offset, params.ixmax, x.to_sample::<S>(), params.depth);
                if params.is_max {
                    out
                } else {
                    -out
                }
            };
            let mut result: f64 = 0.;
            let (a, b) = (ixmid as f64 - 1., ixmid as f64 + 1.);
            let ixmid_real = brent_maximize(&f, (a, b), &params, 1e-10, &mut result);
            (ixmid_real, result)
        }
    }
}

pub struct HanningLag;

impl<S> WindowType<S> for HanningLag
    where S: Sample
{
    type Output = S;

    fn window(phase: S) -> Self::Output {
        let pi_2 = (PI * 2.).to_sample();
        let v: f64 = (phase.to_float_sample() * pi_2).to_sample::<f64>();
        let one_third: S::Float = (1.0 / 3.0).to_sample();
        let two_thirds: S::Float = (2.0 / 3.0).to_sample();
        let one: S::Float = 1.0.to_sample();
        ((one - phase.to_float_sample()) * (two_thirds + (one_third * v.cos().to_sample()).to_sample::<S::Float>())
            + (one / pi_2) * v.sin().to_sample()).to_sample::<S>()
    }
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
    fn autocorrelate_mut(&self, coeffs: &mut [T]);
    fn autocorrelate(&self, n_coeffs: usize) -> Vec<T> {
        let mut coeffs: Vec<T> = vec![T::EQUILIBRIUM; n_coeffs];
        self.autocorrelate_mut(&mut coeffs[..]);
        coeffs
    }
}

impl<T> Autocorrelate<T> for [T] 
    where T: Sample
{
    fn autocorrelate_mut(&self, coeffs: &mut [T]) {
        for (lag, coeff) in coeffs.iter_mut().enumerate() {
            *coeff = self.iter().enumerate()
                .take(self.len() - lag)
                .skip(1)
                .fold(self[0], |accum, (i, sample)| { 
                    accum.add_amp(sample.mul_amp(self[(i + lag) as usize].to_float_sample()).to_signed_sample())
                });
        }
    }
}

impl<T> Autocorrelate<T> for VecDeque<T>
    where T: Sample
{
    fn autocorrelate_mut(&self, coeffs: &mut [T]) {
        for (lag, coeff) in coeffs.iter_mut().enumerate() {
            *coeff = self.iter().enumerate()
                .take(self.len() - lag)
                .skip(1)
                .fold(self[0], |accum, (i, sample)| { 
                    accum.add_amp(sample.mul_amp(self[(i + lag) as usize].to_float_sample()).to_signed_sample())
                });
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

#[allow(dead_code)]
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
    fn pitch<W: WindowType<f64, Output=f64>>(&self, sample_rate: T, threshold: T, local_peak: S, global_peak: S, min: T, max: T) -> Vec<Pitch<T>>;
}

/// Trait for finding local maxima in a given slice. `local_maxima` should return `Vec<(bin,
/// value)>` where `bin` is the index of the maximum and `value` is the value at that index.
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
    where S: Sample + FromSample<f64>, 
          S::Float: ToPrimitive,
          T: Float + FromPrimitive
{
    /// Find the pitch of a slice of samples, using the autocorrelation technique described in
    /// Boersma 1993. This function assumes that the slice has already been windowed in some way,
    /// and the window must have a corresponding autocorrelation function.
    ///
    /// First pass interpolates the peaks based on parabolic interpolation, using sinc
    /// interpolation to find the values of the peaks for calculating harmonics-to-noise ratio (the
    /// "strength" of the signal).
    ///
    /// Second pass maximizes the function more accurately using a 700-sample depth sinc
    /// interpolation and the Brent golden section parabolic maximization algorithm. This results
    /// in a collection of possible pitches and their given HNR ratings.
    ///
    /// A third pass, using PitchExtractor, should find a path through these candidates that
    /// maximizes both the smoothness of the pitch contour and the strength of the pitches.
    fn pitch<W: WindowType<f64, Output=f64>>(&self, sample_rate: T, threshold: T, local_peak: S, global_peak: S, min: T, max: T) -> Vec<Pitch<T>> {
        // TODO: need 2 empty mutable Vecs, 
        // self_lag: [T; 2*self.len()]
        // maxima: [Pitch<T>; max_maxima], theoretically could be up to (0.5 * self.len()).ceil()
        let window_lag = Window::<[S; 1], W>::new(self.len()).take(self.len()).map(|x| x.to_sample_slice()[0]);

        // TODO: remove allocation
        let mut self_lag = self.autocorrelate(self.len());
        self_lag.normalize();

        for (s, w) in self_lag.iter_mut().zip(window_lag) {
            *s = (s.to_float_sample() / w.to_float_sample()).to_sample::<S>();
        }

        // TODO: remove allocation
        self_lag.resize(self.len() * 2, S::from_sample(0.));

        let interpolation_depth = 0.5;
        let brent_ixmax = (interpolation_depth * self.len() as f64).floor() as usize;

        // TODO: remove allocation
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
                // Not entirely sure what "offset" does
                let offset = -(brent_ixmax as isize) - 1;
                let nx = (brent_ixmax as isize - offset) as usize;
                // Assumed frequency
                let n = (sample_rate / freq - T::from(offset).unwrap()).to_f64().unwrap().to_sample::<S>();
                let mut strn = interpolate_sinc(&self_lag[..], offset, nx, n, 30);
                // Reflect high values due to short sampling periods around 1.0
                if strn > 1. { strn = 1. / strn; }

                Pitch { frequency: freq, strength: T::from_f64(strn).unwrap() }
            })
            .filter(|x| ((x.frequency) == T::from_f64(0f64).unwrap()) || (x.frequency > min && x.frequency < max))
            .map(|mut p| {
                let offset = -(brent_ixmax as isize) - 1;
                let nx = (brent_ixmax as isize - offset) as usize;
                let n = (sample_rate / p.frequency - T::from(offset).unwrap()).to_f64().unwrap();
                let (mut xmid, mut ymid) = improve_extremum(&self_lag[..], offset, nx, n, Interpolation::Sinc(1200), true);
                xmid += offset as f64;
                if ymid > 1. { ymid = 1. / ymid; }
                p.frequency = sample_rate / T::from(xmid).unwrap();
                p.strength = T::from(ymid).unwrap();
                p
            })
            .collect();
        maxima.push(Pitch::new(T::from_usize(0).unwrap(), threshold)); // Index of 0 == no pitch
        maxima.sort_by(|a, b| (b.strength).partial_cmp(&a.strength).unwrap());
        maxima
    }
}

#[cfg(test)]
mod tests {
    extern crate dasp;

    use super::*;
    use super::super::waves::*;

    use dasp::Signal;
    use dasp::signal::Sine;
    use dasp::signal::window::Windower;
    use dasp::window::Hanning;
    use std::cmp::Ordering;
    use std::f64::consts::PI;

    fn sine(len: usize) -> Vec<f64> {
        let rate = dasp::signal::rate(len as f64).const_hz(1.0);
        rate.clone().sine().take(len).collect::<Vec<f64>>().to_sample_slice().to_vec()
    }

    #[test]
    fn test_ac() { 
        let sine = sine(16);
        let mut coeffs: Vec<f64> = vec![0.; 16];
        sine.autocorrelate_mut(&mut coeffs[..]);
        let out = sine.autocorrelate(16);
        assert_eq!(coeffs, out);
    }

    #[test]
    fn test_pitch() {
        let exp_freq = 150.0;
        let bin = 2048;
        let hop = 1024;

        let mut signal = dasp::signal::rate(44100.).const_hz(exp_freq).sine();
        let vector: Vec<f64> = signal.take(bin + 1).collect();
        let mut maxima: f64 = vector.to_sample_slice().max_amplitude();
        for chunk in Windower::hanning(&vector[..], bin, hop) {
            let chunk_data: Vec<f64> = chunk.take(bin).collect();
            let pitch = chunk_data.to_sample_slice().pitch::<Hanning>(44100., 0.2, maxima, maxima, 100., 500.);
            println!("pitch: {:?}", pitch);
            assert!((pitch[0].frequency - exp_freq).abs() < 1.0e-2);
        }
    }
}
