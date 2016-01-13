#![feature(box_raw, plugin, test)]
#![plugin(clippy)]

extern crate num;
extern crate sample;
extern crate libc;

use num::complex::Complex;
use num::{Float, FromPrimitive};
use std::iter::Take;
use sample::Sample;
use libc::{size_t, c_int, c_void};
use std::mem;

// Declare local mods
pub mod complex;
pub mod polynomial;
pub mod waves;
pub mod resample;

// Use std
use std::iter::Iterator;
use std::f64::consts::PI;
use std::ops::*;
use std::cmp::Ordering::{Less, Equal, Greater};
use std::cmp::PartialOrd;
use std::fmt::Debug;

use waves::Windowing;
use waves::Window;

use complex::{SquareRoot, ToComplexVec, ToComplex};
use polynomial::Polynomial;

pub trait Autocorrelates<'a, T> {
    fn autocorrelate(&self, n_coeffs: usize) -> Vec<T>;
    fn autocorrelate_mut(&self, n_coeffs: usize, coeffs: &'a mut [T]) -> &'a mut [T];
    fn normalize(&mut self);
    fn max(&self) -> T;
}

impl<'a, T> Autocorrelates<'a, T> for [T] where T: Mul<T, Output=T> + Add<T, Output=T> + Copy + std::cmp::PartialOrd + Div<T, Output=T> {
    fn autocorrelate(&self, n_coeffs: usize) -> Vec<T> {
        let mut coeffs: Vec<T> = Vec::with_capacity(n_coeffs);
        for lag in 0..n_coeffs {
            let mut accum = self[0];
            for i in 1..(self.len() - (lag)) {
                accum = accum + (self[i] * self[(i + lag) as usize]);
            }
            coeffs.push(accum);
        }
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

    fn normalize(&mut self) {
        let max = self.max();
        for i in 0..self.len() {
            self[i] = self[i] / max;
        }
    }
}

pub trait LPC<T> {
    fn lpc_mut(&self, n_coeffs: usize, ac: &mut [T], kc: &mut [T], tmp: &mut [T]);
    fn lpc(&self, n_coeffs: usize) -> Vec<T>;
}

impl<T> LPC<T> for [T] where T: Float { 
    fn lpc_mut(&self, n_coeffs: usize, ac: &mut [T], kc: &mut [T], tmp: &mut [T]) {
        /* order 0 */
        let mut err = self[0];
        ac[0] = T::one();

        /* order >= 1 */
        for i in 1..n_coeffs+1 {
            let mut acc = self[i];
            for j in 1..i {
                acc = acc + (ac[j] * self[i-j]);
            }
            kc[i-1] = acc.neg() / err;
            ac[i] = kc[i-1];
            for j in 0..n_coeffs {
                tmp[j] = ac[j];
            }
            for j in 1..i {
                ac[j] = ac[j] + (kc[i-1] * tmp[i-j]);
            }
            err = err * (T::one() - (kc[i-1] * kc[i-1]));
        };
    }

    fn lpc(&self, n_coeffs: usize) -> Vec<T> {
        let mut ac: Vec<T> = vec![T::zero(); n_coeffs + 1];
        let mut kc: Vec<T> = vec![T::zero(); n_coeffs];
        let mut tmp: Vec<T> = vec![T::zero(); n_coeffs];
        self.lpc_mut(n_coeffs, &mut ac[..], &mut kc[..], &mut tmp[..]);
        ac
    }
}

#[derive(Clone, Debug)]
pub struct Resonance<T> {
    pub frequency: T,
    pub amplitude: T
}

impl<T: Float + FromPrimitive> Resonance<T> {
    pub fn from_root(root: &Complex<T>, sample_rate: T) -> Option<Resonance<T>> {
        let freq_mul: T = T::from_f64(sample_rate.to_f64().unwrap() / (PI * 2f64)).unwrap();
        if root.im >= T::zero() {
            let res = Resonance::<T> { 
                frequency: root.im.atan2(root.re) * freq_mul,
                amplitude: (root.im.powi(2) + root.re.powi(2)).sqrt()
            };
            if res.frequency > T::one() {
                Some(res)
            } else { None }
        } else { None }
    }
}

pub trait ToResonance<T> {
    fn to_resonance(&self, sample_rate: T) -> Vec<Resonance<T>>;
}

impl<T> ToResonance<T> for Vec<Complex<T>> where T: Float + FromPrimitive {
    // Give it some roots, it'll find the resonances
    fn to_resonance(&self, sample_rate: T) -> Vec<Resonance<T>> {
        let mut res: Vec<Resonance<T>> = self.iter()
            .filter_map(|r| Resonance::<T>::from_root(r, sample_rate)).collect();
        res.sort_by(|a, b| (a.frequency.partial_cmp(&b.frequency)).unwrap());
        res
    }
}

pub struct FormantFrame<T: Float> {
    frequency: T,
}

pub struct FormantExtractor<'a, T: 'a + Float> {
    num_formants: usize,
    frame_index: usize,
    resonances: &'a Vec<Vec<T>>,
    pub estimates: Vec<T>
}

impl<'a, T: 'a + Float + PartialEq> FormantExtractor<'a, T> {
    pub fn new(
        num_formants: usize, 
        resonances: &'a Vec<Vec<T>>, 
        starting_estimates: Vec<T>) -> Self {
        FormantExtractor { 
            num_formants: num_formants, 
            resonances: resonances, 
            frame_index: 0,
            estimates: starting_estimates 
        }
    }
}

impl<'a, T: 'a + Float + PartialEq + FromPrimitive> Iterator for FormantExtractor<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.resonances.len() == self.frame_index {
            return None;
        }

        let frame = self.resonances[self.frame_index].clone();
        let mut slots: Vec<Option<T>> = self.estimates.iter()
        .map(|estimate| {
            let mut indices: Vec<usize> = (0..frame.len()).collect();
            indices.sort_by(|a, b| {
                (frame[*a] - *estimate).abs().partial_cmp(&(frame[*b] - *estimate).abs()).unwrap()
            });
            let win = indices.first().unwrap().clone();
            Some(frame[win])
        }).collect();

        // Step 3: Remove duplicates. If the same peak p_j fills more than one slots S_i keep it
        // only in the slot S_k which corresponds to the estimate EST_k that it is closest to in
        // frequency, and remove it from any other slots.
        let mut w: usize = 0;
        let mut has_unassigned: bool = false;
        for r in 1..slots.len() {
            match slots[r] {
                Some(v) => { 
                    if v == slots[w].unwrap() {
                        if (v - self.estimates[r]).abs() < (v - self.estimates[w]).abs() {
                            slots[w] = None;
                            has_unassigned = true;
                            w = r;
                        } else {
                            slots[r] = None;
                            has_unassigned = true;
                        }
                    } else {
                        w = r;
                    }
                },
                None => { }
            }
        }

        if has_unassigned {
            // Step 4: Deal with unassigned peaks. If there are no unassigned peaks p_j, go to Step 5.
            // Otherwise, try to fill empty slots with peaks not assigned in Step 2 as follows.
            for j in 0..frame.len() {
                let peak = Some(frame[j]);
                if slots.contains(&peak) { continue; }
                match slots.clone().get(j) {
                    Some(&s) => {
                        match s {
                            Some(_) => { },
                            None => { slots[j] = peak; continue; }
                        }
                    }
                    None => { }
                }
                if j > 0 && j < slots.len() {
                    match slots.clone().get(j-1) {
                        Some(&s) => {
                            match s {
                                Some(_) => { },
                                None => { slots.swap(j, j-1); slots[j] = peak; continue; }
                            }
                        }
                        None => { }
                    }
                }
                match slots.clone().get(j+1) {
                    Some(&s) => {
                        match s {
                            Some(_) => { },
                            None => { slots.swap(j, j+1); slots[j] = peak; continue; }
                        }
                    }
                    None => { }
                }
            }
        }

        let mut winners: Vec<T> = slots.iter().filter_map(|v| *v).filter(|v| *v > T::zero()).collect();
        self.estimates = winners.clone();
        self.frame_index += 1;
        winners.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(winners)
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

pub trait HasRMS<T> {
    fn rms(&self) -> T;
}

impl HasRMS<f64> for Vec<f64> {
    fn rms(&self) -> f64 {
        (self.iter().fold(0f64, |acc, &item: &f64| acc + item.powi(2)) / (self.len() as f64)).sqrt()
    }
}

pub trait HasPitch<T> {
    fn pitch(&self, sample_rate: T, threshold: T, silence_threshold: T, local_peak: T, global_peak: T, octave_cost: T, min: T, max: T, window_type: Window) -> Vec<Pitch<T>>;
    fn local_maxima(&self) -> Vec<(usize, T)>;
}

impl<T: Float + PartialOrd + FromPrimitive + Debug> HasPitch<T> for Vec<T> {
    // Assumes that it's being passed a windowed signal
    // Returns (frequency, strength)
    fn pitch(&self, sample_rate: T, threshold: T, silence_threshold: T, local_peak: T, global_peak: T, octave_cost: T, min: T, max: T, window_type: Window) -> Vec<Pitch<T>> {
        let window_lag: Vec<T> = match window_type {
            Window::Hanning => { Vec::<T>::hanning_autocor(self.len()) },
            _ => { panic!() }
        };
        let mut self_lag = self[..].autocorrelate(self.len());
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

#[no_mangle]
pub unsafe extern fn vox_box_autocorrelate_f32(input: *mut f32, size: size_t, n_coeffs: size_t) -> *mut [f32] {
    let buf = Vec::<f32>::from_raw_parts(input, size, size);
    let mut auto = buf.autocorrelate(n_coeffs);
    auto.normalize();
    let out = Box::into_raw(auto.into_boxed_slice());
    mem::forget(buf); // don't want to free this one
    out
}

/// Calculates autocorrelation without allocating any memory
///
/// const float* input: input buffer to calculate from
/// size_t size:        size of input buffer
/// size_t n_coeffs:    number of coefficients to calculate
/// float* coeffs:      output buffer
#[no_mangle]
pub unsafe extern fn vox_box_autocorrelate_mut_f32(input: *const f32, size: size_t, n_coeffs: size_t, coeffs: *mut f32) {
    let buf = std::slice::from_raw_parts(input, size);
    let mut cof = std::slice::from_raw_parts_mut(coeffs, size);
    buf.autocorrelate_mut(n_coeffs, &mut cof);
}

#[no_mangle]
pub unsafe extern fn vox_box_resample_mut_f32(input: *const f32, size: size_t, new_size: size_t, out: *mut f32) {
    let buf = std::slice::from_raw_parts(input, size);
    let mut resampled = std::slice::from_raw_parts_mut(out, new_size);
    for i in 0..new_size {
        let phase = (i as f32) / ((new_size-1) as f32);
        let index = phase * ((buf.len()-1) as f32);
        let a = buf[index.floor() as usize];
        let b = buf[index.ceil() as usize];
        let t = (index - index.floor()) as f32;
        resampled[i] = a + (b - a) * t;
    }
}

/// Normalizes the input buffer.
///
/// float* buffer: buffer to be normalized
/// size_t size:   size of buffer
#[no_mangle]
pub unsafe extern fn vox_box_normalize_f32(buffer: *mut f32, size: size_t) {
    let mut buf = std::slice::from_raw_parts_mut(buffer, size);
    buf.normalize();
}

#[no_mangle]
pub unsafe extern fn vox_box_lpc_f32(buffer: *mut f32, size: size_t, n_coeffs: size_t) -> *mut [f32] {
    let buf = Vec::<f32>::from_raw_parts(buffer, size, size);
    let out = Box::into_raw(buf.lpc(n_coeffs).into_boxed_slice());
    mem::forget(buf);
    out
}

/// Given a set of autocorrelation coefficients, calculates the LPC coefficients using a mutable
/// buffer. This is the preferred way to calculate LPC repeatedly with a changing buffer, as it
/// does not allocate any memory on the heap.
///
/// float* coeffs: autocorrelation coefficients
/// size_t size:   size of the autocorrelation coefficient vector
/// size_t n_coeffs: number of coefficients to find
/// float* out:    coefficient output buffer, c type float*. Must be at least (sizeof(float)*n_coeffs)+1.
/// float* work:   workspace for the LPC calculation, to avoid allocs. Must be at least
///                (sizeof(float)*n_coeffs*2).
#[no_mangle]
pub unsafe extern fn vox_box_lpc_mut_f32(coeffs: *const f32, size: size_t, n_coeffs: size_t, out: *mut f32, work: *mut f32) {
    let buf = std::slice::from_raw_parts(coeffs, size);
    let mut lpc = std::slice::from_raw_parts_mut(out, n_coeffs + 1);
    let mut kc = std::slice::from_raw_parts_mut(work, n_coeffs);
    let mut tmp = std::slice::from_raw_parts_mut(work.offset(n_coeffs as isize), n_coeffs);
    buf.lpc_mut(n_coeffs, lpc, kc, tmp);
}

#[no_mangle]
pub unsafe extern fn vox_box_resonances_f32(buffer: *mut f32, size: size_t, sample_rate: f32, count: &mut c_int) -> *mut [f32] {
    let buf = std::slice::from_raw_parts(buffer, size);
    let complex = buf.to_vec().to_complex_vec(); // fix this allocation
    let res: Vec<f32> = complex.find_roots().unwrap().to_resonance(sample_rate).iter().map(move |r| r.frequency).collect();
    *count = res.len() as c_int;
    Box::into_raw(res.into_boxed_slice())
}

/// work must be 3*size+2 for complex floats (meaning 6*size+4 of the buffer)
#[no_mangle]
pub unsafe extern fn vox_box_resonances_mut_f32<'a>(buffer: *const f32, size: size_t, sample_rate: f32, count: &mut c_int, work: *mut Complex<f32>, out: *mut f32) {
    // Input buffer
    let buf: &[f32] = std::slice::from_raw_parts(buffer, size);
    let mut res: &mut [f32] = std::slice::from_raw_parts_mut(out, size);
    // Mutable complex slice
    let mut complex: &mut [Complex<f32>] = std::slice::from_raw_parts_mut(work, size); // designate memory for the complex vector
    let mut complex_work: &'a mut [Complex<f32>] = std::slice::from_raw_parts_mut(work.offset(size as isize), size*4 + 2); // designate memory for the complex vector
    for i in 0..size {
        complex[i] = (&buf[i]).to_complex();
    }
    match complex.find_roots_mut(complex_work) {
        Ok(_) => { },
        Err(x) => { println!("Problem: {:?}", x) }
    };
    let freq_mul: f32 = (sample_rate as f64 / (PI * 2f64)) as f32;
    for i in 0..size {
        if complex[i].im >= 0f32 {
            let c = complex[i].im.atan2(complex[i].re) * freq_mul;
            if c > 1f32 {
                res[*count as usize] = c;
                *count = *count + 1;
            }
        } 
    }
    let rpos = res.iter().rposition(|v| *v != 0f32).unwrap_or(0);
    res[0..(rpos+1)].sort_by(|a, b| (a.partial_cmp(b)).unwrap_or(Equal));

    // let res: Vec<f32> = complex.find_roots().unwrap().resonances(sample_rate);
    // *count = res.len() as c_int;
    // let mut resonances = std::slice::from_raw_parts_mut(out, size);
    // for i in 0..res.len() {
    //     resonances[i] = res[i];
    // }
    // mem::forget(resonances);
}

#[no_mangle]
pub unsafe extern fn vox_box_make_raw_vec(raw_buffer: *mut f32, size: size_t) -> *const Vec<f32> {
    &Vec::<f32>::from_raw_parts(raw_buffer, size, size)
}

#[no_mangle]
pub unsafe extern fn vox_box_free(buffer: *mut [f32]) {
    drop(Box::from_raw(buffer));
}

#[no_mangle]
pub unsafe extern fn vox_box_print_f32(buffer: *mut f32) {
    println!("Printing buffer... {:?}", buffer);
}

#[cfg(test)]
mod tests {
    use super::*; 
    use num::complex::Complex64;
    use std::f64::consts::PI;
    use super::waves::*;

    #[test]
    fn test_resonances() {
        let roots = vec![Complex64::new( -0.5, 0.86602540378444 ), Complex64::new( -0.5, -0.86602540378444 )];
        let res = roots.to_resonance(300f64);
        println!("Resonances: {:?}", res);
        assert!((res[0].frequency - 100.0).abs() < 1e-8);
        assert!((res[0].amplitude - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_lpc() {
        let sine = Vec::<f64>::sine(8);
        let mut auto = sine.autocorrelate(8);
        // assert_eq!(maxima[3], (128, 1.0));
        auto.normalize();       
        let auto_exp = vec![1.0, 0.7071, 0.1250, -0.3536, -0.5, -0.3536, -0.1250, 0.0];
        // Rust output:
        let lpc_exp = vec![1.0, -1.3122, 0.8660, -0.0875, -0.0103];
        let lpc = auto.lpc(4);
        // println!("LPC coeffs: {:?}", &lpc);
        for (a, b) in auto.iter().zip(auto_exp.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
        for (a, b) in lpc.iter().zip(lpc_exp.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
    }

    #[test]
    fn test_ac() { 
        let sine = Vec::<f64>::sine(16);
        sine.autocorrelate(16);
    }

    #[test]
    fn test_formant_extractor() {
        let resonances = vec![
            vec![100.0, 150.0, 200.0, 240.0, 300.0], 
            vec![110.0, 180.0, 210.0, 230.0, 310.0],
            vec![230.0, 270.0, 290.0, 350.0, 360.0]
        ];
        let mut extractor = FormantExtractor::new( 3, &resonances, vec![140.0, 230.0, 320.0]);
        // First cycle has initial guesses
        match extractor.next() {
            Some(r) => { assert_eq!(r, vec![150.0, 240.0, 300.0]) },
            None => { panic!() }
        };

        // Second cycle should be different
        match extractor.next() {
            Some(r) => { assert_eq!(r, vec![180.0, 230.0, 310.0]) },
            None => { panic!() }
        };

        // Third cycle should have removed duplicates and shifted to fill all slots
        match extractor.next() {
            Some(r) => { assert_eq!(r, vec![230.0, 270.0, 290.0]) },
            None => { panic!() }
        };
    }

    use std::cmp::Ordering;

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
        for i in 0..4 { vector.extend(signal.iter().cloned()); }
        let mut sortable = vector.clone();
        sortable.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let global_peak = sortable[0];
        let local_peak = global_peak.clone();
        let mut auto = vector[..].autocorrelate(512);
        auto.normalize();
        let maxima = auto.local_maxima();
        assert_eq!(maxima.len(), 95);
        vector.window(Window::Hanning);
        let pitch = vector.pitch(512f64, 0.2f64, 0.05, local_peak, global_peak, 0.01, 10f64, 100f64, Window::Hanning);
        assert_eq!(pitch[0].frequency, 16f64);
    }

    #[test]
    fn test_rms() {
        let sine = Vec::<f64>::sine(64);
        let rms = sine.rms();
        println!("rms is {:?}", rms);
        assert!((rms - 0.707).abs() < 0.001);
    }
}

