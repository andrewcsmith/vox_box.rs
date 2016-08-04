extern crate num;
extern crate rustfft as fft;

use std::f64::consts::PI;
use std::ops::{Index, IndexMut};
use std::default::Default;
use std::marker::PhantomData;
use num::{Complex, Float, ToPrimitive, FromPrimitive};
use num::traits::{Zero, Signed};
use std::fmt::Debug;
use std::cmp::Ordering;

pub struct LPCSolver<'a, T: 'a> {
    n_coeffs: usize,
    ac: &'a mut [T],
    kc: &'a mut [T],
    tmp: &'a mut [T]
}

impl<'a, T: 'a + Float> LPCSolver<'a, T> {
    /// Constructs an LPCSolver without any allocations required.
    ///
    /// work must be at least length `n_coeffs * 3 + 1`.
    pub fn new(n_coeffs: usize, work: &'a mut [T]) -> LPCSolver<'a, T> {
        assert!(work.len() > n_coeffs * 3 + 1);

        let (ac, mut work) = work.split_at_mut(n_coeffs + 1);
        let (kc, tmp) = work.split_at_mut(n_coeffs);

        LPCSolver {
            n_coeffs: n_coeffs,
            ac: ac,
            kc: kc,
            tmp: tmp
        }
    }

    /// Finds the LPC coefficients for the autocorrelated buffer
    pub fn solve(&mut self, buf: &[T]) {
        buf.lpc_mut(self.n_coeffs, self.ac, self.kc, self.tmp);
    }

    /// Returns the slice of LPC coefficients
    pub fn lpc(&self) -> &[T] {
        &self.ac[..]
    }
}

pub trait LPC<T> {
    fn lpc_mut(&self, n_coeffs: usize, ac: &mut [T], kc: &mut [T], tmp: &mut [T]);
    fn lpc(&self, n_coeffs: usize) -> Vec<T>;
}

impl<V: ?Sized, T> LPC<T> for V where 
    T: Float,
    V: Index<usize, Output=T>
{ 
    /// Calculates the LPCs, reusing slices of memory for workspace.
    ///
    /// ac: size must be at least `n_coeffs + 1`
    /// kc: size must be at least `n_coeffs`
    /// tmp: size must be at least `n_coeffs`
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Resonance<T> {
    pub frequency: T,
    pub bandwidth: T
}

impl<T> Resonance<T> {
    pub fn new(f: T, b: T) -> Resonance<T> {
        Resonance {
            frequency: f,
            bandwidth: b
        }
    }
}

impl<T: Float + FromPrimitive> Resonance<T> {
    pub fn from_root(root: &Complex<T>, sample_rate: T) -> Option<Resonance<T>> {
        let freq_mul: T = T::from_f64(sample_rate.to_f64().unwrap() / (PI * 2f64)).unwrap();
        let samp_interval: T = T::from(1.).unwrap() / sample_rate;
        if root.im >= T::zero() {
            let (_, theta) = root.to_polar();
            let r2 = root.norm_sqr();
            let res = Resonance::<T> { 
                frequency: freq_mul * theta,
                bandwidth: freq_mul * r2.ln()
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

impl<T> ToResonance<T> for [Complex<T>] 
    where T: Float + 
             FromPrimitive 
{
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

pub trait EstimateFormants<T> {
    type FormantSlots;
    fn estimate_formants(&mut self, resonances: &[Resonance<T>]);
}

fn diff_func<T: Float>(a: T, b: &T) -> T {
    (a - *b).abs()
}

impl<T: Float> EstimateFormants<T> for [Resonance<T>] {
    /// Let's cap things at 6 formants. Give me a ring if you need extra and I can get my guy to
    /// get a few more.
    type FormantSlots = [Option<Resonance<T>>; 6];

    /// Assumes that [self] is a sequence of Resonances corresponding to either the previous
    /// formant frame or the estimated formants for the next frame.
    fn estimate_formants(&mut self, resonances: &[Resonance<T>]) {
        let mut slots = Self::FormantSlots::default();
        // Step 2: Get the nearest resonance index for each estimated value
        for (estimate, slot) in self.iter().zip(slots.iter_mut()) {
            let start = (resonances[0], diff_func(resonances[0].frequency, &estimate.frequency));
            *slot = Some(resonances.iter().skip(1).fold(start, |acc, item| {
                let distance = diff_func(item.frequency, &estimate.frequency);
                if distance < acc.1 {
                    (*item, distance)
                } else {
                    acc
                }
            }).0)
        }

        // Step 3: Remove duplicates. If the same peak p_j fills more than one slots S_i keep it
        // only in the slot S_k which corresponds to the estimate EST_k that it is closest to in
        // frequency, and remove it from any other slots.
        let mut w = 0usize;
        let mut has_unassigned = false;

        for r in 1..slots.len() {
            match slots[r] {
                Some(v) => { 
                    // If this resonance is the same as the previous one...
                    if v == slots[w].unwrap() {
                        if diff_func(v.frequency, &self[r].frequency) < diff_func(v.frequency, &self[w].frequency) {
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
            for j in 0..resonances.len() {
                let peak = Some(resonances[j]);
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

        slots.sort_by(|a, b| { 
            match *a {
                Some(a_real) => {
                    match *b {
                        Some(b_real) => {
                            a_real.frequency.partial_cmp(&b_real.frequency).unwrap_or(Ordering::Equal)
                        },
                        None => { Ordering::Greater }
                    }
                }
                None => { Ordering::Less }
            }
        });

        // Update the current slice with the new formants that have been decided upon
        for (winner, estimate) in slots.iter()
            .filter_map(|v| *v).filter(|v| v.frequency > T::zero())
            .zip(self.iter_mut()) 
        {
            *estimate = winner;
        }
    }
}

pub struct FormantExtractor<'a, T: 'a + Float, I: Iterator<Item=&'a [Resonance<T>]>> {
    pub estimates: Vec<Resonance<T>>,
    num_formants: usize,
    resonances: I,
    phantom: PhantomData<&'a T>
}

impl<'a, T, I> FormantExtractor<'a, T, I> 
    where T: 'a + Float + PartialEq,
          I: Iterator<Item=&'a [Resonance<T>]>
{
    pub fn new(num_formants: usize, resonances: I, starting_estimates: Vec<Resonance<T>>) -> Self {
        FormantExtractor { 
            num_formants: num_formants, 
            resonances: resonances, 
            estimates: starting_estimates,
            phantom: PhantomData
        }
    }
}

impl<'a, T, I> Iterator for FormantExtractor<'a, T, I> 
    where T: 'a + Float + PartialEq,
          I: Iterator<Item=&'a [Resonance<T>]>
{
    type Item = Vec<Resonance<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = self.resonances.next();
        if frame.is_none() { return None; }
        &self.estimates[..].estimate_formants(frame.unwrap());
        Some(self.estimates.clone())
    }
}

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
    signal.iter().enumerate().map(|(k, _)| {
        T::from_f64(2. * (0..signal.len()).fold(0., |acc, n| {
            acc + signal[n].to_f64().unwrap() * (PI * k as f64 * (2. * n as f64 + 1.) / (2. * signal.len() as f64)).cos()
        })).unwrap()
    }).collect()
}

/// MFCC assumes that it is a windowed signal
impl<T: ?Sized> MFCC<T> for [T] 
    where T: Debug + 
             Float + 
             ToPrimitive + 
             FromPrimitive + 
             Into<Complex<T>> + 
             Zero + 
             Signed
{
    fn mfcc(&self, num_coeffs: usize, freq_bounds: (f64, f64), sample_rate: f64) -> Vec<T> {
        let mel_range = hz_to_mel(freq_bounds.1) - hz_to_mel(freq_bounds.0);
        // Still an iterator
        let points = (0..(num_coeffs + 2)).map(|i| (i as f64 / num_coeffs as f64) * mel_range + hz_to_mel(freq_bounds.0));
        let bins: Vec<usize> = points.map(|point| ((self.len() + 1) as f64 * mel_to_hz(point) / sample_rate).floor() as usize).collect();

        let mut spectrum = vec![Complex::<T>::from(T::zero()); self.len()];
        let mut fft = fft::FFT::new(self.len(), false);
        let signal: Vec<Complex<T>> = self.iter().map(|e| Complex::<T>::from(e)).collect();
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
    extern crate sample;
    extern crate rand;

    use super::*;
    use rand::{thread_rng, Rng};
    use waves::*;
    use periodic::*;
    use sample::{window, ToSampleSlice};
    use num::Complex;

    fn sine(len: usize) -> Vec<f64> {
        let rate = sample::signal::rate(len as f64).const_hz(1.0);
        rate.clone().sine().take(len).collect::<Vec<[f64; 1]>>().to_sample_slice().to_vec()
    }

    #[test]
    fn test_resonances() {
        let roots = vec![Complex::<f64>::new( -0.5, 0.86602540378444 ), Complex::<f64>::new( -0.5, -0.86602540378444 )];
        let res = roots.to_resonance(300f64);
        println!("Resonances: {:?}", res);
        assert!((res[0].frequency - 100.0).abs() < 1e-8);
        assert!((res[0].bandwidth - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_lpc() {
        let sine = sine(8);
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
    fn test_formant_extractor() {
        let resonances: Vec<Vec<Resonance<f64>>> = vec![
            vec![100.0, 150.0, 200.0, 240.0, 300.0], 
            vec![110.0, 180.0, 210.0, 230.0, 310.0],
            vec![230.0, 270.0, 290.0, 350.0, 360.0]
        ].iter().map(|z| z.iter().map(|r| Resonance::<f64> { frequency: *r, bandwidth: 1. }).collect()).collect();
        let estimates = vec![140., 230., 320.].iter().map(|r| Resonance::<f64> { frequency: *r, bandwidth: 1. }).collect();

        let mut extractor = FormantExtractor::new(3, resonances.iter().map(|r| &r[..]), estimates);

        // First cycle has initial guesses
        match extractor.next() {
            Some(r) => { 
                let freqs: Vec<f64> = r.iter().map(|f| f.frequency).collect();
                // Post-step-3 should be: 150, 240, 300 
                assert_eq!(freqs, vec![150.0, 240.0, 300.0]) 
            },
            None => { panic!() }
        }

        // Second cycle should be different
        match extractor.next() {
            Some(r) => { 
                let freqs: Vec<f64> = r.iter().map(|f| f.frequency).collect();
                // Post-step-3 should be: 180, 230, 310
                assert_eq!(freqs, vec![180.0, 230.0, 310.0]) 
            },
            None => { panic!() }
        }

        // Third cycle should have removed duplicates and shifted to fill all slots
        match extractor.next() {
            Some(r) => { 
                let freqs: Vec<f64> = r.iter().map(|f| f.frequency).collect();
                // Post-step-3 should be: None, 230, 290
                assert_eq!(freqs, vec![230.0, 270.0, 290.0]) 
            },
            None => { panic!() }
        }
    }

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
        let mut vec: Vec<f64> = (0..256).map(|_| rng.gen_range::<f64>(-1., 1.)).collect();
        vec.preemphasis(0.1f64 * 22_050.);
        let hanning_window: Vec<[f64; 1]> = window::hanning(256).take(256).collect();
        for (v, w) in vec.iter_mut().zip(hanning_window.to_sample_slice().iter()) {
            *v *= *w;
        }
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
