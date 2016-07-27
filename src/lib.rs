extern crate num;
extern crate rand;
extern crate sample;

// Declare local mods
pub mod complex;
pub mod periodic;
pub mod polynomial;
pub mod spectrum;
pub mod waves;

use sample::Sample;
use sample::conv::Duplex;
use sample::rate::Converter;
use sample::slice::to_frame_slice;

use spectrum::{LPCSolver, Resonance, EstimateFormants};
use polynomial::Polynomial;
use waves::Normalize;
use periodic::Autocorrelate;

use num::{Float, Complex, FromPrimitive};

const MAX_RESONANCES: usize = 32;
const MALE_FORMANT_ESTIMATES: [f64; 4] = [320., 1440., 2760., 3200.];
const FEMALE_FORMANT_ESTIMATES: [f64; 4] = [480., 1760., 3200., 3520.];

pub fn find_formants_real_work_size<S>(resampled_len: usize, n_coeffs: usize) 
    -> usize 
    where S: Sample + Duplex<f64> + Float
{
    resampled_len + n_coeffs * 22 + 2
}

pub fn find_formants_complex_work_size(n_coeffs: usize) -> usize {
    n_coeffs * 7 + 4
}

/// Calculates the next frame of formants based on given estimates. The user must provide
/// sufficient workspace to carry out these calculations.
pub fn find_formants<S>(buf: &[S], resample_factor: S, sample_rate: S, n_coeffs: usize, work: &mut [S], complex_work: &mut [Complex<S>], formants: &mut [Resonance<S>]) 
    -> Result<(), &'static str> 
    where S: Sample + Duplex<f64> + Float + FromPrimitive
{
    let factor = resample_factor.to_f64().unwrap();
    let resampled_len = (buf.len() as f64 / factor).ceil() as usize;
    let new_sample_rate = sample_rate * resample_factor;
    if work.len() < resampled_len + n_coeffs * 22 + 2 {
        return Err("Not enough workspace allocated");
    }

    let mut resonances = [Resonance::new(0f64.to_sample::<S>(), 0f64.to_sample::<S>()); MAX_RESONANCES];
    let (mut resampled, mut work) = work.split_at_mut(resampled_len);
    let (mut lpc_coeffs, mut work) = work.split_at_mut(n_coeffs);
    let (mut auto_coeffs, mut work) = work.split_at_mut(n_coeffs + 2);
    // Final work slice is used for calculations

    let frame_slice: &[[S; 1]] = to_frame_slice(buf).unwrap();
    let resampler = Converter::scale_sample_hz(frame_slice.iter().cloned(), factor);
    for (idx, s) in resampler.enumerate() {
        resampled[idx] = s[0];
    }

    resampled.autocorrelate_mut(&mut auto_coeffs[..]);
    let auto_coeffs_max = auto_coeffs[0];
    auto_coeffs.normalize_with_max(Some(auto_coeffs_max));

    {
        // Use the workspace for the given scope only
        let mut solver = LPCSolver::new(n_coeffs, &mut work);
        solver.solve(auto_coeffs);
        for (c, d) in solver.lpc().iter().zip(lpc_coeffs.iter_mut()) {
            *d = *c;
        }
    }

    let resonances = {
        let mut count: usize = 0;  
        let (mut complex_lpc, mut complex_work) = complex_work.split_at_mut(n_coeffs);
        for (real, complex) in lpc_coeffs.iter().zip(complex_lpc.iter_mut()) {
            *complex = Complex::<S>::from(real);
        }
        complex_lpc.find_roots_mut(&mut complex_work).expect("Problem finding roots");
        for root in complex_lpc.iter() {
            match Resonance::from_root(root, new_sample_rate) {
                Some(res) => {
                    resonances[count] = res; 
                    count += 1; 
                }
                None => { }
            }
        }
        let rpos = resonances.iter().rposition(|v| v.frequency != 0f64.to_sample::<S>()).unwrap();
        resonances[0..(rpos+1)].sort_by(|a, b| { 
            a.frequency.partial_cmp(&b.frequency).unwrap()
        });
        resonances
    };

    formants.estimate_formants(&resonances);
    Ok(())
}

