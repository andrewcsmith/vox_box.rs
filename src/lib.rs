extern crate num;
extern crate rand;
extern crate sample;
extern crate rustfft;

// Declare local mods
pub mod complex;
pub mod periodic;
pub mod polynomial;
pub mod spectrum;
pub mod waves;
pub mod error;

use sample::Sample;
use sample::conv::Duplex;
use sample::window::Type;
use sample::interpolate::{Linear, Converter};

use spectrum::{LPC, Resonance, EstimateFormants};
use polynomial::Polynomial;
use error::*;

use num::{Float, Complex, FromPrimitive};

pub const MAX_RESONANCES: usize = 32;
pub const MALE_FORMANT_ESTIMATES: [f64; 4] = [320., 1440., 2760., 3200.];
pub const FEMALE_FORMANT_ESTIMATES: [f64; 4] = [480., 1760., 3200., 3520.];

pub fn find_formants_real_work_size(buf_len: usize, n_coeffs: usize) -> usize {
    buf_len * 2 + n_coeffs * 23 + 2
}

pub fn find_formants_complex_work_size(n_coeffs: usize) -> usize {
    n_coeffs * 7 + 4
}

/// Calculates the next frame of formants based on given estimates. The user must provide
/// sufficient workspace to carry out these calculations.
pub fn find_formants<S>(buf: &mut [S], sample_rate: S, resample_ratio: f64, resampled_buf: &mut [S], n_coeffs: usize, work: &mut [S], complex_work: &mut [Complex<S>], formants: &mut [Resonance<S>]) 
    -> VoxBoxResult<()> 
    where S: Sample + Duplex<f64> + Float + FromPrimitive
{
    let resampled_len = (resample_ratio * buf.len() as f64).ceil() as usize; 

    if work.len() < find_formants_real_work_size(resampled_len, n_coeffs) {
        return Err(VoxBoxError::Workspace);
    }

    // if complex_work.len() < find_formants_complex_work_size(n_coeffs) {
    //     return Err(VoxBoxError::Workspace);
    // }

    assert!(resampled_len <= resampled_buf.len());
    let mut resonances = [Resonance::new(0f64.to_sample::<S>(), 0f64.to_sample::<S>()); MAX_RESONANCES];
    let (mut lpc_coeffs, mut work) = work.split_at_mut(n_coeffs);
    if resample_ratio != 1.0 {
        let mut buf_iter = buf.iter().map(|b| [*b]);
        let linear = Linear::new(buf_iter.next().unwrap(), buf_iter.next());
        let sig = Converter::scale_sample_hz(buf_iter, linear, resample_ratio);
        for (r, s) in resampled_buf.iter_mut().zip(sig) { *r = s[0]; }
    } else {
        for (r, s) in resampled_buf.iter_mut().zip(buf.iter()) { *r = *s; }
    }

    let len_inv = 1f64 / resampled_len as f64;
    for (idx, s) in resampled_buf.iter_mut().enumerate() {
        let window = sample::window::Hanning::at_phase(S::from_sample(idx as f64 * len_inv));
        *s = *s * window;
    }

    let (mut lpc_work, work) = work.split_at_mut(resampled_buf.len() * 2 + n_coeffs);
    let (auto_coeffs, _) = work.split_at_mut(n_coeffs + 2);

    resampled_buf.lpc_praat_mut(n_coeffs, &mut lpc_coeffs, &mut lpc_work)?;
    let one = [1.0.to_sample::<S>()];

    let resonances = {
        let mut count: usize = 0;
        let (mut complex_lpc, mut complex_work) = complex_work.split_at_mut(n_coeffs + 1);

        {
            let rc = one.iter().chain(lpc_coeffs.iter()).rev().zip(complex_lpc.iter_mut());
            for (r, c) in rc {
                *c = Complex::<S>::from(r);
            }
        }
        
        complex_lpc.find_roots_mut(&mut complex_work)?;
        for root in complex_lpc.iter() {
            if root.im > 0.0.to_sample::<S>() {
                match Resonance::from_root(root, sample_rate) {
                    Some(res) => {
                        resonances[count] = res; 
                        count += 1; 
                    }
                    None => { }
                }
            }
        }
        let rpos = resonances.iter().rposition(|v| {
            v.frequency != 0f64.to_sample::<S>()
        }).unwrap_or(0);
        resonances[0..(rpos+1)].sort_by(|a, b| { 
            a.frequency.partial_cmp(&b.frequency).unwrap_or(std::cmp::Ordering::Less)
        });
        resonances
    };

    formants.estimate_formants(&resonances);
    Ok(())
}

