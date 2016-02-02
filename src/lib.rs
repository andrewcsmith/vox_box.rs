#![feature(plugin, test)]

extern crate num;
extern crate rand;
extern crate sample;
extern crate libc;

use num::complex::Complex;
use num::{Float, FromPrimitive};
use std::iter::Take;
use sample::Sample;
use libc::{size_t, c_int, c_void};
use std::mem;

// Declare local mods
pub mod polynomial;
pub mod periodic;
pub mod waves;
pub mod spectrum;

// Use std
use std::iter::Iterator;
use std::f64::consts::PI;
use std::ops::*;
use std::collections::VecDeque;
use std::cmp::Ordering::{Less, Equal, Greater};
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::marker::Sized;

use waves::*;

use polynomial::Polynomial;
use periodic::*;
use spectrum::*;

/// Calculates autocorrelation and will allocate memory
///
/// const float* input: input buffer to calculate from
/// size_t size:        size of input buffer
/// size_t n_coeffs:    number of coefficients to calculate
///
/// Returns a pointer to *float of length n_coeffs
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
    let complex: Vec<Complex<f32>> = buf.iter().map(|v| Complex::<f32>::from(v)).collect(); 
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
        complex[i] = Complex::<f32>::from(&buf[i]);
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

