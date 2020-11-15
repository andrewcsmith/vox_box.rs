#![cfg_attr(feature = "valgrind", feature(alloc_system))]

#[cfg(feature = "valgrind")]
extern crate alloc_system;

extern crate dasp;
extern crate vox_box;
extern crate num;

use vox_box::periodic::*;
use vox_box::waves::*;

use sample::{window, Signal, ToSampleSlice};

fn get_pitch() -> Result<(), ()> {
    let exp_freq = 150.0;
    let signal = sample::signal::rate(44100.).const_hz(exp_freq).sine();
    let vector: Vec<[f64; 1]> = signal.take(2048 * 1 + 1).collect();
    let maxima: f64 = vector.to_sample_slice().max_amplitude();
    let mut pitches_out: Vec<Vec<Pitch<f64>>> = Vec::new();

    let mut chunk_data: Vec<f64> = Vec::with_capacity(2048);
    for chunk in window::Windower::hanning(&vector[..], 2048, 1024) {
        for d in chunk.take(2048) {
            chunk_data.push(d[0]);
        }
        let local_maxima = chunk_data.to_sample_slice().max_amplitude();
        pitches_out.push(analyze_pitch(&chunk_data[..], maxima, local_maxima)?);
        chunk_data.clear();
    }
    println!("pitches_out: {:?}", pitches_out);
    Ok(())
}

/// Valgrind:
///
/// Without pitch analysis: 
///   total heap usage: 272 allocs, 81 frees, 28,805 bytes allocated
///
/// With pitch analysis:
///   total heap usage: 276 allocs, 82 frees, 29,053 bytes allocated
///
fn main() {
    get_pitch().unwrap();
}

fn analyze_pitch(chunk_data: &[f64], maxima: f64, local_maxima: f64) -> Result<Vec<Pitch<f64>>, ()> {
    Ok(chunk_data.pitch::<window::Hanning>(44100., 0.2, local_maxima, maxima, 100., 500.))
}

