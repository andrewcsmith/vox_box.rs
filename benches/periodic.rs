#![cfg(feature = "nightly")]
#![feature(test)]

extern crate dasp;
extern crate test;
extern crate vox_box;
extern crate num;

use vox_box::periodic::*;
use vox_box::waves::*;

use sample::{window, ToSampleSlice};
use sample::signal::Sine;
use std::cmp::Ordering;
use std::f64::consts::PI;

fn sine(len: usize) -> Vec<f64> {
    let rate = sample::signal::rate(len as f64).const_hz(1.0);
    rate.clone().sine().take(len).collect::<Vec<[f64; 1]>>().to_sample_slice().to_vec()
}

#[bench]
/// Currently gives results:
///
/// test bench_pitch ... bench:  13,197,760 ns/iter (+/- 2,671,434)
fn bench_pitch(b: &mut test::Bencher) {
    let exp_freq = 150.0;
    let mut signal = sample::signal::rate(44100.).const_hz(exp_freq).sine();
    let vector: Vec<[f64; 1]> = signal.take(4096 + 1).collect();
    let mut maxima: f64 = vector.to_sample_slice().max_amplitude();

    let mut chunk_data: Vec<f64> = Vec::with_capacity(4096);

    b.iter(|| {
        for chunk in window::Windower::hanning(&vector[..], 4096, 1024) {
            for d in chunk.take(4096) {
                chunk_data.push(d[0]);
            }
            let pitch = chunk_data.pitch::<window::Hanning>(44100., 0.2, 0.05, maxima, maxima, 0.01, 100., 500.);
            chunk_data.clear();
        }
    });
}
