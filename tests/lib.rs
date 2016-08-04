extern crate num;
extern crate hound;
extern crate sample;
extern crate vox_box;

use hound::WavReader;
use vox_box::waves::*;
use vox_box::spectrum::Resonance;
use sample::{window, ToFrameSlice, ToSampleSlice};
use num::Complex;
use std::i32;

#[test]
fn test_formant_calculation() {
    let mut reader = WavReader::open("./tests/short_sample.wav").unwrap();
    let bits = reader.spec().bits_per_sample;
    let samples: Vec<f64> = reader.samples::<i32>().map(|sample| {
        sample.unwrap() as f64 / (i32::MAX >> (32 - bits)) as f64
    }).collect();

    let resample_factor = 2.0;
    let n_coeffs = 26;

    let bin = 2048;
    let hop = 1024;
    let sample_rate = reader.spec().sample_rate as f64;
    let resampled_len = (bin as f64 / resample_factor) as usize;
    let mut work = vec![0f64; vox_box::find_formants_real_work_size(resampled_len, n_coeffs)];
    let mut complex_work = vec![Complex::new(0f64, 0.); vox_box::find_formants_complex_work_size(n_coeffs)];

    let mut formants: Vec<Resonance<f64>> = vox_box::MALE_FORMANT_ESTIMATES.iter().map(|f| Resonance::new(*f, 1.0)).collect();
    let mut all_formants: Vec<Vec<Resonance<f64>>> = Vec::new();
    let mut frame_buffer: Vec<f64> = Vec::with_capacity(bin);
    let mut powers: Vec<f64> = Vec::new();

    let sample_frames: &[[f64; 1]] = sample::slice::to_frame_slice(&samples[..]).unwrap();

    for frame in window::Windower::hanning(sample_frames, bin, hop) {
        for s in frame { 
            frame_buffer.push(s[0]); 
        }
        vox_box::find_formants(&frame_buffer[..], resample_factor, sample_rate, 
                               n_coeffs, &mut work[..], &mut complex_work[..],
                               &mut formants[..]);
        all_formants.push(formants.clone());
        let rms: f64 = (frame_buffer.iter().fold(0., |acc, v| acc + v.powi(2)) / bin as f64).sqrt();
        powers.push(rms);
        frame_buffer.clear();
    }

    for ((idx, frame), rms) in all_formants.iter().enumerate().zip(powers.iter()) {
        println!("{:?}", frame);
        assert!((frame[0].frequency - 662.0).abs() < 10.0);
    }
}
