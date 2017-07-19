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
fn test_against_praat() {
    let mut reader = WavReader::open("./tests/down_sampled.wav").unwrap();
    let bits = reader.spec().bits_per_sample;
    let mut samples: Vec<f64> = reader.samples::<i32>().map(|sample| {
        sample.unwrap() as f64 / (i32::MAX >> (32 - bits)) as f64
    }).collect();

    let start = 21735;
    let end = 22011;
    let n_coeffs = 13;

    // subtract the mean of the segment
    let mean = samples[start..end].iter().fold(0., |a, s| a + s) / (end - start) as f64;
    let segment: Vec<f64> = samples[start..end].iter().map(|s| s - mean).collect();

    let sample_rate = reader.spec().sample_rate as f64;
    let resample_ratio = 1.0;
    let resampled_len = (resample_ratio * samples.len() as f64).ceil() as usize;
    let mut resampled_buf = vec![0f64; resampled_len];
    let mut work = vec![0f64; vox_box::find_formants_real_work_size(resampled_len, n_coeffs)];
    let mut complex_work = vec![Complex::new(0f64, 0.); vox_box::find_formants_complex_work_size(n_coeffs)];

    let mut formants: Vec<Resonance<f64>> = vox_box::MALE_FORMANT_ESTIMATES.iter().map(|f| Resonance::new(*f, 1.0)).collect();
    vox_box::find_formants(&mut samples[..], sample_rate,
                           resample_ratio, &mut resampled_buf[..],
                           n_coeffs, &mut work[..], &mut complex_work[..],
                           &mut formants[..]).unwrap();
    println!("formants: {:?}", formants);
}

#[test]
fn test_formant_calculation() {
    let mut reader = WavReader::open("./tests/short_sample.wav").unwrap();
    let bits = reader.spec().bits_per_sample;
    let mut samples: Vec<f64> = reader.samples::<i32>().map(|sample| {
        sample.unwrap() as f64 / (i32::MAX >> (32 - bits)) as f64
    }).collect();

    let n_coeffs = 10;

    let resample_ratio = 1.0;

    let bin = 1024;
    let hop = 512;
    let sample_rate = reader.spec().sample_rate as f64;
    let resampled_len = (samples.len() as f64 * resample_ratio).ceil() as usize;
    let mut formants: Vec<Resonance<f64>> = vox_box::MALE_FORMANT_ESTIMATES.iter().map(|f| Resonance::new(*f, 1.0)).collect();
    let mut all_formants: Vec<Vec<Resonance<f64>>> = Vec::new();
    let mut frame_buffer: Vec<f64> = Vec::with_capacity(bin);
    let mut powers: Vec<f64> = Vec::new();

    let sample_frames: &[[f64; 1]] = sample::slice::to_frame_slice(&samples[..]).unwrap();
    let mut resampled_buf = vec![0f64; resampled_len];

    let mut work = vec![0f64; vox_box::find_formants_real_work_size(resampled_len, n_coeffs)];
    let mut complex_work = vec![Complex::new(0f64, 0.); vox_box::find_formants_complex_work_size(n_coeffs)];

    for frame in window::Windower::rectangle(sample_frames, bin, hop) {
        for s in frame.take(bin) { 
            frame_buffer.push(s[0]); 
        }
        vox_box::find_formants(&mut frame_buffer[..], sample_rate, 
                               resample_ratio, &mut resampled_buf[..],
                               n_coeffs, &mut work[..], &mut complex_work[..],
                               &mut formants[..]).unwrap();
        all_formants.push(formants.clone());
        let rms: f64 = (frame_buffer.iter().fold(0., |acc, v| acc + v.powi(2)) / bin as f64).sqrt();
        powers.push(rms);
        frame_buffer.clear();
    }

    for ((idx, frame), rms) in all_formants.iter().enumerate().zip(powers.iter()) {
        // let freqs: Vec<f64> = frame.iter().map(|r| r.frequency).collect();
        println!("{:?}", frame);
        // assert!((frame[0].frequency - 662.0).abs() < 10.0);
    }
}
