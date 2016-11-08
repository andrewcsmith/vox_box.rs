extern crate hound;
extern crate sample;
extern crate vox_box;
extern crate num;
extern crate rustfft;

use std::error::Error;
use std::{i32, i16};

use hound::WavReader;
use vox_box::spectrum::Resonance;
use sample::window;
use sample::interpolate::{Sinc, Converter};
use num::Complex;

/// Prints the time stamp, then RMS, followed by the center frequency and bandwidth of 5 formants
///
/// Call the following to see the output plotted as a figure (requires gnuplot)
///
/// ```
/// cargo run > output.txt
/// gnuplot
/// gnuplot> set log y2 2
/// gnuplot> plot 'output.txt' using 1:10 with lines,\
///               '' using 1:2 with lines axes x1y2, \
///               '' using 1:4 with lines axes x1y2, \
///               '' using 1:6 with lines axes x1y2, \
///               '' using 1:8 with lines axes x1y2, \
///               '' using 1:10 with lines axes x1y2
/// ```
fn go() -> Result<(), Box<Error>> {
    let mut reader = try!(WavReader::open("./sample-two_vowels.wav"));
    let samples: Vec<[f64; 1]> = reader.samples::<i32>().map(|sample| {
        [sample.unwrap() as f64 / (i32::MAX >> 8) as f64]
    }).collect();

    let sample_rate = reader.spec().sample_rate as f64;
    let new_sample_rate = 10000.0;
    let sinc = Sinc::new(50, samples.iter().cloned());
    let conv = Converter::scale_sample_hz(samples.iter().cloned(), sinc, new_sample_rate / sample_rate);
    let resampled: Vec<[f64; 1]> = conv.collect();

    let n_coeffs = 13;
    let bin = (new_sample_rate * 0.05).ceil() as usize;
    let hop = (new_sample_rate * 0.01).ceil() as usize;

    println!("# bin: {}, hop: {}", bin, hop);

    let mut work = vec![0f64; vox_box::find_formants_real_work_size(resampled.len(), n_coeffs)];
    let mut complex_work = vec![Complex::new(0f64, 0.); vox_box::find_formants_complex_work_size(n_coeffs)];

    let mut formants: Vec<Resonance<f64>> = vox_box::MALE_FORMANT_ESTIMATES.iter().map(|f| Resonance::new(*f, 1.0)).collect();
    let mut all_formants: Vec<Vec<Resonance<f64>>> = Vec::new();
    let mut frame_buffer: Vec<f64> = Vec::with_capacity(bin);
    let mut powers: Vec<f64> = Vec::new();

    let sample_frames: &[[f64; 1]] = sample::slice::to_frame_slice(&resampled[..]).unwrap();

    for frame in window::Windower::rectangle(sample_frames, bin, hop) {
        for s in frame { 
            frame_buffer.push(s[0]); 
        }
        vox_box::find_formants(&mut frame_buffer[..], new_sample_rate, 
                               n_coeffs, &mut work[..], &mut complex_work[..],
                               &mut formants[..]).unwrap();
        all_formants.push(formants.clone());
        let rms: f64 = (frame_buffer.iter().fold(0., |acc, v| acc + v.powi(2)) / bin as f64).sqrt();
        powers.push(rms);
        frame_buffer.clear();
    }

    for ((idx, frame), rms) in all_formants.iter().enumerate().zip(powers.iter()) {
        print!("{:?} ", (idx * hop) as f64 / sample_rate as f64);
        for res in frame {
            print!("{:?} {:?} ", res.frequency, res.bandwidth);
        }
        println!("{:?}", rms);
    }

    Ok(())
}

fn main() {
    go().unwrap();
}
