extern crate hound;
extern crate sample;
extern crate vox_box;
extern crate num;
extern crate rustfft;

use std::error::Error;
use std::i32;

use hound::WavReader;
use vox_box::spectrum::Resonance;
use vox_box::periodic::{Hanning, Pitched};
use vox_box::waves::Filter;
use sample::{window, ToSampleSliceMut, ToSampleSlice};
use sample::signal::Signal;
use sample::interpolate::{Sinc, Converter, Linear};
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

    let (sample_rate, bit_depth) = {
        (reader.spec().sample_rate as f64, reader.spec().bits_per_sample)
    };

    let mut samples = reader.samples::<i32>().map(|sample| {
        [sample.unwrap().clone() as f64 / (i32::MAX << (32 - bit_depth)) as f64]
    });

    let new_sample_rate = 10000.0;
    let sinc = Sinc::new(50, &mut samples);
    let sig = samples.from_hz_to_hz(sinc, sample_rate, new_sample_rate);
    let len_upper_bound = sig.size_hint().1.unwrap();

    let n_coeffs = 13;
    let bin = (new_sample_rate * 0.05).ceil() as usize;
    let hop = (new_sample_rate * 0.01).ceil() as usize;

    println!("# bin: {}, hop: {}", bin, hop);

    let mut work = vec![0f64; vox_box::find_formants_real_work_size(len_upper_bound, n_coeffs)];
    let mut complex_work = vec![Complex::new(0f64, 0.); vox_box::find_formants_complex_work_size(n_coeffs)];

    let mut formants: Vec<Resonance<f64>> = vox_box::MALE_FORMANT_ESTIMATES.iter().map(|f| Resonance::new(*f, 1.0)).collect();
    let mut all_formants: Vec<Vec<Resonance<f64>>> = Vec::new();
    let mut frame_buffer: Vec<f64> = Vec::with_capacity(bin);
    let mut powers: Vec<f64> = Vec::new();
    let mut pitches: Vec<f64> = Vec::new();

    let mut frames: Vec<[f64; 1]> = sig.collect();

    for frame in window::Windower::hanning(&frames[..], bin, hop) {
        for s in frame { 
            frame_buffer.push(s[0]); 
        }
        let pitch: f64 = frame_buffer.to_sample_slice().pitch::<Hanning>(new_sample_rate, 0.2, 0.05, 1.0, 1.0, 1.0, 50., 200.)[0].frequency;

        vox_box::find_formants(&mut frame_buffer[..], new_sample_rate, 
                               n_coeffs, &mut work[..], &mut complex_work[..],
                               &mut formants[..]).unwrap();
        all_formants.push(formants.clone());
        let rms: f64 = (frame_buffer.iter().fold(0., |acc, v| acc + v.powi(2)) / bin as f64).sqrt();
        powers.push(rms);
        pitches.push(pitch);
        frame_buffer.clear();
    }

    for ((idx, frame), (rms, pitch)) in all_formants.iter().enumerate().zip(powers.iter().zip(pitches.iter())) {
        print!("{:?} ", (idx * hop) as f64 / sample_rate as f64);

        for res in frame.iter().take(4) {
            print!("{:?} {:?} ", res.frequency, res.bandwidth);
        }

        println!("{:?} {:?}", rms, pitch);
    }

    Ok(())
}

fn main() {
    go().unwrap();
}
