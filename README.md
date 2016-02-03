# vox_box.rs

A tool to let you hack away at voice audio recordings in Rust.

## How do I do it?

[Documentation](http://www.andrewchristophersmith.com/docs/vox_box/vox_box/index.html)

```rust
extern crate hound;
use std::path::Path;

// Read in some audio file
let file_path = &Path(&path_to_file);
let audio = hound::WavReader::open(&file_path).unwrap();
// Copy the samples to an f64 buffer
let mut samples: Vec<f64> = audio.samples::<i32>().map(|v| *v as f64).collect();
// Give a 6db/oct boost at 50 hz and above
samples.preemphasis(50.0 / 44100.0); 
// Hanning window iterator, with a hop of 256 and bin of 512
let mut window = Windower::new(WindowType::Hanning, &samples[..], 256, 512);
// For each window, get 26 MFCC coefficients between 100 and 8000 Hz
windows.map(|frame: Vec<f64>| frame.mfcc(26, (100., 8000.), 44_100.)).collect();
```

* MFCC calculation: [examples/cosine_sim.rs](https://github.com/andrewcsmith/vox_box.rs/blob/master/examples/cosine_sim.rs)
* Formant calculation: [examples/find_formants.rs](https://github.com/andrewcsmith/vox_box.rs/blob/master/examples/find_formants.rs)

## What's included

The following taxonomy is more for my reorganization work than anything else.

* mod waves
    * Resampler
    * Sine and Saw wave generators
    * Windower (Hanning / Hamming)
    * Filter preemphasis
    * Normalization
    * Max value finding
* mod periodic
    * Autocorrelation calculation
    * Pitch path finder (Boersma tracker, from Praat)
* mod spectrum
    * Mel-Frequency Cepstral Coefficient (MFCC) calculation
    * Linear Predictive Coding (LPC) coefficient calculation
    * Resonance calculation
    * Formant path finder (McCandless algorithm, from Praat)
* mod polynomial
    * Polynomial division
    * Laguerre root finding

## Why is it broken?

Open an issue! There are many incomplete aspects to this library, as it is highly specialized to my personal projects. Once others begin using it, I expect that it will become more generalized.

