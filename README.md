[![Build Status](https://travis-ci.org/andrewcsmith/vox_box.rs.svg?branch=master)](https://travis-ci.org/andrewcsmith/vox_box.rs)

# vox_box.rs 

A tool to let you hack away at voice audio recordings in Rust.

## How do I do it?

[Documentation](https://docs.rs/vox_box/0.3.0/vox_box/)

## What's included

* Filter preemphasis, normalization, RMS calculation
* Autocorrleation calculation
* Laguerre root finding and polynomial division
* Mel-Frequency Cepstral Coefficient (MFCC) calculation
* Linear Predictive Coding (LPC) coefficient calculation
* Formant path finder (McCandless algorithm, from Praat)
* Pitch finding (Boersma autocorrelation method, from Praat)

## Why is it broken?

Open an issue! There are many incomplete aspects to this library, as it is highly specialized to my personal projects. Once others begin using it, I expect that it will become more generalized.

