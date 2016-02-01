# vox_box.rs

A tool to let you hack away at voice audio recordings in Rust.

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

Open an issue!
