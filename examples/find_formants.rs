use std::path::Path;
use std::fs::File;
use std::io::{Write, Cursor, Read};
use std::fmt;
use std::i32;
use std::net::{SocketAddr};

extern crate hound;
extern crate num;
extern crate vox_box;

extern crate rustc_serialize;
extern crate docopt;
extern crate byteorder;

use vox_box::spectrum::{FormantExtractor, LPC, ToResonance, Resonance};
use vox_box::polynomial::Polynomial;
use vox_box::periodic::{Pitch, HasPitch, Autocorrelate};
use vox_box::waves::{Resample, Filter, HasRMS, Normalize, Windower, WindowType};
use num::complex::Complex;

use docopt::Docopt;
use byteorder::{ReadBytesExt, WriteBytesExt, BigEndian, LittleEndian};

const USAGE: &'static str = "
Formant resonance extrator.

Usage:
    find_formants <file> [--bin=<size> --hop=<size> --socket=<addr> --threshold=<thresh> --formants --pitch]
    find_formants (-h | --help)
    find_formants --version

Options:
    --bin=<size>    Set bin size
    --hop=<size>    Set hop size
    --socket=<addr>     Set socket address
    --threshold=<thresh>    Set voiced threshold
    --formants      Analyze formants
    --pitch         Analyze pitch
    -h --help       Show this screen
    --version       Show version
";

const VERSION: &'static str = "0.1.0";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_file: String,
    flag_hop: usize,
    flag_bin: usize,
    flag_threshold: f64,
    flag_socket: String,
    flag_formants: bool,
    flag_pitch: bool
}

/// Run help for assistance with the args
/// Output format: 7-dimensional vector, all little-endian double precision floats
///
/// f0 frequency
/// f0 strength
/// f1 frequency
/// f2 frequency
/// f3 frequency
/// f4 frequency
/// rms
fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.help(true).version(Some(String::from(VERSION))).decode() )
        .unwrap_or_else(|e| e.exit());

    let hop_size: usize = if args.flag_hop == 0 { 128 } else { args.flag_hop };
    let bin_size: usize = if args.flag_bin == 0 { 512 } else { args.flag_bin };
    let voiced_threshold: f64 = if args.flag_threshold == 0f64 { 0.0 } else { args.flag_threshold };

    let file_path = &Path::new(&args.arg_file);
    let mut file = hound::WavReader::open(file_path).unwrap();
    let mut samples: Vec<f64> = file.samples::<i32>().map(|s| s.unwrap() as f64).collect();
    let global_max = samples.iter().max_by_key(|s| (*s * i32::MAX as f64) as i32).map(|s| *s as f64 / i32::MAX as f64).unwrap();
    samples.preemphasis(50.0 / 44100.0);
    let window = Windower::new(WindowType::Hanning, &samples[..], hop_size, bin_size);
    // println!("Splitting into {:?} bins", window.len());

    let frames: Vec<(Vec<Pitch<f64>>, Vec<Resonance<f64>>, f64)> = window.map(|data| {
        let rms = data.rms();
        let local_max = data.iter().max_by_key(|s| (*s * i32::MAX as f64) as i32).map(|s| *s as f64 / i32::MAX as f64).unwrap();
        let pitches: Vec<Pitch<f64>> = data.pitch(44100f64, voiced_threshold, 0.05, local_max, global_max, 0.01, 75f64, 150f64, WindowType::Hanning);

        let mut auto = data.resample_linear(bin_size / 4).autocorrelate(16);
        auto.normalize();
        let lpc: Vec<f64> = auto.lpc(14);
        let complex_coeffs: Vec<Complex<f64>> = lpc.iter().map(|x| Complex::<f64>::from(x)).collect();
        let resonances: Vec<Resonance<f64>> = complex_coeffs.find_roots().unwrap().to_resonance(11025f64);
        (pitches, resonances, rms)
    }).collect();

    if !args.flag_socket.is_empty() {
        let pipe_path = &Path::new(&args.flag_socket);
        let mut pipe_writer = File::create(pipe_path).unwrap();
        pipe_writer.write_i32::<LittleEndian>(frames.len() as i32);

        if args.flag_formants {
            let frames_formants = frames.iter().map(|f| f.1.clone()).collect();
            let frames_pitches: Vec<Vec<Pitch<f64>>> = frames.iter().map(|f| f.0.clone()).collect();
            let frames_rms: Vec<f64> = frames.iter().map(|f| f.2).collect();
            let formants: Vec<Vec<f64>> = FormantExtractor::<f64>::new(4, &frames_formants, vec![320f64, 1440.0, 2760.0, 3200.0]).collect();

            for frame_index in (0..frames.len()) {
                let pitch: Vec<Pitch<f64>> = frames_pitches[frame_index].clone();
                let formant: Vec<f64> = formants[frame_index].clone();
                let rms: f64 = frames_rms[frame_index];

                if pitch.len() > 0 {
                    pipe_writer.write_f64::<LittleEndian>(pitch[0].frequency);
                    pipe_writer.write_f64::<LittleEndian>(pitch[0].strength);
                }

                for i in (0..4) {
                    if i < formant.len() {
                        pipe_writer.write_f64::<LittleEndian>(formant[i]);
                    } else {
                        pipe_writer.write_f64::<LittleEndian>(0f64);
                    }
                }

                pipe_writer.write_f64::<LittleEndian>(rms);
            }
        }
    }
}
