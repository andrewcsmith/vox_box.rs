extern crate nalgebra as na;
extern crate num;
extern crate docopt;
extern crate vox_box;
extern crate rustc_serialize;
extern crate byteorder;
extern crate hound;

use std::path::Path;

use na::{DVec, Norm, Dot};
use docopt::Docopt;

use vox_box::spectrum::MFCC;
use vox_box::waves::{WindowType, Windower, Filter};

use byteorder::{WriteBytesExt, LittleEndian};

const USAGE: &'static str = "
Finds the MFCC cosine similarity of two sound files. If the sound files are not equal length, the
excess of the longer sound file is discarded.

Usage: 
    cosine_sim <file1> <file2> [--binary]
    cosine_sim (-h | --help)
    cosine_sim --version

Options:
    -h --help   Show this screen
    --version   Show version
    --binary    Emit binary data instead of ASCII
";

const VERSION: &'static str = "0.1.0";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_file1: String,
    arg_file2: String,
    flag_binary: bool
}

const NCOEFFS: usize = 26;

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.help(true).version(Some(String::from(VERSION))).decode())
        .unwrap_or_else(|e| e.exit());

    let mut file1 = hound::WavReader::open(&Path::new(&args.arg_file1)).unwrap();
    let mut file2 = hound::WavReader::open(&Path::new(&args.arg_file2)).unwrap();
    let mut samples1: Vec<f32> = file1.samples::<i32>().map(|s| s.unwrap() as f32).collect();
    let mut samples2: Vec<f32> = file2.samples::<i32>().map(|s| s.unwrap() as f32).collect();
    samples1.preemphasis(50.0 / 44100 as f32);
    samples2.preemphasis(50.0 / 44100 as f32);
    let file1_windows = Windower::new(WindowType::Hanning, samples1.as_slice(), 128, 512);
    let file2_windows = Windower::new(WindowType::Hanning, samples2.as_slice(), 128, 512);
    let mfcc_getter = |frame: Vec<f32>| -> Vec<f32> { frame.mfcc(NCOEFFS, (100., 8000.), 44_100.) };
    let file1_mfccs: Vec<Vec<f32>> = file1_windows.map(&mfcc_getter).collect();
    let file2_mfccs: Vec<Vec<f32>> = file2_windows.map(&mfcc_getter).collect();
    
    let min_length = std::cmp::min(file1_mfccs.len(), file2_mfccs.len());

    let mut file1_mfccs_trunc: Vec<f32> = Vec::<f32>::with_capacity(NCOEFFS * min_length);
    let mut file2_mfccs_trunc: Vec<f32> = Vec::<f32>::with_capacity(NCOEFFS * min_length);

    for i in 0..min_length {
        file1_mfccs_trunc.extend_from_slice(&file1_mfccs[i][..]);
        file2_mfccs_trunc.extend_from_slice(&file2_mfccs[i][..]);
    }

    let advec = DVec::<f32>::from_slice(NCOEFFS * min_length, &file1_mfccs_trunc[..]);
    let bdvec = DVec::<f32>::from_slice(NCOEFFS * min_length, &file2_mfccs_trunc[..]);

    let cosine_sim: f32 = advec.dot(&bdvec) / (advec.norm() * bdvec.norm());
    if args.flag_binary {
        std::io::stdout().write_f32::<LittleEndian>(cosine_sim).unwrap();
    } else {
        println!("sim: {}", cosine_sim);
    }
}

