use std::iter::{Iterator, IntoIterator};
use std::f64::consts::PI;
use std::i16;
use std::fs::File;
use std::ops::*;
use std::path::Path;
use std::cmp::Ordering::Equal;

pub trait Osc {
    fn sine(size: usize) -> Vec<f64>;
    fn saw(size: usize) -> Vec<f64>;
}

impl Osc for Vec<f64> {
    fn sine(size: usize) -> Vec<f64> {
        let mut sine: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = (i as f64) * 2.0 * PI / ((size) as f64);
            sine.push(phase.sin());
        }
        sine
    }

    fn saw(size: usize) -> Vec<f64> {
        let mut saw: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = (i as f64) / (size) as f64;
            saw.push((phase - 0.5) * -2.0);
        }
        saw
    }
}

pub trait Windowing {
    fn hanning(size: usize) -> Vec<f64>;
    fn hamming(size: usize) -> Vec<f64>;
}

impl Windowing for Vec<f64> {
    fn hanning(size: usize) -> Vec<f64> {
        let mut win: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(0.5 * (1.0 - phase.cos()));
        }
        win
    }
    
    fn hamming(size: usize) -> Vec<f64> {
        let mut win: Vec<f64> = Vec::with_capacity(size);
        for i in 0..size {
            let phase: f64 = ((i as f64) / ((size - 1) as f64)) * 2.0 * PI;
            win.push(0.54 - (0.46 * phase.cos()));
        }
        win
    }
}

pub trait Filter<T> {
    fn preemphasis(&mut self, factor: f64) -> &mut Vec<T>;
}

/// Factor is center frequency / sample_rate
impl<T> Filter<T> for Vec<T> where T: Mul<f64, Output=T> + Sub<T, Output=T> + Copy {
    fn preemphasis<'a>(&'a mut self, factor: f64) -> &'a mut Vec<T> {
        let filter = (-2.0 * PI * factor).exp();
        for i in (1..self.len()).rev() {
            self[i] = self[i] - (self[i-1] * filter);
        };
        self
    }
}

pub trait Autocorrelates<T> {
    fn autocorrelate(&self, n_coeffs: i32) -> Vec<T>;
    fn normalize(&mut self);
}

impl<T> Autocorrelates<T> for Vec<T> where T: Mul<T, Output=T> + Add<T, Output=T> + Copy + std::cmp::PartialOrd + Div<T, Output=T> {
    fn autocorrelate(&self, n_coeffs: i32) -> Vec<T> {
        let mut coeffs: Vec<T> = Vec::with_capacity(n_coeffs as usize);
        for lag in 0..n_coeffs {
            let mut accum = self[0];
            for i in 1..(self.len() - (lag as usize)) {
                accum = accum + (self[i as usize] * self[((i as i32) + lag) as usize]);
            }
            coeffs.push(accum);
        }
        coeffs
    }

    fn normalize(&mut self) {
        let mut sorted = self.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Equal));
        let max = sorted[0];
        for i in (0..self.len()) {
            self[i] = self[i] / max;
        }
    }
}

pub trait Polynomial<T> {
    fn find_roots(&self) -> Result<Vec<T>, &str>;
    fn laguerre(&self, z: Complex<f64>) -> Complex<T>;
    fn degree(&self) -> usize;
    fn off_low(&self) -> usize;
}

#[derive(Debug, PartialEq)]
pub struct Complex<T> {
    real: T,
    imag: T
}

impl Complex<f64> {
    pub fn zero() -> Complex<f64> {
        Complex { real: 0f64, imag: 0f64 }
    }

    pub fn real(r: f64) -> Complex<f64> {
        Complex { real: r, imag: 0.0 }
    }

    fn magnitude(&self) -> f64 {
        ((self.real.powi(2)) + (self.imag.powi(2))).sqrt()
    }

    fn neg(&self) -> Complex<f64> {
        Complex { real: (self.real * -1.0), imag: (self.imag * -1.0) }
    }

    fn sqrt(&self) -> Option<Complex<f64>> {
        let a = (self.real.powi(2) + self.imag.powi(2)).sqrt().sqrt();
        let b = 0.5 * self.real.atan2(self.imag);
        match b {
            Some(b) => {
                let c = Complex { 
                    real: b.unwrap().cos(), 
                    imag: b.unwrap().sin() 
                };
                Some(&c * &a)
            },
            None => { None }
        }

        // let xysq = (self.real.powi(2) + self.imag.powi(2)).sqrt();
        // let sgn_y = if self.imag >= 0f64 { 1.0 } else { 0.0 };
        // let rhs = Complex {
        //     real: (xysq + self.real).sqrt(),
        //     imag: sgn_y * (xysq - self.real).sqrt()
        // };
        // &rhs * &Complex::<f64>::real(0.5 * (2f64.sqrt()))
    }
}

impl<'a, 'b> Mul<&'b Complex<f64>> for &'a Complex<f64> {
    type Output = Complex<f64>;

    fn mul(self, rhs: &'b Complex<f64>) -> Complex<f64> {
        Complex {
            real: (self.real * rhs.real) + (self.imag * rhs.imag),
            imag: (self.real * rhs.imag) + (self.imag * rhs.real)
        }
    }
}

impl<'a, 'b> Add<&'b Complex<f64>> for &'a Complex<f64> {
    type Output = Complex<f64>;

    fn add(self, rhs: &'b Complex<f64>) -> Complex<f64> {
        Complex {
            real: (self.real + rhs.real),
            imag: (self.imag + rhs.imag)
        }
    }
}

impl<'a, 'b> Sub<&'b Complex<f64>> for &'a Complex<f64> {
    type Output = Complex<f64>;

    fn sub(self, rhs: &'b Complex<f64>) -> Complex<f64> {
        Complex {
            real: (self.real - rhs.real),
            imag: (self.imag - rhs.imag)
        }
    }
}

impl<'a, 'b> Div<&'b Complex<f64>> for &'a Complex<f64> {
    type Output = Complex<f64>;

    fn div(self, rhs: &'b Complex<f64>) -> Complex<f64> {
        self * &Complex {
            real: rhs.real,
            imag: rhs.imag * -1.0
        }
    }
}

impl<'b> Div<&'b Complex<f64>> for f64 {
    type Output = Complex<f64>;

    fn div(self, rhs: &'b Complex<f64>) -> Complex<f64> {
        &Complex::<f64>::real(self) / &rhs
    }
}

impl<'b> Mul<&'b Complex<f64>> for f64 {
    type Output = Complex<f64>;

    fn mul(self, rhs: &'b Complex<f64>) -> Complex<f64> {
        &Complex::<f64>::real(self) * &rhs
    }
}

impl Polynomial<f64> for Vec<f64> {
    fn find_roots(&self) -> Result<Vec<f64>, &str> {
        let mut m = self.degree();
        if m < 1 { return Err("Zero degree polynomial: no roots to be found.") }
        let mut coH = m;
        let mut coL = self.off_low();
        let mut z_roots = vec![0f64; coL];
        m = m - coL;
        // let mut coeffs = self[coL..(coH+1)];
        //
        // while m > 2 {
        //     let mut z = coeffs.laguerre(Complex<f64> {real: -64.0, imag: -64.0});
        //     m = m - 1;
        // }
        Err("Not yet implemented")
    }

    fn laguerre(&self, mut z: Complex<f64>) -> Complex<f64> {
        let n: usize = self.len() - 1;
        let max_iter: usize = 20;
        let mut k = 1;
        while k <= max_iter {
            let mut alpha = Complex::<f64>::real(self[n]);
            let mut beta = Complex::<f64>::zero();
            let mut gamma = Complex::<f64>::zero();

            for j in (0..n).rev() {
                gamma = &(&z * &gamma) + &beta;
                beta = &(&z * &beta) + &alpha;
                alpha = &(&z * &alpha) + &Complex::<f64>::real(self[j]);
            }

            // Return if the magnitude of alpha is too low
            if alpha.magnitude() <= 1e-14 { return z; }

            let ca = &beta.neg() / &alpha;
            let ca2 = &ca * &ca;
            let cb = &ca2 - &(&(&Complex::<f64>::real(2.0) * &gamma) / &alpha);
            let c1 = &(((n-1) as f64) * &(&((n as f64)*&cb) - &ca2)).sqrt();
            let cc1: Complex<f64> = &ca + &c1;
            let cc2: Complex<f64> = &ca - &c1;
            let mut cc;
            if cc1.magnitude() > cc2.magnitude() {
                cc = cc1
            } else {
                cc = cc2
            }
            cc = &cc / &Complex::<f64>::real(n as f64);
            let c2 = 1.0 / &cc;
            z = &z + &c2;
            k = &k + &1;
        }
        return z
    }

    fn degree(&self) -> usize {
        let mut d = (self.len() - 1);
        while self[d] == 0.0 { d = d - 1; }
        d
    }

    fn off_low(&self) -> usize {
        let mut index = 0;
        while self[index] == 0.0 { index = index + 1 };
        index
    }
}

pub trait LPC<T> {
    fn lpc(&self, n_coeffs: usize) -> Vec<T>;
}

impl LPC<f64> for Vec<f64> {
    fn lpc(&self, n_coeffs: usize) -> Vec<f64> {
        let mut ac: Vec<f64> = vec![0f64; n_coeffs + 1];
        let mut kc: Vec<f64> = vec![0f64; n_coeffs];
        let mut tmp: Vec<f64> = vec![0f64; n_coeffs];

        /* order 0 */
        let mut err = self[0];
        ac[0] = 1.0;

        /* order >= 1 */
        for i in (1..n_coeffs+1) {
            let mut acc = self[i];
            for j in (1..i) {
                acc = acc + (ac[j] * self[i-j]);
            }
            kc[i-1] = -1.0 * acc/err;
            ac[i] = kc[i-1];
            for j in (0..n_coeffs) {
                tmp[j] = ac[j];
            }
            for j in (1..i) {
                ac[j] = ac[j] + (kc[i-1] * tmp[i-j]);
            }
            err = err * (1f64 - (kc[i-1] * kc[i-1]));
        };
        ac
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    use std::cmp::Ordering::*;

    #[test]
    fn test_ac() { 
        let sine = Vec::<f64>::sine(16);
        let auto = sine.autocorrelate(16);
    }

    #[test]
    fn test_lpc() {
        let sine = Vec::<f64>::sine(8);
        let mut auto = sine.autocorrelate(8);
        auto.normalize();       
        let auto_exp = vec![1.0, 0.7071, 0.1250, -0.3536, -0.5, -0.3536, -0.1250, 0.0];
        // Rust output:
        // let lpc_exp = vec![1.0, -0.7071, 0.7500, -0.4041, 2.4146];
        let lpc_exp = vec![1.0, -1.3122, 0.8660, -0.0875, -0.0103];
        let lpc = auto.lpc(4);
        // println!("LPC coeffs: {:?}", &lpc);
        for (a, b) in auto.iter().zip(auto_exp.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
        for (a, b) in lpc.iter().zip(lpc_exp.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
    }

    #[test]
    fn test_roots() {
        let lpc_exp: Vec<f64> = vec![1.0, -1.3122, 0.8660, -0.0875, -0.0103];
        let roots_exp: Vec<f64> = vec![ -14.760427263199, 2.0324665076629, 4.2426443833034 ];
        let roots: Vec<f64> = lpc_exp.find_roots().unwrap();
        println!("roots: {:?}", roots);
        for (a, b) in roots_exp.iter().zip(roots.iter()) {
            assert![(a - b).abs() < 0.0001];
        }
    }

    #[test]
    fn test_laguerre() {
        let vec: Vec<f64> = vec![1.0, 2.0];
        let point: Complex<f64> = Complex { real: -64.0, imag: -64.0 };
        let res = vec.laguerre(point);
        println!("res: {:?}", res);
        assert_eq!(res, Complex {real: -0.5, imag: 0.0});
    }

    #[test]
    fn test_complex_sqrt() {
        let sqred = Complex {
            real: 2.0,
            imag: -3.0
        };
        let complex_sqrt = Complex { 
            real: 1.6741492280355, 
            imag: -0.89597747612984
        };
        let sqrt = sqred.sqrt().unwrap();
        println!("sqrt: {:?}", sqrt);
        assert!(&sqrt.eq(&complex_sqrt));
    }

    #[test]
    fn test_pe() {
        let mut saw = Vec::<f64>::saw(32);
        let mut sine = Vec::<f64>::sine(32);
        let pre_saw = saw.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
        let pre_sine = sine.preemphasis(0.1f64); // preemphasize at 0.1 * sampling rate
    }
}

