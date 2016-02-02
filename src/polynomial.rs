#![feature(test)]
extern crate num;

use std::ops::Neg;
use std::iter::*;
use std::fmt::Debug;
use std::marker::Sized;

use num::{Float, Num, Zero, One, FromPrimitive, Complex};

pub trait Polynomial<'a, T> {
    fn degree(&self) -> usize;
    fn off_low(&self) -> usize;
    fn laguerre(&self, z: Complex<T>) -> Complex<T>;

    fn find_roots(&self) -> Result<Vec<Complex<T>>, &str>;
    fn find_roots_mut<'b>(&'b mut self, &'b mut [Complex<T>]) -> Result<(), &str>;

    fn div_polynomial(&mut self, z: Complex<T>) -> Result<Vec<Complex<T>>, &str>;
    fn div_polynomial_mut(&'a mut self, other: Complex<T>, rem: &'a mut [Complex<T>]) -> Result<(), &str>;
}

impl<'a, T> Polynomial<'a, T> for [Complex<T>] 
    where T: Float +
             Num +
             Clone +
             Debug +
             FromPrimitive +
             Into<Complex<T>>
{
    fn degree(&self) -> usize {
        self.iter().rposition(|r| r != &Complex::<T>::zero()).unwrap_or(0)
    }

    fn off_low(&self) -> usize {
        self.iter().position(|r| r != &Complex::<T>::zero()).unwrap_or(0)
    }

    fn laguerre(&self, start: Complex<T>) -> Complex<T> {
        let n: usize = self.len() - 1;
        let mut z = start;
        // max iterations of 20
        for k in 0..20 {
            let mut abg = [self[n], Complex::<T>::zero(), Complex::<T>::zero()];

            for j in (0..n).rev() {
                abg[2] = abg[2] * z + abg[1];
                abg[1] = abg[1] * z + abg[0];
                abg[0] = abg[0] * z + self[j];
            }

            if abg[0].norm() <= T::from_f32(1e-14f32).unwrap() { return z; }

            let ca: Complex<T> = abg[1].neg() / abg[0];
            let ca2: Complex<T> = ca * ca;
            let cb: Complex<T> = ca2 - ((Complex::<T>::from(T::one() + T::one()) * abg[2]) / abg[0]);
            let c1: Complex<T> = ((Complex::<T>::from(T::from_usize(n-1).unwrap()) *
                                  Complex::<T>::from(T::from_usize(n).unwrap()) * cb) - ca2).sqrt();

            let cc1: Complex<T> = ca + c1;
            let cc2: Complex<T> = ca - c1;

            let cc = if cc1.norm() > cc2.norm() {
                cc1 / Complex::<T>::from(T::from_usize(n).unwrap())
            } else {
                cc2 / Complex::<T>::from(T::from_usize(n).unwrap())
            };

            let c2 = cc.inv();
            z = z + c2;
        }
        z
    }

    fn find_roots(&self) -> Result<Vec<Complex<T>>, &str> {
        let work_size = self.len() * 6 + 4;
        let mut work: Vec<Complex<T>> = vec![Complex::<T>::from(T::zero()); work_size];
        let mut other = self.to_vec();
        {
            other.find_roots_mut(&mut work[..]);
        }
        while other[other.len()-1] == Complex::<T>::zero() {
            other.pop();
        }
        Ok(other)
    }

    /// work must be 3*size+2 for complex floats (meaning 6*size+4 of the buffer)
    fn find_roots_mut<'b>(&'b mut self, work: &'b mut [Complex<T>]) -> Result<(), &str> {
        // Initialize coefficient highs and lows
        let coeff_high = self.degree();
        if coeff_high < 1 { return Err("Zero degree polynomial: no roots to be found.") }

        let coeff_low: usize = self.off_low();
        let mut m = coeff_high - coeff_low;

        // work should be 2*self.len()
        let (mut z_roots, mut work) = work.split_at_mut(2*self.len());
        let mut z_root_index = 0;
        for i in 0..coeff_low {
            z_roots[i] = Complex::<T>::zero();
            z_root_index = z_root_index + 1;
        }

        let (mut rem, mut work) = work.split_at_mut(coeff_high - coeff_low + 1);
        let (mut coeffs, mut work) = work.split_at_mut(coeff_high - coeff_low + 1);
        for co in coeff_low..(coeff_high+1) {
            coeffs[co] = self[co];
        }
        // println!("&[] coeffs: {:?}", coeffs);

        // Use the Laguerre method to factor out a single root
        for i in (3..(m+1)).rev() {
            let z = coeffs.laguerre(Complex::<T>::new(T::from_f32(-64.0f32).unwrap(), T::from_f32(-64.0f32).unwrap()));
            z_roots[z_root_index] = z;
            z_root_index += 1;
            // println!("z is {:?}", z);
            match coeffs.div_polynomial_mut(z.neg(), &mut rem) {
                Err(_) => { return Err("Failed to find roots") },
                Ok(_) => { }
            }
            // println!("&[] coeffs are: {:?}", coeffs);
            m = m - 1;
        }

        // Solve quadradic equation
        if m == 2 {
            let a2 = coeffs[2] + coeffs[2];
            let d = ((coeffs[1] * coeffs[1]) - (Complex::<T>::from(T::from_i8(4i8).unwrap()) * coeffs[2] * coeffs[0])).sqrt();
            let x = coeffs[1].neg();
            // println!("a2: {:?}, d: {:?}, x: {:?}", a2, d, x);
            z_roots[z_root_index] = (x + d) / a2;
            z_roots[z_root_index + 1] = (x - d) / a2;
            z_root_index += 2;
        }
        // Solve linear equation
        if m == 1 {
            z_roots[z_root_index] = coeffs[0].neg() / coeffs[1];
            z_root_index += 1;
        }
        for i in 0..(z_root_index+1) {
            self[i] = z_roots[i];
        }
        for i in (z_root_index+1)..self.len() {
            self[i] = Complex::<T>::zero();
        }
        Ok(())
    }

    /// Divides self by other, and stores the remainder in rem
    fn div_polynomial_mut(&'a mut self, other: Complex<T>, rem: &'a mut [Complex<T>]) -> Result<(), &str> {
        for i in 0..self.len() {
            rem[i] = self[i];
        }

        if other != Complex::<T>::zero() {
            let ns = self.iter().rposition(|x| *x != Complex::<T>::zero()).unwrap_or(0);
            let ds = 1;
            for i in (0..(ns - ds + 1)).rev() {
                self[i] = rem[ds + i];
                for j in i..(ds + i) {
                    if j - i == 0 {
                        rem[j] = rem[j] - (self[i] * other);
                    } else if j - i == 1 {
                        rem[j] = rem[j] - (self[i] * Complex::<T>::one());
                    }
                }
            }
            // println!("self: {:?}", self);
            for i in ds..(ns+1) { 
                let l = rem.iter().rposition(|x| *x != Complex::<T>::zero()).unwrap_or(0);
                rem[l] = Complex::<T>::zero();
            }
            let l = self.iter().rposition(|x| *x != Complex::<T>::zero()).unwrap_or(0);
            // println!("ns, ds: {:?}, {:?}, {:?}", ns, ds, l + 1);
            for i in 0..((l + 1) - ns - ds + 1) { 
                let l = self.iter().rposition(|x| *x != Complex::<T>::zero()).unwrap_or(0);
                self[l] = Complex::<T>::zero();
            } 
            // println!("self: {:?}", &self);
            // println!("rem: {:?}", &rem);
            Ok(())
        } else {
            if other != Complex::<T>::zero() {
                for f in rem.iter_mut() {
                    *f = *f / other;
                }
                Ok(())
            } else {
                Err("Tried to divide by zero")
            }
        }
    }

    /// Returns the remainder
    fn div_polynomial(&mut self, other: Complex<T>) -> Result<Vec<Complex<T>>, &str> {
        let mut rem = self.to_vec();
        let mut res: Result<(), &str> = Ok(());
        {
            self.div_polynomial_mut(other, &mut rem[..]);
        }
        Ok(rem)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    extern crate num;

    use super::*;
    use num::complex::Complex;

    #[bench]
    fn bench_degree(b: &mut test::Bencher) {
        let mut x: Vec<Complex<f32>> = vec![0.0, 0.0, 3.0, 4.0, 2.0, 6.0, 0.0, 0.0].iter().map(|v| Complex::<f32>::from(v)).collect();
        b.iter(|| (&mut x[..]).degree());
    }

    #[bench]
    fn bench_off_low(b: &mut test::Bencher) {
        let mut x: Vec<Complex<f32>> = vec![0.0, 0.0, 3.0, 4.0, 2.0, 6.0, 0.0, 0.0].iter().map(|v| Complex::<f32>::from(v)).collect();
        b.iter(|| (&mut x[..]).off_low());
    }

    #[bench]
    // 3,901 ns/iter (+/- 707)
    fn bench_laguerre_slice(b: &mut test::Bencher) {
        let mut vec: Vec<Complex<f64>> = vec![1.0, 2.5, 2.0, 3.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let point: Complex<f64> = Complex::<f64>::new(-64.0, -64.0);
        b.iter(|| (&mut vec[..]).laguerre(point));
    }

    #[bench]
    fn bench_div_polynomial_mut_slice(b: &mut test::Bencher) {
        let x: Vec<Complex<f64>> = vec![1f64, 2.0, -2.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let mut vec: Vec<Complex<f64>> = vec![1f64, 2.0, -2.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let other: Complex<f64> = Complex::<f64>::new(1f64, 2.5);
        let mut rem: [Complex<f64>; 3] = [Complex::<f64>::new(0f64, 0f64); 3];
        b.iter(|| {
            (&mut vec[..]).div_polynomial_mut(other, &mut rem[..]);
            for i in 0..vec.len() { vec[i] = x[i]; }
        });
    }

    // Broken until I actually implement a POLYNOMIAL division
    // #[test]
    // fn test_div_polynomial() {
    //     let exp_quo: Vec<Complex<f64>> = vec![1.32, -0.8].iter().map(|v| Complex::<f64>::from(v)).collect();
    //     let exp_rem: Vec<Complex<f64>> = vec![-0.32].iter().map(|v| Complex::<f64>::from(v)).collect();
    //     let mut a: Vec<Complex<f64>> = vec![1f64, 2.5, -2.0].iter().map(|v| Complex::<f64>::from(v)).collect();
    //     let b = vec![1f64, 2.5].iter().map(|v| Complex::<f64>::from(v)).collect();
    //     {
    //         let rem = a.div_polynomial(b).unwrap();
    //         assert_eq!(rem.len(), exp_rem.len());
    //         for i in 0..exp_rem.len() {
    //             let diff: Complex<f64> = rem[i] - exp_rem[i];
    //             let re: f64 = diff.re;
    //             let im: f64 = diff.im;
    //             println!("diff: {:?}", diff);
    //             assert!(re.abs() < 1e-10);
    //             assert!(im.abs() < 1e-10);
    //         }
    //     }
    //     let eqlen: usize = exp_quo.len();
    //     assert_eq!(a.len(), eqlen);
    //     for i in 0..eqlen {
    //         let diff: Complex<f64> = a[i] - exp_quo[i];
    //         let re: f64 = diff.re;
    //         let im: f64 = diff.im;
    //         println!("diff: {:?}", diff);
    //         assert!(re.abs() < 1e-10);
    //         assert!(im.abs() < 1e-10);
    //     }
    // }
    //
    // #[test]
    // fn test_div_polynomial_f32() {
    //     let exp_quo: Vec<Complex<f32>> = vec![1.32f32, -0.8].iter().map(|v| Complex::<f32>::from(v)).collect();
    //     let exp_rem: Vec<Complex<f32>> = vec![-0.32f32].iter().map(|v| Complex::<f32>::from(v)).collect();
    //     let mut a: Vec<Complex<f32>> = vec![1f32, 2.5, -2.0].iter().map(|v| Complex::<f32>::from(v)).collect();
    //     let b: Vec<Complex<f32>> = vec![1f32, 2.5].iter().map(|v| Complex::<f32>::from(v)).collect();
    //     {
    //         let rem = a.div_polynomial(b).unwrap();
    //         println!("rem: {:?}", rem);
    //         assert_eq!(rem.len(), exp_rem.len());
    //         for i in 0..exp_rem.len() {
    //             let diff = rem[i] - exp_rem[i];
    //             assert!(diff.re.abs() < 1e-5);
    //             assert!(diff.im.abs() < 1e-5);
    //         }
    //     }
    //     assert_eq!(a.len(), exp_quo.len());
    //     for i in 0..exp_quo.len() {
    //         let diff = a[i] - exp_quo[i];
    //         assert!(diff.re.abs() < 1e-5);
    //         assert!(diff.im.abs() < 1e-5);
    //     }
    // }

    #[test]
    fn test_degree() {
        let a: Vec<Complex<f64>> = vec![3.0, 2.0, 4.0, 0.0, 0.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        assert_eq!(a.degree(), 2);
    }

    #[test]
    fn test_off_low() {
        let a: Vec<Complex<f64>> = vec![0.0f64, 0.0, 3.0, 2.0, 4.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        assert_eq!(a.off_low(), 2);
    }

    #[test]
    fn test_laguerre() {
        let vec: Vec<Complex<f64>> = vec![1.0, 2.5, 2.0, 3.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let exp: Complex<f64> = Complex::<f64>::new( -0.1070229535872, -0.8514680262155 );
        let point: Complex<f64> = Complex::<f64>::new(-64.0, -64.0);
        let res = vec.laguerre(point);
        let diff = exp - res;
        println!("res: {:?}", res);
        println!("diff: {:?}", diff);
        assert!(diff.re < 0.00000001);
        assert!(diff.im < 0.00000001);
    }

    #[test]
    fn test_1d_roots() {
        let poly: Vec<Complex<f64>> = vec![1.0, 2.5].iter().map(|v| Complex::<f64>::from(v)).collect();
        let roots = poly.find_roots().unwrap();
        let roots_exp = vec![Complex::<f64>::new(-0.4, 0.0)];
        assert_eq!(roots.len(), 1);
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-12);
            assert!(diff.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_roots() {
        let poly: Vec<Complex<f64>> = vec![1.0, 2.5, -2.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let roots = poly.find_roots().unwrap();
        let roots_exp = vec![Complex::<f64>::new(-0.31872930440884, 0.0), Complex::<f64>::new(1.5687293044088, 0.0)];
        println!("Roots found: {:?}", roots);
        assert_eq!(roots.len(), roots_exp.len());
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-12);
            assert!(diff.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_complex_roots() {
        let coeffs: Vec<Complex<f64>> = vec![1.0, -2.5, 2.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let roots = coeffs.find_roots().unwrap();
        let roots_exp = vec![Complex::<f64>::new( 0.625, -0.33071891388307 ), Complex::<f64>::new( 0.625, 0.33071891388307 )];
        assert_eq!(roots.len(), roots_exp.len());
        println!("Roots found: {:?}", roots);
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-12);
            assert!(diff.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_complex_roots_f32() {
        let coeffs: Vec<Complex<f32>> = vec![1.0, -2.5, 2.0].iter().map(|v| Complex::<f32>::from(v)).collect();
        let roots = coeffs.find_roots().unwrap();
        let roots_exp = vec![Complex::<f32>::new( 0.625, -0.33071891388307 ), Complex::<f32>::new( 0.625, 0.33071891388307 )];
        assert_eq!(roots.len(), roots_exp.len());
        println!("Roots found: {:?}", roots);
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-12);
            assert!(diff.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_hi_d_roots() {
        let lpc_exp: Vec<Complex<f64>> = vec![1.0, 2.5, -2.0, -3.0].iter().map(|v| Complex::<f64>::from(v)).collect();
        let roots_exp = vec![Complex::<f64>::new(-1.1409835232292, 0.0), Complex::<f64>::new(-0.35308705904629, 0.0), Complex::<f64>::new(0.82740391560878, 0.0)];
        let roots = lpc_exp.find_roots().unwrap();
        println!("Roots: {:?}", roots);

        assert_eq!(roots.len(), roots_exp.len());
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-12);
            assert!(diff.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_hi_d_roots_f32() {
        let lpc_exp: Vec<Complex<f32>> = vec![1.0, 2.5, -2.0, -3.0].iter().map(|v| Complex::<f32>::from(v)).collect();
        let roots_exp = vec![Complex::<f32>::new(-1.1409835232292, 0.0), Complex::<f32>::new(-0.35308705904629, 0.0), Complex::<f32>::new(0.82740391560878, 0.0)];
        let roots = lpc_exp.find_roots().unwrap();
        println!("Roots: {:?}", roots);

        assert_eq!(roots.len(), roots_exp.len());
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-6);
            assert!(diff.im.abs() < 1e-6);
        }
    }

    #[test]
    fn test_f32_roots() {
        let lpc_coeffs: Vec<Complex<f32>> = vec![1.0, -0.99640256, 0.25383306, -0.25471634, 0.5084799, -0.0685858, -0.35042483, 0.07676613, -0.12874511, 0.11829436, 0.023972526].iter().map(|v| Complex::<f32>::from(v)).collect();
        let roots = lpc_coeffs.laguerre(Complex::<f32>::new(-64.0, -64.0));
        println!("Roots: {:?}", roots);
        assert!(roots.re.is_finite());
        assert!(roots.im.is_finite());
    }
}

