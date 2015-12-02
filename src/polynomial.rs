#![feature(test)]
use complex::*;
use std::ops::Neg;
use std::iter::*;
use num::traits::{Zero, One, FromPrimitive};
use num::Float;
use std::fmt::Debug;
use std::marker::Sized;

pub trait Polynomial<'a, 'b, T> : Sized {
    fn degree(&self) -> usize;
    fn off_low(&self) -> usize;
    fn laguerre(&self, z: Complex<T>) -> Complex<T>;
    fn find_roots_mut(&mut self, Self) -> Result<(), &str>;
    fn find_roots(&self) -> Result<Self, &str>;
    fn div_polynomial_mut(&'b mut self, other: Complex<T>, rem: &'b mut [Complex<T>]) -> Result<(), &str>;
    fn div_polynomial(&'b mut self, z: Vec<Complex<T>>) -> Result<Vec<Complex<T>>, &str>;
}

impl<'a, 'b, T: Float + Debug + ToComplex<T> + FromPrimitive + Sized> Polynomial<'a, 'b, T> for &'a mut [Complex<T>] {
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
            let cb: Complex<T> = ca2 - ((T::from_f32(2f32).unwrap().to_complex() * abg[2]) / abg[0]);
            let c1: Complex<T> = ((T::from_usize(n-1).unwrap().to_complex() *
                                  T::from_usize(n).unwrap().to_complex() * cb) - ca2).sqrt();

            let cc1: Complex<T> = ca + c1;
            let cc2: Complex<T> = ca - c1;

            let cc = if cc1.norm() > cc2.norm() {
                cc1 / T::from_usize(n).unwrap().to_complex()
            } else {
                cc2 / T::from_usize(n).unwrap().to_complex()
            };

            let c2 = cc.inv();
            z = z + c2;
        }
        z
    }

    fn find_roots_mut(&mut self, work: Self) -> Result<(), &str> {
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
            let d = ((coeffs[1] * coeffs[1]) - (T::from_i8(4i8).unwrap().to_complex() * coeffs[2] * coeffs[0])).sqrt();
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

    fn find_roots(&self) -> Result<Self, &str> {
        Err("Not implemented.")
    }

    /// Divides self by other, and stores the remainder in rem
    fn div_polynomial_mut(&'b mut self, other: Complex<T>, rem: &'b mut [Complex<T>]) -> Result<(), &str> {
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
    fn div_polynomial(&'b mut self, other: Vec<Complex<T>>) -> Result<Vec<Complex<T>>, &str> {
        Err("Not implemented")
    }
}

impl<'a, 'b, T: Float + Debug + ToComplex<T> + FromPrimitive> Polynomial<'a, 'b, T> for Vec<Complex<T>> {
    fn div_polynomial_mut(&mut self, other: Complex<T>, rem: &'b mut [Complex<T>]) -> Result<(), &str> {
        Err("Not implemented")
    }

    fn div_polynomial(&'b mut self, z: Vec<Complex<T>>) -> Result<Vec<Complex<T>>, &str> {
        let mut rem = self.clone();
        let other = z;
        if other.len() > 1 {
            let ns = self.len() - 1;
            let ds = match other.iter().rposition(|x| *x != Complex::<T>::zero()) {
                Some(x) => { x },
                None => { return Err("Tried to divide by zero") }
            };
            // println!("ns, ds: {:?}, {:?}, {:?}", ns, ds, self.len());
            for i in (0..(ns - ds + 1)).rev() {
                self[i] = rem[ds + i] / other[ds];
                for j in i..(ds + i) {
                    rem[j] = rem[j] - (self[i] * other[j-i]);
                }
            }
            // println!("self: {:?}", self);
            for i in ds..(ns+1) { rem.pop(); }
            for i in 0..(self.len() - ns - ds + 1) { self.pop(); } 
            // println!("self: {:?}", &self);
            // println!("rem: {:?}", &rem);
            Ok(rem)
        } else {
            let divisor = other.first().unwrap_or(return Err("Something went wrong."));
            if *divisor != Complex::<T>::zero() {
                Ok(self.iter().map(|f| { f / divisor }).collect())
            } else {
                Err("Tried to divide by zero")
            }
        }
    }

    fn find_roots_mut(&mut self, work: Self) -> Result<(), &str> {
        Err("Not implemented")
    }

    fn find_roots(&self) -> Result<Self, &str> {
        // Initialize coefficient highs and lows
        let coeff_high = self.degree();
        if coeff_high < 1 { return Err("Zero degree polynomial: no roots to be found.") }

        let coeff_low: usize = self.off_low();
        let mut m = coeff_high - coeff_low;

        // Initialize roots to output
        let mut z_roots: Vec<Complex<T>> = vec![Complex::<T>::zero(); coeff_low];
        let mut coeffs: Vec<Complex<T>> = Vec::<Complex<T>>::with_capacity(coeff_high - coeff_low);
        for co in self[coeff_low..(coeff_high+1)].iter() {
            coeffs.push(*co);
        }
        // println!("Vec coeffs: {:?}", coeffs);

        for i in (3..(m+1)).rev() {
            let z = coeffs.laguerre(Complex::<T>::new(T::from_f32(-64.0f32).unwrap(), T::from_f32(-64.0f32).unwrap()));
            // Some margin of error for mostly-real roots
            z_roots.push(z);
            // println!("z is {:?}", z);
            match coeffs.div_polynomial(vec![z.neg(), Complex::<T>::one()]) {
                Err(_) => { return Err("Failed to find roots") },
                Ok(_) => { }
            }
            // println!("Vec coeffs are: {:?}", coeffs);
            m = m - 1;
        }

        // Solve quadradic equation
        if m == 2 {
            let a2 = coeffs[2] + coeffs[2];
            let d = ((coeffs[1] * coeffs[1]) - (T::from_i8(4i8).unwrap().to_complex() * coeffs[2] * coeffs[0])).sqrt();
            let x = coeffs[1].neg();
            // println!("a2: {:?}, d: {:?}, x: {:?}", a2, d, x);
            z_roots.push((x + d) / a2);
            z_roots.push((x - d) / a2);
        }
        // Solve linear equation
        if m == 1 {
            z_roots.push(coeffs[0].neg() / coeffs[1]);
        }
        Ok(z_roots)
    }

    fn degree(&self) -> usize {
        self.iter().rposition(|r| *r != Complex::<T>::zero()).unwrap()
    }

    fn off_low(&self) -> usize {
        self.iter().position(|r| *r != Complex::<T>::zero()).unwrap()
    }

    fn laguerre(&self, mut z: Complex<T>) -> Complex<T> {
        let n: usize = self.len() - 1;
        // max iterations of 20
        for k in 0..20 {
            let mut alpha = self[n];
            let mut beta = Complex::<T>::zero();
            let mut gamma = Complex::<T>::zero();

            for j in (0..n).rev() {
                gamma = (z * gamma) + beta;
                beta = (z * beta) + alpha;
                alpha = (z * alpha) + self[j];
            }

            if alpha.norm() <= T::from_f32(1e-14f32).unwrap() { return z; }

            let ca: Complex<T> = beta.neg() / alpha;
            let ca2: Complex<T> = ca * ca;
            let cb: Complex<T> = ca2 - ((T::from_f32(2f32).unwrap().to_complex() * gamma) / alpha);
            let c1: Complex<T> = ((T::from_usize(n-1).unwrap().to_complex() *
                                  T::from_usize(n).unwrap().to_complex() * cb) - ca2).sqrt();

            let cc1: Complex<T> = ca + c1;
            let cc2: Complex<T> = ca - c1;

            let cc = if cc1.norm() > cc2.norm() {
                cc1 / T::from_usize(n).unwrap().to_complex()
            } else {
                cc2 / T::from_usize(n).unwrap().to_complex()
            };

            let c2 = cc.inv();
            z = z + c2;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use super::super::complex::*;

    #[bench]
    fn bench_degree(b: &mut test::Bencher) {
        let mut x: Vec<Complex<f32>> = vec![0.0, 0.0, 3.0, 4.0, 2.0, 6.0, 0.0, 0.0].to_complex_vec();
        b.iter(|| (&mut x[..]).degree());
    }

    #[bench]
    fn bench_off_low(b: &mut test::Bencher) {
        let mut x: Vec<Complex<f32>> = vec![0.0, 0.0, 3.0, 4.0, 2.0, 6.0, 0.0, 0.0].to_complex_vec();
        b.iter(|| (&mut x[..]).off_low());
    }

    #[bench]
    // 3,901 ns/iter (+/- 707)
    fn bench_laguerre_slice(b: &mut test::Bencher) {
        let mut vec: Vec<Complex<f64>> = vec![1.0, 2.5, 2.0, 3.0].to_complex_vec();
        let point: Complex64 = Complex64::new(-64.0, -64.0);
        b.iter(|| (&mut vec[..]).laguerre(point));
    }

    #[bench]
    fn bench_div_polynomial_mut_slice(b: &mut test::Bencher) {
        let x = vec![1f64, 2.0, -2.0].to_complex_vec();
        let mut vec: Vec<Complex<f64>> = vec![1f64, 2.0, -2.0].to_complex_vec();
        let other: Complex<f64> = Complex::<f64>::new(1f64, 2.5);
        let mut rem: [Complex<f64>; 3] = [Complex64::new(0f64, 0f64); 3];
        b.iter(|| {
            (&mut vec[..]).div_polynomial_mut(other, &mut rem[..]);
            for i in 0..vec.len() { vec[i] = x[i]; }
        });
    }

    #[test]
    fn test_div_polynomial() {
        let exp_quo = vec![1.32, -0.8].to_complex_vec();
        let exp_rem = vec![-0.32].to_complex_vec();
        let mut a = vec![1f64, 2.5, -2.0].to_complex_vec();
        let b = vec![1f64, 2.5].to_complex_vec();
        {
            let rem = a.div_polynomial(b).unwrap();
            assert_eq!(rem.len(), exp_rem.len());
            for i in 0..exp_rem.len() {
                let diff = rem[i] - exp_rem[i];
                println!("diff: {:?}", diff);
                assert!(diff.re.abs() < 1e-10);
                assert!(diff.im.abs() < 1e-10);
            }
        }
        assert_eq!(a.len(), exp_quo.len());
        for i in 0..exp_quo.len() {
            let diff = a[i] - exp_quo[i];
            println!("diff: {:?}", diff);
            assert!(diff.re.abs() < 1e-10);
            assert!(diff.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_div_polynomial_f32() {
        let exp_quo = vec![1.32f32, -0.8].to_complex_vec();
        let exp_rem = vec![-0.32f32].to_complex_vec();
        let mut a = vec![1f32, 2.5, -2.0].to_complex_vec();
        let b = vec![1f32, 2.5].to_complex_vec();
        {
            let rem = a.div_polynomial(b).unwrap();
            println!("rem: {:?}", rem);
            assert_eq!(rem.len(), exp_rem.len());
            for i in 0..exp_rem.len() {
                let diff = rem[i] - exp_rem[i];
                assert!(diff.re.abs() < 1e-5);
                assert!(diff.im.abs() < 1e-5);
            }
        }
        assert_eq!(a.len(), exp_quo.len());
        for i in 0..exp_quo.len() {
            let diff = a[i] - exp_quo[i];
            assert!(diff.re.abs() < 1e-5);
            assert!(diff.im.abs() < 1e-5);
        }
    }

    #[test]
    fn test_degree() {
        let a: Vec<Complex<f64>> = vec![3.0, 2.0, 4.0, 0.0, 0.0].to_complex_vec();
        assert_eq!(a.degree(), 2);
    }

    #[test]
    fn test_off_low() {
        let a: Vec<Complex<f64>> = vec![0.0f64, 0.0, 3.0, 2.0, 4.0].to_complex_vec();
        assert_eq!(a.off_low(), 2);
    }

    #[test]
    fn test_laguerre() {
        let vec: Vec<Complex<f64>> = vec![1.0, 2.5, 2.0, 3.0].to_complex_vec();
        let exp: Complex64 = Complex64::new( -0.1070229535872, -0.8514680262155 );
        let point: Complex64 = Complex64::new(-64.0, -64.0);
        let res = vec.laguerre(point);
        let diff = exp - res;
        println!("res: {:?}", res);
        println!("diff: {:?}", diff);
        assert!(diff.re < 0.00000001);
        assert!(diff.im < 0.00000001);
    }

    #[test]
    fn test_1d_roots() {
        let poly: Vec<Complex<f64>> = vec![1.0, 2.5].to_complex_vec();
        let roots = poly.find_roots().unwrap();
        let roots_exp = vec![Complex64::new(-0.4, 0.0)];
        assert_eq!(roots.len(), 1);
        for i in 0..roots_exp.len() {
            let diff = roots[i] - roots_exp[i];
            assert!(diff.re.abs() < 1e-12);
            assert!(diff.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_roots() {
        let poly: Vec<Complex<f64>> = vec![1.0, 2.5, -2.0].to_complex_vec();
        let roots = poly.find_roots().unwrap();
        let roots_exp = vec![Complex64::new(-0.31872930440884, 0.0), Complex64::new(1.5687293044088, 0.0)];
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
        let coeffs: Vec<Complex<f64>> = vec![1.0, -2.5, 2.0].to_complex_vec();
        let roots = coeffs.find_roots().unwrap();
        let roots_exp = vec![Complex64::new( 0.625, -0.33071891388307 ), Complex64::new( 0.625, 0.33071891388307 )];
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
        let coeffs: Vec<Complex<f32>> = vec![1.0, -2.5, 2.0].to_complex_vec();
        let roots = coeffs.find_roots().unwrap();
        let roots_exp = vec![Complex32::new( 0.625, -0.33071891388307 ), Complex32::new( 0.625, 0.33071891388307 )];
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
        let lpc_exp: Vec<Complex<f64>> = vec![1.0, 2.5, -2.0, -3.0].to_complex_vec();
        let roots_exp = vec![Complex64::new(-1.1409835232292, 0.0), Complex64::new(-0.35308705904629, 0.0), Complex64::new(0.82740391560878, 0.0)];
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
        let lpc_exp: Vec<Complex<f32>> = vec![1.0, 2.5, -2.0, -3.0].to_complex_vec();
        let roots_exp = vec![Complex32::new(-1.1409835232292, 0.0), Complex32::new(-0.35308705904629, 0.0), Complex32::new(0.82740391560878, 0.0)];
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
        let lpc_coeffs: Vec<Complex<f32>> = vec![1.0, -0.99640256, 0.25383306, -0.25471634, 0.5084799, -0.0685858, -0.35042483, 0.07676613, -0.12874511, 0.11829436, 0.023972526].to_complex_vec();
        let roots = lpc_coeffs.laguerre(Complex32::new(-64.0, -64.0));
        println!("Roots: {:?}", roots);
        assert!(roots.re.is_finite());
        assert!(roots.im.is_finite());
    }
}

