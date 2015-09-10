use super::complex::*;
use std::ops::{Neg, Add, Sub};
use std::iter::*;

pub trait Polynomial<T: Neg> {
    fn degree(&self) -> usize;
    fn off_low(&self) -> usize;
    fn laguerre(&self, z: Complex64) -> Complex64;
    fn find_roots(&self) -> Result<Vec<Complex64>, &str>;
    fn div_polynomial(&mut self, Vec<T>) -> Result<Vec<T>, &str>;
}

impl Polynomial<f64> for Vec<f64> {
    fn div_polynomial(&mut self, other: Vec<f64>) -> Result<Vec<f64>, &str> {
        if other.len() > 1 {
            let mut ds = other.len() - 1;
            let mut ns = self.len() - 1;
            while ds >= 0 && other[ds] == 0.0 { ds = ds - 1; }
            if ds < 0 { return Err("tried to divide by zero"); }
            let mut rem = self.clone();
            for i in (0..(ns - ds + 1)).rev() {
                self[i] = rem[ds + i] / other[ds];
                for j in (i..(ds + i)) {
                    rem[j] = rem[j] - (self[i] * other[j-i]);
                }
            }
            for i in ds..(ns+1) {
                rem.pop();
            }
            for i in (0..(self.len() - ns - ds + 1)) {
                self.pop();
            }
            Ok(rem)
        } else {
            let divisor = other.first().unwrap();
            if *divisor != 0.0 {
                Ok(self.iter().map(|f| { f / divisor }).collect())
            } else {
                Err("Tried to divide by zero")
            }
        }
    }

    fn find_roots(&self) -> Result<Vec<Complex64>, &str> {
        let mut m: usize = self.degree();
        if m < 1 { return Err("Zero degree polynomial: no roots to be found.") }

        // Initialize coefficient highs and lows
        let mut coH: usize = m;
        let mut coL: usize = self.off_low();
        m = m - coL;

        // Initialize roots to output
        let mut z_roots: Vec<Complex64> = vec![Complex64::new(0f64, 0f64); coL];
        let mut coeffs: Vec<f64> = Vec::<f64>::with_capacity(coH - coL);
        for co in self[coL..(coH+1)].iter() {
            coeffs.push(*co);
        }

        while m > 2 {
            let mut z = coeffs.laguerre(Complex64::new(-64.0, -64.0));
            // Some margin of error for mostly-real roots
            z_roots.push(z);
            match coeffs.div_polynomial(vec![z.re.neg(), 1f64]) {
                Err(x) => { return Err("Failed to find roots") },
                Ok(_) => { }
            }
            m = m - 1;
        }

        // Solve quadradic equation
        if m == 2 {
            let a2 = Complex64::new(coeffs[2] * 2f64, 0f64);
            let d = Complex64::new(((coeffs[1].powi(2)) - (4f64 * coeffs[2] * coeffs[0])), 0f64);
            let x = Complex64::new(coeffs[1].neg(), 0f64);
            // println!("a2: {:?}, d: {:?}, x: {:?}", a2, d, x);
            z_roots.push((x + d) / a2);
            z_roots.push((x - d) / a2);
        }
        // Solve linear equation
        if m == 1 {
            z_roots.push(Complex64::new(coeffs[0].neg() / coeffs[1], 0f64));
        }
        Ok(z_roots)
    }

    fn degree(&self) -> usize {
        let mut d = self.len() - 1;
        while self[d] == 0.0 { d = d - 1; }
        d
    }

    fn off_low(&self) -> usize {
        let mut index = 0;
        while self[index] == 0.0 { index = index + 1 };
        index
    }

    fn laguerre(&self, mut z: Complex64) -> Complex64 {
        let n: usize = self.len() - 1;
        let max_iter: usize = 20;
        let mut k = 1;
        while k <= max_iter {
            let mut alpha = Complex64::new(self[n], 0.0);
            let mut beta = Complex64::new(0.0, 0.0);
            let mut gamma = Complex64::new(0.0, 0.0);

            for j in (0..n).rev() {
                gamma = (z * gamma) + beta;
                beta = (z * beta) + alpha;
                alpha = (z * alpha) + Complex64::new(self[j], 0.0);
            }

            if alpha.norm() <= 1e-14 { return z; }

            let ca = beta.neg() / alpha;
            let ca2 = ca * ca;
            let cb = ca2 - ((Complex64::new(2.0, 0.0) * gamma) / alpha);
            let c1 = (Complex64::new(((n-1) as f64), 0.0) * ((Complex64::new((n as f64), 0.0) * cb) - ca2)).sqrt();
            let cc1: Complex64 = ca + c1;
            let cc2: Complex64 = ca - c1;
            let mut cc;
            if cc1.norm() > cc2.norm() {
                cc = cc1
            } else {
                cc = cc2
            }
            cc = cc / Complex64::new(n as f64, 0.0);
            let c2 = cc.inv();
            z = z + c2;
            k = k + 1;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::complex::*;

    #[test]
    fn test_div_polynomial() {
        let exp_quo = vec![1.32, -0.8];
        let exp_rem = vec![-0.32];
        let mut a = vec![1.0f64, 2.5, -2.0];
        let b = vec![1.0f64, 2.5];
        {
            let rem = a.div_polynomial(b).unwrap();
            assert_eq!(rem.len(), exp_rem.len());
            for i in 0..exp_rem.len() {
                assert!((rem[i] - exp_rem[i]).abs() < 1e-10);
            }
        }
        assert_eq!(a.len(), exp_quo.len());
        for i in 0..exp_quo.len() {
            assert!((a[i] - exp_quo[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_degree() {
        let a: Vec<f64> = vec![3.0, 2.0, 4.0, 0.0, 0.0];
        assert_eq!(a.degree(), 2);
    }

    #[test]
    fn test_off_low() {
        let a: Vec<f64> = vec![0.0f64, 0.0, 3.0, 2.0, 4.0];
        assert_eq!(a.off_low(), 2);
    }

    #[test]
    fn test_laguerre() {
        let vec: Vec<f64> = vec![1.0, 2.5, 2.0, 3.0];
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
        let poly = vec![1.0, 2.5];
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
        let poly = vec![1.0, 2.5, -2.0];
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
        let coeffs: Vec<f64> = vec![1.0, -2.5, 2.0];
        let roots = coeffs.find_roots().unwrap();
        let roots_exp = vec![Complex64::new( 0.625, 0.33071891388307 ), Complex64::new( 0.625, -0.33071891388307 )];
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
        let lpc_exp: Vec<f64> = vec![1.0, 2.5, -2.0, -3.0];
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
}

