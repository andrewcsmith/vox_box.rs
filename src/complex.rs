extern crate num;
pub use num::complex::{Complex, Complex32, Complex64};
pub use std::ops::{Add, Sub};

pub trait SquareRoot<T> {
    fn sqrt(&self) -> T;
}

// http://math.stackexchange.com/questions/44406/how-do-i-get-the-square-root-of-a-complex-number
impl SquareRoot<Complex64> for Complex64 {
    fn sqrt(&self) -> Complex64 {
        let (r, theta) = self.to_polar();
        Complex64::from_polar(&r.sqrt(), &(theta / 2.0))
    }
}

pub trait ToComplex<T> {
    fn to_complex(self) -> Complex<T>;
}

pub trait ToComplexVec<T> {
    fn to_complex_vec(self) -> Vec<Complex<T>>;
}

impl ToComplex<f64> for Complex<f64> {
    fn to_complex(self) -> Complex<f64> {
        self
    }
}

impl ToComplex<f64> for f64 {
    fn to_complex(self) -> Complex<f64> {
        Complex::<f64>::new(self, 0f64)
    }
}

impl ToComplexVec<f64> for Vec<f64> {
    fn to_complex_vec(self) -> Vec<Complex<f64>> {
        self.iter().map(move |r| {
            Complex::<f64>::new(*r, 0f64)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_root() {
        let delta = 0.000001;
        let c = Complex64::new(2.0, -3.0);
        let root = Complex64::new(1.6741492280355, -0.89597747612984);

        let squared = (&root * &root) - &c;
        assert!(squared.re.abs() < delta);
        assert!(squared.im.abs() < delta);

        let result: Complex64 = c.sqrt() - &root;
        println!("Complex is {:?}", result);
        assert!(result.re.abs() < delta);
        assert!(result.im.abs() < delta);
    }

    #[test]
    fn test_negative_complex_roots() {
        let c = Complex64::new(-9.0, 0.0);
        let exp_root = Complex64::new(0.0, 3.0);
        let root = c.sqrt();
        println!("Root found was: {:?}", root);
        assert!((root.re - exp_root.re).abs() < 1e-12);
        assert!((root.im - exp_root.im).abs() < 1e-12);
    }

    #[test]
    fn test_to_complex() {
        let c = 3f64;
        assert_eq!(c.to_complex(), Complex64::new(3f64, 0f64));
    }

    #[test]
    fn test_to_complex_vec() {
        let c = vec![3f64, 2f64];
        assert_eq!(c.to_complex_vec(), vec![Complex64::new(3f64, 0f64), Complex64::new(2f64, 0f64)]);
    }
}
