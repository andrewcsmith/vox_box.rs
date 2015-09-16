extern crate num;
extern crate sample;

pub use num::complex::*;
use num::Float;
use num::traits::FromPrimitive;
use std::ops::{Add, Sub};
use sample::Sample;

pub trait SquareRoot<T> {
    fn sqrt(&self) -> Complex<T>;
}

// http://math.stackexchange.com/questions/44406/how-do-i-get-the-square-root-of-a-complex-number
impl<T: Float + FromPrimitive> SquareRoot<T> for Complex<T> {
    fn sqrt(&self) -> Complex<T> {
        let (r, theta) = self.to_polar();
        Complex::<T>::from_polar(&r.sqrt(), &(theta / T::from_f32(2.0f32).unwrap()))
    }
}

pub trait ToComplex<T> {
    fn to_complex(self) -> Complex<T>;
}

impl ToComplex<f32> for f32 {
    fn to_complex(self) -> Complex<f32> {
        Complex::<f32>::new(self, 0f32)
    }
}

impl ToComplex<f64> for f64 {
    fn to_complex(self) -> Complex<f64> {
        Complex::<f64>::new(self, 0f64)
    }
}

pub trait ToComplexVec<T> {
    fn to_complex_vec(self) -> Vec<Complex<T>>;
}

impl<T: ToComplex<T> + Copy> ToComplexVec<T> for Vec<T> {
    fn to_complex_vec(self) -> Vec<Complex<T>> {
        self.iter().map(|v| v.to_complex()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_root() {
        let delta = 0.000001;
        let c = Complex::<f64>::new(2.0, -3.0);
        let root = Complex::<f64>::new(1.6741492280355, -0.89597747612984);

        let squared = (&root * &root) - &c;
        assert!(squared.re.abs() < delta);
        assert!(squared.im.abs() < delta);

        let result: Complex<f64> = c.sqrt() - &root;
        println!("Complex is {:?}", result);
        assert!(result.re.abs() < delta);
        assert!(result.im.abs() < delta);
    }

    #[test]
    fn test_negative_complex_roots() {
        let c = Complex::<f64>::new(-9.0, 0.0);
        let exp_root = Complex::<f64>::new(0.0, 3.0);
        let root = c.sqrt();
        println!("Root found was: {:?}", root);
        assert!((root.re - exp_root.re).abs() < 1e-12);
        assert!((root.im - exp_root.im).abs() < 1e-12);
    }

    #[test]
    fn test_to_complex() {
        let c = 3f64;
        assert_eq!(c.to_complex(), Complex::<f64>::new(3f64, 0f64));
    }

    #[test]
    fn test_to_complex_vec() {
        let c = vec![3f64, 2f64];
        assert_eq!(c.to_complex_vec(), vec![Complex::<f64>::new(3f64, 0f64), Complex::<f64>::new(2f64, 0f64)]);
    }
}
