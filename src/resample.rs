extern crate num;

use num::Float;
use num::traits::FromPrimitive;
use std::ops::{Add, Sub, Mul};

pub trait Resample<T> {
    fn resample_linear(self, usize) -> Self;
}

impl<T: Float + FromPrimitive> Resample<T> for Vec<T> {
    fn resample_linear(self, new_size: usize) -> Self {
        let mut resampled = Vec::<T>::with_capacity(new_size);
        for i in 0..new_size {
            let phase = (i as f64) / ((new_size-1) as f64);
            let index = phase * ((self.len()-1) as f64);
            let a = self[index.floor() as usize];
            let b = self[index.ceil() as usize];
            let t = T::from_f64(index - index.floor()).unwrap();
            resampled.push(a + (b - a) * t);
        }
        resampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_vec32() {
        let upsamp: Vec<f32> = vec![1f32, 2.0, 3.0, 4.0];
        let exp: Vec<f32> = vec![1f32, 4.0];
        let res: Vec<f32> = upsamp.resample_linear(2);
        assert_eq!(res, exp);
    }

    #[test]
    fn test_resample_vec64() {
        let upsamp: Vec<f64> = vec![1f64, 2.0, 3.0, 4.0];
        let exp: Vec<f64> = vec![1f64, 2.5, 4.0];
        let res: Vec<f64> = upsamp.resample_linear(3);
        assert_eq!(res, exp);
    }
}
