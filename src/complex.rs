extern crate num;
pub use num::complex::{Complex64};

pub trait SquareRoot<T> {
    fn sqrt(&self) -> T;
}

// http://math.stackexchange.com/questions/44406/how-do-i-get-the-square-root-of-a-complex-number
impl SquareRoot<Complex64> for Complex64 {
    fn sqrt(&self) -> Complex64 {
        let c = self + Complex64::new(self.norm(), 0.0);
        Complex64::new(self.norm().sqrt(), 0.0) * c / Complex64::new(c.norm(), 0.0)
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
}
