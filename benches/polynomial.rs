#![cfg(feature = "nightly")]
#![feature(test)]

extern crate vox_box;
extern crate num;

#[cfg(all(feature = "nightly", test))]
mod bench {
    extern crate test;

    use vox_box::*;
    use vox_box::polynomial::*;
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
}

