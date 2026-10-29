use rand::RngExt;
use rand_distr::StandardNormal;

use crate::tensor::Vector;

pub trait Data<const N: usize> {
    fn vector(&self) -> Vector<f64, N>;
}

pub trait DataClass<const N: usize>: Data<N> {
    fn class(&self) -> u8;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SpiralData {
    x: f64,
    y: f64,
    class: u8,
}

impl Data<2> for SpiralData {
    fn vector(&self) -> Vector<f64, 2> {
        [self.x, self.y].into()
    }
}

impl DataClass<2> for SpiralData {
    fn class(&self) -> u8 {
        self.class
    }
}

pub fn spiral_data<R: RngExt>(samples: u32, classes: u8, rng: &mut R) -> Vec<SpiralData> {
    let mut result = vec![];
    for class in 0..classes {
        for i in 0..samples {
            let r = f64::from(i) / f64::from(samples - 1);
            let mut t = (r + f64::from(class)) * 10.;
            let t_rand: f64 = rng.sample(StandardNormal);
            t += 0.5 * t_rand;

            let (mut x, mut y) = t.sin_cos();
            x *= r;
            y *= r;
            result.push(SpiralData { x, y, class });
        }
    }

    result
}
