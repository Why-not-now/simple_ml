use std::iter::{Chain, empty};
use std::{array, iter::Empty};

use rand::prelude::RngExt;
use rand_distr::{Normal, StandardNormal};

use crate::tensor::{Matrix, Vector};

pub trait Full<const I: usize, const O: usize> {
    type Loss;

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, I>,
        expected: Vector<f64, O>,
        rng: &mut R,
    ) -> (Vector<f64, I>, Self::Loss);
    fn predict(&self, inputs: Vector<f64, I>) -> Vector<f64, O>;
    fn update(&mut self, loss: Self::Loss);
}

pub trait Layer<const I: usize, const O: usize> {
    type Cache;
    type Data: LayerData;

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, I>,
        rng: &mut R,
    ) -> (Vector<f64, O>, Self::Cache);
    fn predict(&self, inputs: Vector<f64, I>) -> Vector<f64, O>;
    fn loss(&self) -> f64;
    fn backward(
        &self,
        cache: Self::Cache,
        gradient: Vector<f64, O>,
    ) -> (Vector<f64, I>, Self::Data);
    fn update(&mut self, change: Self::Data);

    fn merge<L2: Layer<O, O2>, const O2: usize>(self, other: L2) -> MergedLayer<I, O, O2, Self, L2>
    where
        Self: std::marker::Sized,
    {
        MergedLayer {
            left: self,
            right: other,
        }
    }
}

pub trait LayerData {
    type Iter<'a>: Iterator<Item = &'a f64>
    where
        Self: 'a;
    type IterMut<'a>: Iterator<Item = &'a mut f64>
    where
        Self: 'a;

    fn zeroes() -> Self;
    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn update_each<F>(&mut self, mut update_fn: F)
    where
        F: FnMut(f64) -> f64,
    {
        for mut_elem in self.iter_mut() {
            *mut_elem = update_fn(*mut_elem);
        }
    }

    fn merge_each<F>(&mut self, other: &Self, mut update_fn: F)
    where
        F: FnMut(f64, f64) -> f64,
    {
        for (mut_elem, other_elem) in self.iter_mut().zip(other.iter()) {
            *mut_elem = update_fn(*mut_elem, *other_elem);
        }
    }

    fn merge<L2: LayerData>(self, other: L2) -> MergedData<Self, L2>
    where
        Self: std::marker::Sized,
    {
        MergedData {
            left: self,
            right: other,
        }
    }
}

impl LayerData for () {
    type Iter<'a>
        = Empty<&'a f64>
    where
        Self: 'a;
    type IterMut<'a>
        = Empty<&'a mut f64>
    where
        Self: 'a;

    fn zeroes() -> Self {}

    fn iter(&self) -> Self::Iter<'_> {
        empty()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergedData<A, B> {
    left: A,
    right: B,
}

impl<A, B> MergedData<A, B>{
    pub fn destruct(self) -> (A, B) {
        (self.left, self.right)
    }
}

impl<A: LayerData, B: LayerData> LayerData for MergedData<A, B> {
    type Iter<'a>
        = Chain<A::Iter<'a>, B::Iter<'a>>
    where
        Self: 'a;
    type IterMut<'a>
        = Chain<A::IterMut<'a>, B::IterMut<'a>>
    where
        Self: 'a;

    fn zeroes() -> Self {
        MergedData {
            left: A::zeroes(),
            right: B::zeroes(),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.left.iter().chain(self.right.iter())
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.left.iter_mut().chain(self.right.iter_mut())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergedLayer<
    const I: usize,
    const M: usize,
    const O: usize,
    A: Layer<I, M>,
    B: Layer<M, O>,
> {
    left: A,
    right: B,
}

impl<const I: usize, const M: usize, const O: usize, A: Layer<I, M>, B: Layer<M, O>> Layer<I, O>
    for MergedLayer<I, M, O, A, B>
{
    type Cache = MergedData<A::Cache, B::Cache>;
    type Data = MergedData<A::Data, B::Data>;

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, I>,
        rng: &mut R,
    ) -> (Vector<f64, O>, Self::Cache) {
        let (middle, left_cache) = self.left.train(inputs, rng);
        let (output, right_cache) = self.right.train(middle, rng);

        (
            output,
            MergedData {
                left: left_cache,
                right: right_cache,
            },
        )
    }

    fn predict(&self, inputs: Vector<f64, I>) -> Vector<f64, O> {
        self.right.predict(self.left.predict(inputs))
    }

    fn loss(&self) -> f64 {
        self.left.loss() + self.right.loss()
    }

    fn backward(
        &self,
        cache: Self::Cache,
        gradient: Vector<f64, O>,
    ) -> (Vector<f64, I>, Self::Data) {
        let (delta_middle, right_cache) = self.right.backward(cache.right, gradient);
        let (delta_final, left_cache) = self.left.backward(cache.left, delta_middle);

        (
            delta_final,
            MergedData {
                left: left_cache,
                right: right_cache,
            },
        )
    }

    fn update(&mut self, change: Self::Data) {
        self.left.update(change.left);
        self.right.update(change.right);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseLayerData<const I: usize, const O: usize> {
    weights: Matrix<f64, O, I>,
    biases: Vector<f64, O>,
}

impl<const I: usize, const O: usize> LayerData for DenseLayerData<I, O> {
    type Iter<'a> = Chain<
        <&'a Matrix<f64, O, I> as IntoIterator>::IntoIter,
        <&'a Vector<f64, O> as IntoIterator>::IntoIter,
    >;
    type IterMut<'a> = Chain<
        <&'a mut Matrix<f64, O, I> as IntoIterator>::IntoIter,
        <&'a mut Vector<f64, O> as IntoIterator>::IntoIter,
    >;

    fn zeroes() -> Self {
        DenseLayerData {
            weights: Matrix::new([[0.; I]; O]),
            biases: Vector::new([0.; O]),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.weights.iter().chain(self.biases.iter())
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.weights.iter_mut().chain(self.biases.iter_mut())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseLayer<const I: usize, const O: usize> {
    weights: Matrix<f64, O, I>,
    biases: Vector<f64, O>,
    w_l1_reg: f64,
    w_l2_reg: f64,
    b_l1_reg: f64,
    b_l2_reg: f64,
}

impl<const I: usize, const O: usize> DenseLayer<I, O> {
    pub fn uniform_standard_weights<R: RngExt>(rng: &mut R) -> Matrix<f64, O, I> {
        Matrix::new(array::from_fn(|_row| {
            array::from_fn(|_col| rng.random_range(-1. ..1.))
        }))
    }

    pub fn uniform_xavier_weights<R: RngExt>(rng: &mut R) -> Matrix<f64, O, I> {
        let bound = (6. / (I + O) as f64).sqrt();
        Matrix::new(array::from_fn(|_row| {
            array::from_fn(|_col| rng.random_range(-bound..bound))
        }))
    }

    pub fn normal_standard_weights<R: RngExt>(rng: &mut R) -> Matrix<f64, O, I> {
        Matrix::new(array::from_fn(|_row| {
            array::from_fn(|_col| rng.sample(StandardNormal))
        }))
    }

    pub fn normal_xavier_weights<R: RngExt>(rng: &mut R) -> Matrix<f64, O, I> {
        let std_dev = (2. / (I + O) as f64).sqrt();
        let dist =
            Normal::new(0., std_dev).expect("Xavier distribution normal implementation is invalid");

        Matrix::new(array::from_fn(|_row| {
            array::from_fn(|_col| rng.sample(dist))
        }))
    }

    pub fn zero_bias() -> Vector<f64, O> {
        Vector::new([0.; O])
    }

    pub fn new(
        weights: Matrix<f64, O, I>,
        biases: Vector<f64, O>,
        w_l1_reg: f64,
        w_l2_reg: f64,
        b_l1_reg: f64,
        b_l2_reg: f64,
    ) -> Self {
        Self {
            weights,
            biases,
            w_l1_reg,
            w_l2_reg,
            b_l1_reg,
            b_l2_reg,
        }
    }

    pub fn new_default<R: RngExt>(rng: &mut R
    ) -> Self {
        Self {
            weights: Self::normal_xavier_weights(rng),
            biases: Self::zero_bias(),
            w_l1_reg: 0.,
            w_l2_reg: 5e-4,
            b_l1_reg: 0.,
            b_l2_reg: 5e-4,
        }
    }
}

impl<const I: usize, const O: usize> Layer<I, O> for DenseLayer<I, O> {
    type Cache = Vector<f64, I>;
    type Data = DenseLayerData<I, O>;

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, I>,
        _rng: &mut R,
    ) -> (Vector<f64, O>, Self::Cache) {
        (self.weights.dot_vec(&inputs) + self.biases, inputs)
    }

    fn predict(&self, inputs: Vector<f64, I>) -> Vector<f64, O> {
        self.weights.dot_vec(&inputs) + self.biases
    }

    fn loss(&self) -> f64 {
        let mut loss = 0.;

        if self.w_l1_reg != 0. {
            loss += self.weights.iter().sum::<f64>() * self.w_l1_reg;
        }
        if self.b_l1_reg != 0. {
            loss += self.biases.iter().sum::<f64>() * self.w_l1_reg;
        }
        if self.w_l2_reg != 0. {
            loss += self.weights.iter().map(|weight| weight.powi(2)).sum::<f64>() * self.w_l1_reg;
        }
        if self.b_l2_reg != 0. {
            loss += self.weights.iter().map(|bias| bias.powi(2)).sum::<f64>() * self.w_l1_reg;
        }

        loss
    }

    fn backward(
        &self,
        cache: Self::Cache,
        gradient: Vector<f64, O>,
    ) -> (Vector<f64, I>, Self::Data) {
        let mut biases = gradient;
        let mut weights = biases.outer(&cache);
        let backprop = biases.dot_mat(&self.weights);

        if self.w_l1_reg != 0. {
            let mut w_reg = self.weights;
            for weight in w_reg.iter_mut() {
                *weight = if weight.is_sign_positive() {
                    *weight
                } else {
                    -*weight
                };
            }
            w_reg.mul_left(self.w_l1_reg);
            weights -= w_reg;
        }
        if self.b_l1_reg != 0. {
            let mut b_reg = self.biases;
            for bias in b_reg.iter_mut() {
                *bias = if bias.is_sign_positive() {
                    *bias
                } else {
                    -*bias
                };
            }
            b_reg.mul_left(self.b_l1_reg);
            biases -= b_reg;
        }
        if self.w_l2_reg != 0. {
            let mut w_reg = self.weights;
            w_reg.mul_right(2. * self.w_l2_reg);
            weights -= w_reg;
        }
        if self.b_l2_reg != 0. {
            let mut b_reg = self.biases;
            b_reg.mul_right(2. * self.b_l2_reg);
            b_reg.mul_left(self.b_l2_reg);
            biases -= b_reg;
        }

        let data = DenseLayerData { weights, biases };
        (backprop, data)
    }

    fn update(&mut self, change: Self::Data) {
        self.weights -= change.weights;
        self.biases -= change.biases;
    }
}
