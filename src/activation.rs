use std::{
    cmp::Ordering::{Greater, Less},
    mem::MaybeUninit,
};

use rand::prelude::RngExt;

use crate::{layer::Layer, tensor::Vector};

pub trait FixedActivation {
    type Cache;

    fn train_fn<R: RngExt>(&self, value: f64, rng: &mut R) -> (f64, Self::Cache);
    fn predict_fn(&self, value: f64) -> f64;
    fn derivative(&self, cache: Self::Cache) -> f64;

    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        inputs.map(|val| self.predict_fn(val))
    }

    fn train<const N: usize, R: RngExt>(
        &self,
        inputs: Vector<f64, N>,
        rng: &mut R,
    ) -> (Vector<f64, N>, [Self::Cache; N]) {
        let mut outputs = [const { MaybeUninit::uninit() }; N];
        let mut cache = [const { MaybeUninit::uninit() }; N];

        for i in 0..N {
            let (each_value, each_cache) = self.train_fn(inputs[i], rng);
            outputs[i].write(each_value);
            cache[i].write(each_cache);
        }

        let output: MaybeUninit<[f64; N]> = outputs.into();
        let cache: MaybeUninit<[Self::Cache; N]> = cache.into();
        (unsafe { output.assume_init() }.into(), unsafe {
            cache.assume_init()
        })
    }

    fn backward<const N: usize>(
        &self,
        cache: [Self::Cache; N],
        gradient: Vector<f64, N>,
    ) -> Vector<f64, N> {
        let mut backprop = [const { MaybeUninit::uninit() }; N];
        let cache: [MaybeUninit<Self::Cache>; N] = MaybeUninit::new(cache).into();

        for i in 0..N {
            backprop[i]
                .write(self.derivative(unsafe { cache[i].assume_init_read() }) * gradient[i]);
        }

        let backprop: MaybeUninit<[f64; N]> = backprop.into();
        unsafe { backprop.assume_init() }.into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationLayer<T: FixedActivation>(T);

impl<T: FixedActivation> FixedActivation for ActivationLayer<T> {
    type Cache = T::Cache;

    fn train_fn<R: RngExt>(&self, value: f64, rng: &mut R) -> (f64, Self::Cache) {
        self.0.train_fn(value, rng)
    }

    fn predict_fn(&self, value: f64) -> f64 {
        self.0.predict_fn(value)
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        self.0.derivative(cache)
    }

    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        self.0.predict(inputs)
    }

    fn train<const N: usize, R: RngExt>(
        &self,
        inputs: Vector<f64, N>,
        rng: &mut R,
    ) -> (Vector<f64, N>, [Self::Cache; N]) {
        self.0.train(inputs, rng)
    }

    fn backward<const N: usize>(
        &self,
        cache: [Self::Cache; N],
        gradient: Vector<f64, N>,
    ) -> Vector<f64, N> {
        self.0.backward(cache, gradient)
    }
}

impl<T: FixedActivation> From<T> for ActivationLayer<T> {
    fn from(value: T) -> Self {
        ActivationLayer(value)
    }
}

impl<T: FixedActivation, const N: usize> Layer<N, N> for ActivationLayer<T> {
    type Cache = [T::Cache; N];
    type Data = ();

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, N>,
        rng: &mut R,
    ) -> (Vector<f64, N>, Self::Cache) {
        self.0.train(inputs, rng)
    }

    fn predict(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        self.0.predict(inputs)
    }

    fn loss(&self) -> f64 {
        0.
    }

    fn backward(
        &self,
        cache: Self::Cache,
        gradient: Vector<f64, N>,
    ) -> (Vector<f64, N>, Self::Data) {
        (self.0.backward(cache, gradient), ())
    }

    fn update(&mut self, _change: Self::Data) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearActivation;

impl FixedActivation for LinearActivation {
    type Cache = ();

    fn train_fn<R: RngExt>(&self, value: f64, _rng: &mut R) -> (f64, Self::Cache) {
        (value, ())
    }

    fn predict_fn(&self, value: f64) -> f64 {
        value
    }

    fn derivative(&self, _cache: Self::Cache) -> f64 {
        1.
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReluActivation;

impl FixedActivation for ReluActivation {
    type Cache = bool;

    fn train_fn<R: RngExt>(&self, value: f64, _rng: &mut R) -> (f64, Self::Cache) {
        let cache = value.is_sign_positive();

        (if cache { value } else { 0. }, cache)
    }

    fn predict_fn(&self, value: f64) -> f64 {
        value.max(0.)
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        if cache { 1. } else { 0. }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeakyreluActivation {
    coefficient: f64,
}

impl LeakyreluActivation {
    pub fn new(coefficient: f64) -> Option<Self> {
        if coefficient.partial_cmp(&1.)? != Less {
            return None;
        }
        Some(Self { coefficient })
    }
}

impl FixedActivation for LeakyreluActivation {
    type Cache = bool;

    fn train_fn<R: RngExt>(&self, value: f64, _rng: &mut R) -> (f64, Self::Cache) {
        let cache = value.is_sign_positive();

        (
            if cache {
                value
            } else {
                self.coefficient * value
            },
            cache,
        )
    }

    fn predict_fn(&self, value: f64) -> f64 {
        if value.is_sign_positive() {
            value
        } else {
            self.coefficient * value
        }
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        if cache { 1. } else { self.coefficient }
    }
}

// TODO: Prelu, use https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-prelu-with-keras.md

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RreluActivation {
    lower_bound: f64,
    upper_bound: f64,
}

impl RreluActivation {
    pub fn new(lower_bound: f64, upper_bound: f64) -> Option<Self> {
        if lower_bound.partial_cmp(&upper_bound)? != Less {
            return None;
        }
        Some(Self {
            lower_bound,
            upper_bound,
        })
    }
}

impl FixedActivation for RreluActivation {
    type Cache = f64;

    fn train_fn<R: RngExt>(&self, value: f64, rng: &mut R) -> (f64, Self::Cache) {
        let is_positive = value.is_sign_positive();

        if is_positive {
            (value, 1.)
        } else {
            let coefficient = rng.random_range(self.lower_bound..self.upper_bound);
            (coefficient * coefficient, coefficient)
        }
    }

    fn predict_fn(&self, value: f64) -> f64 {
        if value.is_sign_positive() {
            value
        } else {
            ((self.lower_bound + self.upper_bound) / 2.) * value
        }
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        cache
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DropoutActivation {
    rate: f64,
}

impl DropoutActivation {
    pub fn new(rate: f64) -> Option<Self> {
        if rate.partial_cmp(&0.)? != Greater || rate.partial_cmp(&1.)? != Less {
            return None;
        }
        Some(Self { rate })
    }
}

impl FixedActivation for DropoutActivation {
    type Cache = bool;

    fn train_fn<R: RngExt>(&self, value: f64, rng: &mut R) -> (f64, Self::Cache) {
        let result = rng.random_range(0. ..1.) > self.rate;
        (if result { value / (1. - self.rate) } else { 0. }, result)
    }

    fn predict_fn(&self, value: f64) -> f64 {
        value
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        if cache { 1. } else { 0. }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SigmoidActivation;

impl FixedActivation for SigmoidActivation {
    type Cache = f64;

    fn train_fn<R: RngExt>(&self, value: f64, _rng: &mut R) -> (f64, Self::Cache) {
        let result = 1. / (1. + (-value).exp());
        (result, result)
    }

    fn predict_fn(&self, value: f64) -> f64 {
        1. / (1. + (-value).exp())
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        cache * (1. - cache)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TanhActivation;

impl FixedActivation for TanhActivation {
    type Cache = f64;

    fn train_fn<R: RngExt>(&self, value: f64, _rng: &mut R) -> (f64, Self::Cache) {
        let result = value.tanh();
        (result, result)
    }

    fn predict_fn(&self, value: f64) -> f64 {
        value.tanh()
    }

    fn derivative(&self, cache: Self::Cache) -> f64 {
        1. - cache * cache
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Softmax;

impl<const N: usize> Layer<N, N> for Softmax {
    type Cache = Vector<f64, N>;
    type Data = ();

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, N>,
        _rng: &mut R,
    ) -> (Vector<f64, N>, Self::Cache) {
        let result = self.predict(inputs);
        (result, result)
    }

    fn predict(&self, mut inputs: Vector<f64, N>) -> Vector<f64, N> {
        let max = inputs.into_iter().reduce(f64::max).unwrap_or(0.);

        for input in inputs.iter_mut() {
            *input = (*input - max).exp();
        }
        let denominator: f64 = inputs.iter().sum();
        for input in inputs.iter_mut() {
            *input /= denominator;
        }

        inputs
    }

    fn loss(&self) -> f64 {
        0.
    }

    fn backward(
        &self,
        cache: Self::Cache,
        gradient: Vector<f64, N>,
    ) -> (Vector<f64, N>, Self::Data) {
        let gradient: Vector<_, N> = gradient;
        let mut jacobian = -gradient.outer(&gradient);
        for (i, each_grad) in gradient.iter().enumerate() {
            jacobian.content_mut()[i][i] -= *each_grad;
        }

        (jacobian.dot_vec(&cache), ())
    }

    fn update(&mut self, _change: Self::Data) {}
}
