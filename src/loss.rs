use core::slice;

use rand::RngExt;

use crate::activation::Softmax;
use crate::layer::{Layer, LayerData};
use crate::tensor::Vector;

type LossCacheData<T, const N: usize> = <<T as Loss>::Cache<N> as LossCache<N>>::Data;

pub trait Loss {
    type Cache<const N: usize>: LossCache<N>;

    fn train<R: RngExt, const N: usize>(
        &self,
        inputs: Vector<f64, N>,
        rng: &mut R,
    ) -> Self::Cache<N>;
    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N>;
    fn inner_loss(&self) -> f64;
    fn backward<const N: usize>(
        &self,
        cache: Self::Cache<N>,
        expected: Vector<f64, N>,
    ) -> (Vector<f64, N>, LossCacheData<Self, N>);
    fn update<const N: usize>(&mut self, change: LossCacheData<Self, N>);
}

pub trait LossCache<const N: usize> {
    type Data: LayerData;

    fn predictions(&self) -> Vector<f64, N>;
    fn loss(&self, expected: Vector<f64, N>) -> f64;
    fn into_data(self, expected: Vector<f64, N>) -> Self::Data;
}

pub struct LossLayer<T: Loss>(T);

impl<T: Loss> Loss for LossLayer<T> {
    type Cache<const N: usize> = T::Cache<N>;

    fn train<R: RngExt, const N: usize>(
        &self,
        inputs: Vector<f64, N>,
        rng: &mut R,
    ) -> Self::Cache<N> {
        self.0.train(inputs, rng)
    }

    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        self.0.predict(inputs)
    }

    fn inner_loss(&self) -> f64 {
        self.0.inner_loss()
    }

    fn backward<const N: usize>(
        &self,
        cache: Self::Cache<N>,
        expected: Vector<f64, N>,
    ) -> (Vector<f64, N>, LossCacheData<T, N>) {
        self.0.backward(cache, expected)
    }

    fn update<const N: usize>(&mut self, change: LossCacheData<T, N>) {
        self.0.update(change);
    }
}

impl<T: Loss> From<T> for LossLayer<T> {
    fn from(value: T) -> Self {
        LossLayer(value)
    }
}

impl<T: Loss, const N: usize> Layer<N, N> for LossLayer<T> {
    type Cache = T::Cache<N>;
    type Data = LossCacheData<T, N>;

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, N>,
        rng: &mut R,
    ) -> (Vector<f64, N>, Self::Cache) {
        let loss = self.0.train(inputs, rng);
        (loss.predictions(), loss)
    }

    fn predict(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        self.0.predict(inputs)
    }

    fn loss(&self) -> f64 {
        self.inner_loss()
    }

    fn backward(
        &self,
        cache: Self::Cache,
        gradient: Vector<f64, N>,
    ) -> (Vector<f64, N>, Self::Data) {
        self.0.backward(cache, gradient)
    }

    fn update(&mut self, change: Self::Data) {
        self.0.update(change);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoricalLoss;

impl Loss for CategoricalLoss {
    type Cache<const N: usize> = CategoricalLossCache<N>;

    fn train<R: RngExt, const N: usize>(
        &self,
        inputs: Vector<f64, N>,
        _rng: &mut R,
    ) -> Self::Cache<N> {
        CategoricalLossCache {
            predictions: inputs,
        }
    }

    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        inputs
    }

    fn inner_loss(&self) -> f64 {
        0.
    }

    fn backward<const N: usize>(
        &self,
        cache: Self::Cache<N>,
        expected: Vector<f64, N>,
    ) -> (Vector<f64, N>, LossCacheData<Self, N>) {
        ((-(cache.predictions / expected)), cache.into_data(expected))
    }

    fn update<const N: usize>(&mut self, _change: LossCacheData<Self, N>) {}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CategoricalLossCache<const N: usize> {
    predictions: Vector<f64, N>,
}

impl<const N: usize> CategoricalLossCache<N> {
    pub fn loss_fast(&self, expected_index: usize) -> f64 {
        let Some(actual) = self.predictions.get(expected_index) else {
            return f64::NAN;
        };

        -actual.max(f64::EPSILON).ln()
    }
}

impl<const N: usize> LossCache<N> for CategoricalLossCache<N> {
    type Data = ();

    fn predictions(&self) -> Vector<f64, N> {
        self.predictions
    }

    fn loss(&self, expected: Vector<f64, N>) -> f64 {
        (self.predictions)
            .iter()
            .zip(expected.iter())
            .map(|(pred, expect)| -expect * pred.max(f64::EPSILON).ln())
            .sum()
    }

    fn into_data(self, _expected: Vector<f64, N>) -> Self::Data {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxCategoricalLoss;

impl Loss for SoftmaxCategoricalLoss {
    type Cache<const N: usize> = CategoricalLossCache<N>;

    fn train<R: RngExt, const N: usize>(
        &self,
        inputs: Vector<f64, N>,
        _rng: &mut R,
    ) -> Self::Cache<N> {
        CategoricalLossCache {
            predictions: Softmax.predict(inputs),
        }
    }

    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        Softmax.predict(inputs)
    }

    fn inner_loss(&self) -> f64 {
        0.
    }

    fn backward<const N: usize>(
        &self,
        cache: Self::Cache<N>,
        expected: Vector<f64, N>,
    ) -> (Vector<f64, N>, LossCacheData<Self, N>) {
        (cache.predictions - expected, cache.into_data(expected))
    }

    fn update<const N: usize>(&mut self, _change: LossCacheData<Self, N>) {}
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JoinLossCache<const N: usize> {
    predictions: Vector<f64, N>,
}

impl<const N: usize> LossCache<N> for JoinLossCache<N> {
    type Data = Gradient<N>;

    fn predictions(&self) -> Vector<f64, N> {
        self.predictions
    }

    fn loss(&self, _expected: Vector<f64, N>) -> f64 {
        0.
    }

    fn into_data(self, expected: Vector<f64, N>) -> Self::Data {
        Gradient {
            gradients: expected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient<const N: usize> {
    gradients: Vector<f64, N>,
}

impl<const N: usize> Gradient<N> {
    pub fn gradients(&self) -> Vector<f64, N> {
        self.gradients
    }
}

impl<const N: usize> From<Vector<f64, N>> for Gradient<N> {
    fn from(value: Vector<f64, N>) -> Self {
        Gradient { gradients: value }
    }
}

impl<const N: usize> From<Gradient<N>> for Vector<f64, N> {
    fn from(value: Gradient<N>) -> Self {
        value.gradients
    }
}

impl<const N: usize> LayerData for Gradient<N> {
    type Iter<'a>
        = slice::Iter<'a, f64>
    where
        Self: 'a;
    type IterMut<'a>
        = slice::IterMut<'a, f64>
    where
        Self: 'a;

    fn zeroes() -> Self {
        Gradient {
            gradients: Vector::new([0.; N]),
        }
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.gradients.iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.gradients.iter_mut()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinLoss;

impl Loss for JoinLoss {
    type Cache<const N: usize> = JoinLossCache<N>;

    fn train<R: RngExt, const N: usize>(
        &self,
        inputs: Vector<f64, N>,
        _rng: &mut R,
    ) -> Self::Cache<N> {
        JoinLossCache {
            predictions: inputs,
        }
    }

    fn predict<const N: usize>(&self, inputs: Vector<f64, N>) -> Vector<f64, N> {
        inputs
    }

    fn inner_loss(&self) -> f64 {
        0.
    }

    fn backward<const N: usize>(
        &self,
        _cache: Self::Cache<N>,
        expected: Vector<f64, N>,
    ) -> (Vector<f64, N>, LossCacheData<Self, N>) {
        (
            expected,
            Gradient {
                gradients: expected,
            },
        )
    }

    fn update<const N: usize>(&mut self, _change: LossCacheData<Self, N>) {}
}
