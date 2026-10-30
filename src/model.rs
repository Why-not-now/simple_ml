use rand::prelude::*;

use crate::layer::{Full, Layer, LayerData, MergedData,};
use crate::loss::{Loss, LossCache};
use crate::optimiser::{Optimiser, OptimiserData};
use crate::tensor::Vector;

type FullLossData<L, F, const I: usize, const O: usize> =
    MergedData<<L as Layer<I, O>>::Data, <<F as Loss>::Cache<O> as LossCache<O>>::Data>;

// #[derive(Debug, Clone, Copy)]
// pub struct ModelEvaluationParams {
//     loss: bool,
//     accuracy: bool,
// }

// impl ModelEvaluationParams {
//     pub fn new(loss: bool, accuracy: bool) -> Self {
//         Self { loss, accuracy }
//     }
// }

// #[derive(Debug, Clone, Copy)]
// pub struct ModelEvaluation {
//     loss: Option<f64>,
//     accuracy: Option<f64>,
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model<L, F, Q, const I: usize, const O: usize>
where
    L: Layer<I, O>,
    F: Loss,
    Q: Optimiser,
{
    layer: L,
    loss: F,
    optimiser: Q,
    cache: Q::Data<FullLossData<L, F, I, O>>,
}

impl<L, F, Q, const I: usize, const O: usize> Full<I, O> for Model<L, F, Q, I, O>
where
    L: Layer<I, O>,
    F: Loss,
    Q: Optimiser,
{
    type Loss = FullLossData<L, F, I, O>;

    fn train<R: RngExt>(
        &self,
        inputs: Vector<f64, I>,
        expected: Vector<f64, O>,
        rng: &mut R,
    ) -> (Vector<f64, I>, Self::Loss) {
        let (last_layer, layer_cache) = self.layer.train(inputs, rng);
        let loss_cache = self.loss.train(last_layer, rng);
        let (gradient, loss_data) = self.loss.backward(loss_cache, expected);
        let (gradient, layer_data) = self.layer.backward(layer_cache, gradient);

        (gradient, layer_data.merge(loss_data))
    }

    fn predict(&self, inputs: Vector<f64, I>) -> Vector<f64, O> {
        self.loss.predict(self.layer.predict(inputs))
    }

    fn update(&mut self, loss: Self::Loss) {
        let loss = self.optimiser.update(loss, &mut self.cache);
        let (layer_data, loss_data) = loss.destruct();
        self.layer.update(layer_data);
        self.loss.update(loss_data);
        self.optimiser.finish_iteration();
    }
}

impl<L, F, Q, const I: usize, const O: usize> Model<L, F, Q, I, O>
where
    L: Layer<I, O>,
    F: Loss,
    Q: Optimiser,
{
    pub fn new(layer: L, loss: F, optimiser: Q) -> Self {
        let cache = <Q::Data<_>>::empty();
        Self {
            layer,
            loss,
            optimiser,
            cache,
        }
    }

    pub fn train_batch_full<R: RngExt>(
        &mut self,
        inputs: &[Vector<f64, I>],
        expected: &[Vector<f64, O>],
        rng: &mut R,
    ) -> Vec<Vector<f64, I>> {
        assert!(
            inputs.len() == expected.len(),
            "inputs batch have different length than expected batch"
        );
        assert!(!inputs.is_empty(), "inputs batch should not be empty");
        let mut gradients = Vec::with_capacity(inputs.len());
        let length = inputs.len() as f64;

        let train_once = |(input, expect): (&_, &_)| {
            let (last_layer, layer_cache) = self.layer.train(*input, rng);
            let loss_cache = self.loss.train(last_layer, rng);
            let (gradient, mut loss_data) = self.loss.backward(loss_cache, *expect);
            gradient.divide(&length);
            loss_data.update_each(|val| val / length);
            let (gradient, layer_data) = self.layer.backward(layer_cache, gradient);
            gradients.push(gradient);
            layer_data.merge(loss_data)
        };

        let loss_data = inputs
            .iter()
            .zip(expected.iter())
            .map(train_once)
            .reduce(|mut left, right| {
                left.merge_each(&right, |left, right| left + right);
                left
            })
            .expect("inputs batch should not be empty");
        // TODO: use tree_reduce, currently stack is overflowing

        self.update(loss_data);

        gradients
    }
}

// impl<L, Q, const I: usize, const O: usize> Model<L, JoinLoss, Q, I, O>
// where
//     L: Layer<I, O>,
//     Q: Optimiser,
// {
//     pub fn forwards_batch<R: RngExt>(
//         &mut self,
//         inputs: &[Vector<f64, I>],
//         expected: &[Vector<f64, O>],
//         rng: &mut R,
//     ) -> (Vec<Vector<f64, I>>, Vec<<Self as Full<I, O>>::Loss>) {
//         assert!(
//             inputs.len() == expected.len(),
//             "inputs batch have different length than expected batch"
//         );
//         assert!(inputs.is_empty(), "inputs batch should not be empty");
//         let mut gradients = Vec::with_capacity(inputs.len());
//         let mut cache = Vec::with
//         let length = inputs.len() as f64;

//         let train_once = |(input, expect): (&Vector<f64, I>, &Vector<f64, O>)| {
//             let (last_layer, layer_cache) = self.layer.train(*input, rng);
//             let loss_cache = self.loss.train(last_layer, rng);
//             let (gradient, mut loss_data) = self.loss.backward(loss_cache, *expect);
//             gradient.divide(&length);
//             loss_data.update_each(|val| val / length);
//             let (gradient, layer_data) = self.layer.backward(layer_cache, gradient);
//             gradients.push(gradient);
//             layer_data.merge(loss_data)
//         };

//         let loss_data = inputs
//             .iter()
//             .zip(expected.iter())
//             .map(train_once)
//             .tree_reduce(|mut left, right| {
//                 left.merge_each(&right, |left, right| left + right);
//                 left
//             })
//             .expect("inputs batch should not be empty");

//         self.update(loss_data);

//         gradients
//     }

//     pub fn backwards_batch<R: RngExt>(
//         &mut self,
//         inputs: &[Vector<f64, I>],
//         expected: &[Vector<f64, O>],
//         rng: &mut R,
//     ) -> Vec<Vector<f64, I>> {
//         assert!(
//             inputs.len() == expected.len(),
//             "inputs batch have different length than expected batch"
//         );
//         assert!(inputs.is_empty(), "inputs batch should not be empty");
//         let mut gradients = Vec::with_capacity(inputs.len());
//         let length = inputs.len() as f64;

//         let train_once = |(input, expect): (&Vector<f64, I>, &Vector<f64, O>)| {
//             let (last_layer, layer_cache) = self.layer.train(*input, rng);
//             let loss_cache = self.loss.train(last_layer, rng);
//             let (gradient, mut loss_data) = self.loss.backward(loss_cache, *expect);
//             gradient.divide(&length);
//             loss_data.update_each(|val| val / length);
//             let (gradient, layer_data) = self.layer.backward(layer_cache, gradient);
//             gradients.push(gradient);
//             layer_data.merge(loss_data)
//         };

//         let loss_data = inputs
//             .iter()
//             .zip(expected.iter())
//             .map(train_once)
//             .tree_reduce(|mut left, right| {
//                 left.merge_each(&right, |left, right| left + right);
//                 left
//             })
//             .expect("inputs batch should not be empty");

//         self.update(loss_data);

//         gradients
//     }
// }

// impl<L1, Q, const I: usize, const M: usize> Model<L1, JoinLoss, Q, I, M>
// where
//     L1: Layer<I, M>,
//     Q: Optimiser,
// {
//     pub fn merge<L2: Layer<M, O>, F: Loss, const O: usize>(self, other: Model<L2, F, Q, M, O>) -> Model<MergedLayer<I, M, O, L1, L2>, F, Q, I, O>{
//         return Model { layer: self.layer.merge(other.layer), loss: other.loss, optimiser: other.optimiser, cache: Q::merge(self, other.cache) }
//     }
// }
