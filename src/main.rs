use simple_ml::activation::{ActivationLayer, ReluActivation};
use simple_ml::datasets::{Data, DataClass};
use simple_ml::layer::{DenseLayer, Full, Layer};
use simple_ml::loss::SoftmaxCategoricalLoss;
use simple_ml::model::Model;
use simple_ml::optimiser::AdamOptimiser;
use simple_ml::{datasets, split_test};

fn main() {
    let mut rng = rand::rng();

    type L1 = DenseLayer<2, 32>;
    type L2 = DenseLayer<32, 32>;
    type L3 = DenseLayer<32, 3>;

    let data = datasets::spiral_data(1000, 3, &mut rng);
    let layers = L1::new(
        L1::normal_xavier_weights(&mut rng),
        L1::zero_bias(),
        0.,
        5e-4,
        0.,
        5e-4,
    );
    let layers = layers.merge::<ActivationLayer<_>, _>(ReluActivation.into());
    let layers = layers.merge(L2::new(
        L2::normal_xavier_weights(&mut rng),
        L2::zero_bias(),
        0.,
        5e-4,
        0.,
        5e-4,
    ));
    let layers = layers.merge::<ActivationLayer<_>, _>(ReluActivation.into());
    let layers = layers.merge(L3::new(
        L3::normal_xavier_weights(&mut rng),
        L3::zero_bias(),
        0.,
        5e-4,
        0.,
        5e-4,
    ));
    let loss = SoftmaxCategoricalLoss;
    let optimiser = AdamOptimiser::new_default();
    let mut model = Model::new(layers, loss, optimiser);

    const N: usize = 40;
    let mut grid = [['.'; N]; N];

    for datum in &data {
        let [x, y] = datum.vector().into();
        let x_int = (((x + 1.) * 0.5 * N as f64) as usize).min(N - 1);
        let y_int = (((y + 1.) * 0.5 * N as f64) as usize).min(N - 1);

        grid[y_int][x_int] = datum
            .class()
            .to_string()
            .chars()
            .next()
            .expect("Value should be at least 1 characters");
    }

    for row in grid {
        let row_str: String = row.iter().flat_map(|v| [*v, ' ']).collect();
        println!("{row_str}")
    }

    let (train, test) = split_test(data, 0.2, Some(&mut rng));
    for chunk in train.chunks(32) {
        let inputs: Vec<_> = chunk.iter().map(|data| data.vector()).collect();
        let expected: Vec<_> = chunk
            .iter()
            .map(|data| {
                let mut class = [0., 0., 0.];
                class[data.class() as usize] = 1.;
                class.into()
            })
            .collect();
        model.train_batch_full(&inputs, &expected, &mut rng);
    }

    let test_len = test.len();
    let mut accurate = 0.;
    for data in test {
        let result = model.predict(data.vector());
        if result
            .iter()
            .enumerate()
            .max_by(|(_, l_val), (_, r_val)| l_val.total_cmp(r_val))
            .expect("prediction should not be empty")
            .0
            == (data.class() as usize)
        {
            accurate += 1.;
        }
    }

    let accuracy = accurate / test_len as f64;
    println!("{accuracy}")
}
