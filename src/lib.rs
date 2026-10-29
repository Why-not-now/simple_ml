use rand::RngExt;
use rand::seq::SliceRandom;

pub mod activation;
pub mod datasets;
pub mod layer;
pub mod loss;
pub mod model;
pub mod optimiser;
pub mod tensor;

pub fn split_test<T, R: RngExt>(
    mut data: Vec<T>,
    test_frac: f64,
    rng: Option<&mut R>,
) -> (Vec<T>, Vec<T>) {
    if let Some(rng) = rng {
        data.shuffle(rng);
    };

    let left_len = (data.len() as f64 * test_frac) as usize;
    let training = data.split_off(left_len);
    (training, data)
}
