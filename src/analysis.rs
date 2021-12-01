use rand::{distributions::Distribution, Rng};
use rayon::prelude::*;
use statrs::{distribution::Normal, statistics::Statistics};

use crate::{Dimension, LinearPreCompute, TolStack};

#[derive(Debug, Clone)]
pub struct StackResult {
    pub mean: f64,
    pub std_dev: f64,
    pub float_mean: f64,
    pub float_std_dev: f64,
}

pub fn monte_carlo(stack: &TolStack, samples: usize) -> StackResult {
    let basic_len: f64 = stack
        .dimensions
        .iter()
        .filter_map(|s| match s {
            Dimension::Linear { inner: l } => Some(l.length),
            _ => None,
        })
        .sum();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (linear_result, float_result): (Vec<f64>, Vec<f64>) = (0..samples)
        .into_par_iter()
        .map(|_| {
            // Create a single randomly sampled tolerance stack
            stack_total(stack, &normal, &mut rand::thread_rng())
        })
        .collect();

    StackResult {
        mean: basic_len + linear_result.clone().mean(),
        std_dev: linear_result.std_dev(),
        float_mean: float_result.clone().mean(),
        float_std_dev: float_result.std_dev(),
    }
}

fn stack_total<R: Rng + Sized, D: Distribution<f64>>(
    stack: &TolStack,
    distribution: &D,
    rng: &mut R,
) -> (f64, f64) {
    stack
        .dimensions
        .iter()
        .map(|dim| match dim {
            Dimension::Linear { inner: l } => (linear_sample(distribution, rng, l), 0.0),
            Dimension::Float { inner: f } => {
                let hole = linear_sample(distribution, rng, &f.hole);
                let pin = linear_sample(distribution, rng, &f.pin);
                (0.0, hole - pin)
            }
        })
        .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1))
}

/// Generates a random sample for a linear dimension.
fn linear_sample<R: Rng + Sized, D: Distribution<f64>>(
    distribution: &D,
    rng: &mut R,
    dim: &LinearPreCompute,
) -> f64 {
    let sample = distribution.sample(rng);
    dim.midpoint + (sample * dim.range_over_sigma)
}
