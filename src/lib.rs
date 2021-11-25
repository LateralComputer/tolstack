use num_cpus;
use rand::{distributions::Distribution, SeedableRng};
use rayon::prelude::*;
use statrs::{distribution::Normal, statistics::Statistics};

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn mc() {
        let stack = TolStack::default()
            .with_dim(Linear {
                length: 100.0,
                plus: 0.1,
                minus: 0.1,
                sigma: 3.0,
            })
            .with_dim(Linear {
                length: 100.0,
                plus: 0.1,
                minus: 0.1,
                sigma: 3.0,
            })
            .with_dim(Linear {
                length: 100.0,
                plus: 0.1,
                minus: 0.1,
                sigma: 3.0,
            })
            .with_dim(Linear {
                length: 100.0,
                plus: 0.1,
                minus: 0.1,
                sigma: 3.0,
            })
            .with_dim(Linear {
                length: 100.0,
                plus: 0.1,
                minus: 0.1,
                sigma: 3.0,
            })
            .with_dim(Linear {
                length: 100.0,
                plus: 0.1,
                minus: 0.1,
                sigma: 3.0,
            });

        let start = std::time::Instant::now();
        let result = stack.monte_carlo(1_000_000);
        dbg!(&result);
        dbg!(start.elapsed().as_millis());
        assert!((result.mean - 2.0f64).abs() < 0.01);
        assert!((result.std_dev - 2.0f64).abs() < 0.01);
    }
}

#[derive(Debug)]
pub struct Linear {
    length: f64,
    plus: f64,
    minus: f64,
    sigma: f64,
}
#[derive(Debug)]
pub struct Float {
    hole_width: f64,
    pin_width: f64,
    sigma: f64,
}

#[derive(Debug)]
pub struct Tolerance {
    pub mean: f64,
    pub std_dev: f64,
}

trait PushDim<T> {
    fn with_dim(self, item: T) -> Self;
}
#[derive(Debug, Default)]
struct LinearPreCompute {
    length: f64,
    midpoint: f64,
    range_over_sigma: f64,
}
impl From<Linear> for LinearPreCompute {
    fn from(linear: Linear) -> Self {
        let range = linear.minus + linear.plus;
        let midpoint = -linear.minus + range / 2.0;
        let range_over_sigma = range / linear.sigma;
        LinearPreCompute {
            length: linear.length,
            midpoint,
            range_over_sigma,
        }
    }
}
#[derive(Debug, Default)]
pub struct TolStack {
    linear_dims: Vec<LinearPreCompute>,
    floats_dims: Vec<Float>,
}
impl PushDim<Linear> for TolStack {
    fn with_dim(mut self, item: Linear) -> Self {
        self.linear_dims.push(item.into());
        self
    }
}
impl PushDim<Float> for TolStack {
    fn with_dim(mut self, item: Float) -> Self {
        self.floats_dims.push(item);
        self
    }
}
impl TolStack {
    pub fn monte_carlo(&self, samples: usize) -> Tolerance {
        let basic_len: f64 = self.linear_dims.iter().map(|s| s.length).sum();

        let normal = Normal::new(0.0, 1.0).unwrap();

        let result = (0..samples)
            .collect::<Vec<_>>()
            .par_chunks(samples / (num_cpus::get() - 1))
            .map(|chunk| {
                let mut rng = rand_xoshiro::Xoshiro256PlusPlus::from_entropy();
                chunk
                    .iter()
                    .map(|_i| {
                        self.linear_dims
                            .iter()
                            .map(|d| {
                                let sample = normal.sample(&mut rng);
                                d.midpoint + (sample * d.range_over_sigma)
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<f64>>();

        Tolerance {
            mean: basic_len + result.clone().mean(),
            std_dev: result.std_dev(),
        }

        /*let (tol_mean, std_dev) = self
        .linear_dims
        .iter()
        .map(|d| {
            let rng = rand_xoshiro::Xoshiro256PlusPlus::from_entropy();
            let rand_samples: Vec<f64> = Normal::new(0.0, 1.0)
                .unwrap()
                .sample_iter(rng)
                .take(1_000_000)
                .collect();
            let range = d.minus + d.plus;
            let midpoint = -d.minus + range / 2.0;
            let samples: Vec<f64> = rand_samples
                .iter()
                .map(|rand_normal| midpoint + (rand_normal * range / d.sigma))
                .collect();
            (samples.clone().mean(), samples.population_std_dev())
        })
        .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));

        let mean = basic_len + tol_mean;
        Tolerance { mean, std_dev }*/
    }
}
