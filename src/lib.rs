pub mod analysis;

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn mc() {
        let stack = TolStack::new(4.0)
            .with_dim(Linear::new(5.58, 0.03, 0.03, 3.0))
            .with_dim(Linear::new(-25.78, 0.07, 0.07, 3.0))
            .with_dim(Float::new(
                Hole::new(2.18, 0.03, 0.03, 3.0),
                Pin::new(2.13, 0.05, 0.05, 3.0),
            ))
            .with_dim(Linear::new(14.58, 0.05, 0.05, 3.0))
            .with_dim(Float::new(
                Hole::new(1.2, 0.03, 0.03, 3.0),
                Pin::new(1.0, 0.0, 0.0, 3.0),
            ))
            .with_dim(Linear::new(2.5, 0.3, 0.3, 3.0))
            .with_dim(Linear::new(3.85, 0.25, 0.25, 3.0))
            .with_dim(Linear::new(-0.25, 0.1, 0.2, 3.0));

        let start = std::time::Instant::now();
        let result = analysis::monte_carlo(&stack, 10_000_000);
        dbg!(start.elapsed().as_millis());
        dbg!(&result);
        assert!((result.mean - 2f64).abs() < 0.01);
        assert!((result.std_dev - 2f64).abs() < 0.01);
    }
}

#[derive(Debug)]
enum Dimension {
    Linear { inner: LinearPreCompute },
    Float { inner: Float },
}

#[derive(Debug)]
pub struct TolStack {
    assy_sigma: f64,
    dimensions: Vec<Dimension>,
}
impl PushDim<Linear> for TolStack {
    fn with_dim(mut self, linear: Linear) -> Self {
        self.dimensions.push(Dimension::Linear {
            inner: linear.into(),
        });
        self
    }
}
impl PushDim<Float> for TolStack {
    fn with_dim(mut self, float: Float) -> Self {
        self.dimensions.push(Dimension::Float { inner: float });
        self
    }
}
impl TolStack {
    pub fn new(assy_sigma: f64) -> Self {
        TolStack {
            assy_sigma,
            dimensions: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct Linear {
    dim: f64,
    plus: f64,
    minus: f64,
    sigma: f64,
}
impl Linear {
    pub fn new(dim: f64, plus: f64, minus: f64, sigma: f64) -> Linear {
        Linear {
            dim,
            plus,
            minus,
            sigma,
        }
    }
}
#[derive(Debug)]
pub struct Float {
    hole: LinearPreCompute,
    pin: LinearPreCompute,
}

impl Float {
    pub fn new(hole: Hole, pin: Pin) -> Self {
        Self {
            hole: hole.inner,
            pin: pin.inner,
        }
    }
}

#[derive(Debug)]
pub struct Hole {
    inner: LinearPreCompute,
}
impl Hole {
    pub fn new(dim: f64, plus: f64, minus: f64, sigma: f64) -> Self {
        Hole {
            inner: Linear {
                dim,
                plus,
                minus,
                sigma,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
pub struct Pin {
    inner: LinearPreCompute,
}
impl Pin {
    pub fn new(dim: f64, plus: f64, minus: f64, sigma: f64) -> Self {
        Pin {
            inner: Linear {
                dim,
                plus,
                minus,
                sigma,
            }
            .into(),
        }
    }
}

pub trait PushDim<T> {
    fn with_dim(self, item: T) -> Self;
}

/// Pre-computes some values from the [Linear] type. This has a small, but positive impact on performance.
#[derive(Debug, Default, Clone)]
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
            length: linear.dim,
            midpoint,
            range_over_sigma,
        }
    }
}
