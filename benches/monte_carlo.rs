use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tolstack::*;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Monte Carlo: 100,000 samples", |b| {
        b.iter(|| analysis::monte_carlo(&stack(), black_box(100_000)))
    });
    c.bench_function("Monte Carlo: 1,000,000 samples", |b| {
        b.iter(|| analysis::monte_carlo(&stack(), black_box(1_000_000)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn stack() -> TolStack {
    TolStack::new(4.0)
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
        .with_dim(Linear::new(-0.25, 0.1, 0.2, 3.0))
}
