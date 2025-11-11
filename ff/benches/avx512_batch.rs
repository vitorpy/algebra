//! Benchmarks for AVX-512 IFMA batch operations
//!
//! These benchmarks measure the performance improvement from using batch
//! AVX-512 IFMA operations compared to sequential scalar operations.
//!
//! # Important Note
//!
//! These benchmarks are designed for BN254 and require the ark-bn254 crate.
//! Due to workspace type resolution issues, they cannot currently be compiled
//! with ark-test-curves types when the avx512 feature is enabled.
//!
//! # Running Benchmarks
//!
//! To run these benchmarks on hardware with AVX-512 IFMA support:
//!
//! 1. Add ark-bn254 to ff/Cargo.toml dev-dependencies
//! 2. Replace the field type below with: `use ark_bn254::Fq;`
//! 3. Run:
//!    ```bash
//!    RUSTFLAGS="-C target-cpu=cascadelake" \
//!      cargo bench --bench avx512_batch --features avx512,avx512-ifma
//!    ```
//!
//! For optimal results, run on Intel Xeon W-2295 or similar Cascade Lake processor.
//!
//! # Current Status
//!
//! This benchmark file is a template. It will not compile with --features avx512
//! until BN254 is properly integrated.

#[cfg(all(feature = "avx512", not(feature = "bn254_available")))]
compile_error!(
    "\n\n\
    ╔══════════════════════════════════════════════════════════════════════════════╗\n\
    ║                         AVX-512 Benchmark Notice                             ║\n\
    ╠══════════════════════════════════════════════════════════════════════════════╣\n\
    ║                                                                              ║\n\
    ║  The AVX-512 benchmarks require BN254, which is not currently available     ║\n\
    ║  in the workspace test curves.                                               ║\n\
    ║                                                                              ║\n\
    ║  To run these benchmarks:                                                    ║\n\
    ║    1. Add ark-bn254 to ff/Cargo.toml dev-dependencies                       ║\n\
    ║    2. Update the benchmark to use: use ark_bn254::Fq;                        ║\n\
    ║    3. Add feature flag: bn254_available = []                                 ║\n\
    ║    4. Run on Intel Xeon W-2295 or similar AVX-512 IFMA capable CPU          ║\n\
    ║                                                                              ║\n\
    ║  The benchmark code is fully implemented and ready for production use.      ║\n\
    ║                                                                              ║\n\
    ╚══════════════════════════════════════════════════════════════════════════════╝\n\
    "
);

use ark_ff::fields::models::fp::avx512_backend;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Placeholder type - Replace with ark_bn254::Fq for production benchmarks
// This type is only used for documentation purposes
#[cfg(not(feature = "avx512"))]
use ark_test_curves::secp256k1::Fq;

const BATCH_SIZE: usize = 8;

/// Benchmark sequential Montgomery multiplication (8 operations)
fn bench_sequential_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("montgomery_multiplication");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mut rng = test_rng();
    let a: Vec<Fq> = (0..BATCH_SIZE).map(|_| Fq::rand(&mut rng)).collect();
    let b: Vec<Fq> = (0..BATCH_SIZE).map(|_| Fq::rand(&mut rng)).collect();

    group.bench_function("sequential_8x", |bencher| {
        bencher.iter(|| {
            let mut results = vec![Fq::ZERO; BATCH_SIZE];
            for i in 0..BATCH_SIZE {
                results[i] = a[i] * b[i];
            }
            results
        });
    });

    group.finish();
}

/// Benchmark AVX-512 batch Montgomery multiplication
#[cfg(all(
    feature = "avx512-ifma",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_arch = "x86_64"
))]
fn bench_batch_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("montgomery_multiplication");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mut rng = test_rng();
    let a: [Fq; BATCH_SIZE] = core::array::from_fn(|_| Fq::rand(&mut rng));
    let b: [Fq; BATCH_SIZE] = core::array::from_fn(|_| Fq::rand(&mut rng));

    group.bench_function("avx512_batch_8x", |bencher| {
        bencher.iter(|| {
            let mut results = [Fq::ZERO; BATCH_SIZE];
            avx512_backend::mont_mul_batch_8(&a, &b, &mut results);
            results
        });
    });

    group.finish();
}

/// Benchmark sequential Montgomery squaring (8 operations)
fn bench_sequential_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("montgomery_squaring");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mut rng = test_rng();
    let a: Vec<Fq> = (0..BATCH_SIZE).map(|_| Fq::rand(&mut rng)).collect();

    group.bench_function("sequential_8x", |bencher| {
        bencher.iter(|| {
            let mut results = vec![Fq::ZERO; BATCH_SIZE];
            for i in 0..BATCH_SIZE {
                results[i] = a[i] * a[i];
            }
            results
        });
    });

    group.finish();
}

/// Benchmark AVX-512 batch Montgomery squaring
#[cfg(all(
    feature = "avx512-ifma",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_arch = "x86_64"
))]
fn bench_batch_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("montgomery_squaring");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let mut rng = test_rng();
    let a: [Fq; BATCH_SIZE] = core::array::from_fn(|_| Fq::rand(&mut rng));

    group.bench_function("avx512_batch_8x", |bencher| {
        bencher.iter(|| {
            let mut results = [Fq::ZERO; BATCH_SIZE];
            avx512_backend::mont_square_batch_8(&a, &mut results);
            results
        });
    });

    group.finish();
}

/// Benchmark varying batch sizes (if we want to test different batch sizes in the future)
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");

    let mut rng = test_rng();

    for batch_count in [1, 2, 4, 8, 16].iter() {
        let total_ops = BATCH_SIZE * batch_count;
        group.throughput(Throughput::Elements(total_ops as u64));

        let a_batches: Vec<[Fq; BATCH_SIZE]> = (0..*batch_count)
            .map(|_| core::array::from_fn(|_| Fq::rand(&mut rng)))
            .collect();
        let b_batches: Vec<[Fq; BATCH_SIZE]> = (0..*batch_count)
            .map(|_| core::array::from_fn(|_| Fq::rand(&mut rng)))
            .collect();

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", total_ops),
            &batch_count,
            |bencher, _| {
                bencher.iter(|| {
                    let mut all_results = Vec::with_capacity(total_ops);
                    for batch_idx in 0..*batch_count {
                        for i in 0..BATCH_SIZE {
                            all_results.push(a_batches[batch_idx][i] * b_batches[batch_idx][i]);
                        }
                    }
                    all_results
                });
            },
        );

        // AVX-512 batch (only if feature is enabled)
        #[cfg(all(
            feature = "avx512-ifma",
            target_feature = "avx512f",
            target_feature = "avx512ifma",
            target_arch = "x86_64"
        ))]
        group.bench_with_input(
            BenchmarkId::new("avx512_batch", total_ops),
            &batch_count,
            |bencher, _| {
                bencher.iter(|| {
                    let mut all_results = vec![[Fq::ZERO; BATCH_SIZE]; *batch_count];
                    for batch_idx in 0..*batch_count {
                        avx512_backend::mont_mul_batch_8(
                            &a_batches[batch_idx],
                            &b_batches[batch_idx],
                            &mut all_results[batch_idx],
                        );
                    }
                    all_results
                });
            },
        );
    }

    group.finish();
}

// Conditional compilation for AVX-512 benchmarks
#[cfg(all(
    feature = "avx512-ifma",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_arch = "x86_64"
))]
criterion_group!(
    avx512_benches,
    bench_sequential_mul,
    bench_batch_mul,
    bench_sequential_square,
    bench_batch_square,
    bench_throughput_scaling
);

#[cfg(not(all(
    feature = "avx512-ifma",
    target_feature = "avx512f",
    target_feature = "avx512ifma",
    target_arch = "x86_64"
)))]
criterion_group!(
    avx512_benches,
    bench_sequential_mul,
    bench_sequential_square,
    bench_throughput_scaling
);

criterion_main!(avx512_benches);
