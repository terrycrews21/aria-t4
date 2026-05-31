//! ARIAMiner — open-source CPU miner for Pearl (PRL).
//!
//! Library surface shared between the `ariaminer` binary (`src/bin/mine.rs`)
//! and the unit tests. Everything here is pure-CPU: the proof-of-work is an
//! amortized GEMM Int7×Int7 grind with AVX-512 / AVX-VNNI register-blocking
//! and a scalar fallback.

pub mod cpu_engine;
pub mod fast_noise;
pub mod gemm;
pub mod merkle;
pub mod microkernel;
pub mod pearl_compute;
pub mod proof;
pub mod protocol;
pub mod stratum;
pub mod vnni;
