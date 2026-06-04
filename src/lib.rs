//! ARIAMiner — open-source, 0% dev-fee CPU miner for Pearl (PRL).
//!
//! Library surface shared between the `ariaminer` binary and the unit tests.
//! Everything here is pure-CPU: the proof-of-work is the canonical Pearl GEMM
//! Int7×Int7 grind, computed on the AVX-512 VNNI / AVX-VNNI int8 datapath with a
//! scalar fallback. Each share is a real `PlainProof` the network validates.

pub mod cpu_engine;
pub mod fast_noise;
pub mod gemm;
#[cfg(feature = "gpu")]
pub mod gpu_ffi;
pub mod merkle;
pub mod microkernel;
pub mod official_grind;
pub mod official_proof;
pub mod pearl_compute;
pub mod proof;
pub mod protocol;
pub mod stratum;
pub mod stratum_to_official;
pub mod vnni;
