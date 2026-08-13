//! Brique 5 (sous-étape) — valide gen+commit+stir GPU == compute_commitment_hash CPU.
//! `cargo run --release --features gpu --bin commit_seeds_check`
use ariaminer::gpu_ffi;
use pearl_blake3::blake3_digest;

// V3 salting constants (`zk_pow::api::seed`, post-fork @ block 99,000).
const SEED_SALT_A: [u8; 32] = [
    0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8,
    0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b,
];
const SEED_SALT_B: [u8; 32] = [
    0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f,
    0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9,
];

fn bind_root_cpu(root: &[u8; 32], dim: u32, salt: &[u8; 32]) -> [u8; 32] {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(root);
    msg[32..36].copy_from_slice(&dim.to_le_bytes());
    blake3_digest(&msg, Some(*salt))
}

// Réplique SALTED (V3) de official_grind::compute_commitment_hash — 6 blake3.
fn cpu_commit(job_key: &[u8; 32], a_bytes: &[u8], b_bytes: &[u8], m: u32, n: u32) -> ([u8; 32], [u8; 32]) {
    let hash_a = blake3_digest(a_bytes, Some(*job_key));
    let hash_b = blake3_digest(b_bytes, Some(*job_key));
    let hash_a = bind_root_cpu(&hash_a, m, &SEED_SALT_A);
    let hash_b = bind_root_cpu(&hash_b, n, &SEED_SALT_B);
    let mut bi = [0u8; 64];
    bi[..32].copy_from_slice(job_key);
    bi[32..].copy_from_slice(&hash_b);
    let b_seed = blake3_digest(&bi, None);
    let mut ai = [0u8; 64];
    ai[..32].copy_from_slice(&b_seed);
    ai[32..].copy_from_slice(&hash_a);
    let a_seed = blake3_digest(&ai, None);
    (a_seed, b_seed) // (a_noise_seed, b_noise_seed)
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

fn main() {
    let (m, n, k) = (256usize, 128usize, 4032usize);
    let setup_seed: u64 = 0x5080_C0FF_EE12_3456;
    let mut job_key = [0u8; 32];
    for (i, b) in job_key.iter_mut().enumerate() { *b = (i as u8).wrapping_mul(73).wrapping_add(5); }

    // Régénère A,B (mêmes que le GPU via int7_at) en bytes pour le commit CPU.
    let mut a = vec![0u8; m * k];
    let mut b = vec![0u8; n * k];
    for i in 0..m { for j in 0..k { a[i * k + j] = gpu_ffi::int7_at(setup_seed, 0, i as u32, j as u32) as u8; } }
    for i in 0..n { for j in 0..k { b[i * k + j] = gpu_ffi::int7_at(setup_seed, 1, i as u32, j as u32) as u8; } }

    let (cpu_a, cpu_b) = cpu_commit(&job_key, &a, &b, m as u32, n as u32);
    let (gpu_a, gpu_b) = gpu_ffi::commit_seeds(setup_seed, m, n, k, &job_key);

    let oka = cpu_a == gpu_a;
    let okb = cpu_b == gpu_b;
    println!("=== gen+commit+stir GPU vs compute_commitment_hash CPU (m={m} n={n} k={k}) ===");
    println!("  a_noise_seed  CPU={}  GPU={}  {}", hex(&cpu_a[..8]), hex(&gpu_a[..8]), if oka {"✅"} else {"❌"});
    println!("  b_noise_seed  CPU={}  GPU={}  {}", hex(&cpu_b[..8]), hex(&gpu_b[..8]), if okb {"✅"} else {"❌"});
    if oka && okb {
        println!("\n✅ Brique 5 (prologue) : signal+commit+seeds GPU == consensus CPU, bit-exact, sm_120.");
    } else {
        println!("  CPU a_full={}\n  GPU a_full={}", hex(&cpu_a), hex(&gpu_a));
        println!("\n❌ divergence prologue résident.");
        std::process::exit(1);
    }
}
