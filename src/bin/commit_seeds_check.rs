//! Brique 5 (sous-étape) — valide gen+commit+stir GPU == compute_commitment_hash CPU.
//! `cargo run --release --features gpu --bin commit_seeds_check`
use ariaminer::gpu_ffi;
use pearl_blake3::blake3_digest;

// Réplique de official_grind::compute_commitment_hash (privé) — 4 blake3.
fn cpu_commit(job_key: &[u8; 32], a_bytes: &[u8], b_bytes: &[u8]) -> ([u8; 32], [u8; 32]) {
    let hash_a = blake3_digest(a_bytes, Some(*job_key));
    let hash_b = blake3_digest(b_bytes, Some(*job_key));
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

    let (cpu_a, cpu_b) = cpu_commit(&job_key, &a, &b);
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
