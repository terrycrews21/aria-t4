//! Brique 4 — validation bit-exact du noise GPU.
//! `cargo run --release --features gpu --bin noise_check`
//! Compare le noise GPU (port spec) vs `zk_pow::circuit::pearl_noise::compute_noise_for_indices`
//! (réf consensus CPU), côtés A et B, sur des seeds arbitraires.
use ariaminer::gpu_ffi;
use zk_pow::circuit::pearl_noise::compute_noise_for_indices;

fn label(s: &[u8]) -> [u8; 32] {
    let mut r = [0u8; 32];
    r[..s.len()].copy_from_slice(s);
    r
}

fn main() {
    let (m, n, k) = (256usize, 256usize, 4032usize);
    let rank = 128usize;

    // seeds arbitraires mais fixes (a_noise_seed, b_noise_seed).
    let mut a_seed = [0u8; 32];
    let mut b_seed = [0u8; 32];
    for i in 0..32 {
        a_seed[i] = (i as u8).wrapping_mul(37).wrapping_add(11);
        b_seed[i] = (i as u8).wrapping_mul(53).wrapping_add(7);
    }

    let a_rows: Vec<usize> = (0..m).collect();
    let b_cols: Vec<usize> = (0..n).collect();

    // Réf CPU consensus : MMSlice{a, b}. commitment_hash = (b_seed, a_seed).
    let cpu = compute_noise_for_indices(k, rank, (b_seed, a_seed), &a_rows, &b_cols);

    // GPU : A avec SEED_LABEL_A + a_seed ; B avec SEED_LABEL_B + b_seed.
    let gpu_a = gpu_ffi::noise(&label(b"A_tensor"), &a_seed, m, k);
    let gpu_b = gpu_ffi::noise(&label(b"B_tensor"), &b_seed, n, k);

    let check = |name: &str, cpu_rows: &[Vec<i8>], gpu_flat: &[i8]| -> bool {
        let mut bad = 0usize;
        let mut first_bad = (usize::MAX, 0usize);
        for (i, row) in cpu_rows.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                let g = gpu_flat[i * k + j];
                if c != g {
                    bad += 1;
                    if first_bad.0 == usize::MAX {
                        first_bad = (i, j);
                    }
                }
            }
        }
        let total = cpu_rows.len() * k;
        if bad == 0 {
            println!("  {name} : ✅ BIT-EXACT sur {total} éléments ({}×{k})", cpu_rows.len());
            true
        } else {
            let (bi, bj) = first_bad;
            println!(
                "  {name} : ❌ {bad}/{total} faux. 1er @({bi},{bj}) cpu={} gpu={}",
                cpu_rows[bi][bj], gpu_flat[bi * k + bj]
            );
            false
        }
    };

    println!("=== Noise GPU vs compute_noise_for_indices (m={m} n={n} k={k} rank={rank}) ===");
    let ok_a = check("noise A", &cpu.a, &gpu_a);
    let ok_b = check("noise B", &cpu.b, &gpu_b);

    if ok_a && ok_b {
        println!("\n✅ Brique 4 : noise structuré GPU == consensus, bit-exact (A et B), sm_120.");
    } else {
        println!("\n❌ Brique 4 : divergence — port noise à corriger.");
        std::process::exit(1);
    }
}
