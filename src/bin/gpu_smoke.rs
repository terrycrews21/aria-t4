//! Smoke test du pont Rust↔GPU (feature `gpu`).
//! `cargo run --release --features gpu --bin gpu_smoke`
use ariaminer::gpu_ffi;

fn main() {
    let (m, n, k) = (256usize, 256usize, 4096usize);
    let mut a = vec![0i8; m * k];
    let mut b = vec![0i8; n * k];
    for i in 0..a.len() { a[i] = (((i * 131 + 7) % 127) as i64 - 63) as i8; }
    for i in 0..b.len() { b[i] = (((i * 97 + 11) % 127) as i64 - 63) as i8; }

    let mut key = [0u32; 8];
    for i in 0..8 { key[i] = 0x6a09_e667u32 + i as u32; }
    let easy = [0xFFFF_FFFFu32; 8];
    let hard = [0u32; 8];

    let (fe, hits) = gpu_ffi::grind(&a, &b, m, n, k, &key, &easy, 16);
    let (fh, _) = gpu_ffi::grind(&a, &b, m, n, k, &key, &hard, 16);

    let tot = (m / 128) * (n / 128) * 128;
    println!("GPU grind via Rust FFI : bound facile -> {fe} candidats, bound dur -> {fh}");
    if let Some(h) = hits.first() {
        println!("1er hit : {} lignes {:?} × {} colonnes {:?}", h.rows.len(), h.rows, h.cols.len(), h.cols);
    }
    assert_eq!(fe as usize, tot, "bound facile devrait trouver tous les threads");
    assert_eq!(fh, 0, "bound dur ne devrait rien trouver");
    println!("✅ Pont Rust↔GPU FFI opérationnel ({tot} threads-candidats).");
}
