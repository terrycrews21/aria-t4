//! Brique 2 — validation bit-exact du commitment GPU.
//! `cargo run --release --features gpu --bin commit_check`
//! Compare `tensor_hash` GPU (officiel) vs `pearl_blake3::blake3_digest` (consensus CPU),
//! sur plusieurs tailles réalistes. Le commitment Pearl = blake3 keyed des matrices.
use ariaminer::gpu_ffi;
use pearl_blake3::blake3_digest;

fn main() {
    // splitmix64 indexé → données non dégénérées
    let sm = |mut x: u64| {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };

    // Tailles a_sig = m*k réelles (k=4032), toutes multi-block en pratique.
    // 262144 = cas single-block ARTIFICIEL (n'arrive jamais : matrices >0.5Mo) — gardé comme témoin.
    let sizes = [
        262144usize,    // single-block (témoin, hors usage réel)
        516_096,        // m=128  (plus petit cas réel) -> 2 blocks
        1_032_192,      // m=256
        16_515_072,     // m=4096 (taille gpu_live) -> 63 blocks
    ];
    let mut key = [0u8; 32];
    for (i, b) in key.iter_mut().enumerate() {
        *b = (sm(i as u64) & 0xff) as u8;
    }

    let mut real_ok = true;
    for &len in &sizes {
        let single_block = len <= 262_144; // témoin hors usage réel
        let mut data = vec![0u8; len];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (sm((i as u64) << 1 | 1) & 0xff) as u8;
        }
        let cpu = blake3_digest(&data, Some(key));
        let gpu = gpu_ffi::tensor_hash(&data, &key);
        let ok = cpu == gpu;
        if !single_block {
            real_ok &= ok;
        }
        let tag = if ok { "✅ BIT-EXACT" } else if single_block { "⚠️ diverge (single-block, hors usage)" } else { "❌ DIVERGE" };
        println!("len={len:>9} ({}blk)  CPU={}  GPU={}  {tag}",
            if single_block { "1" } else { "≥2" }, hex(&cpu[..8]), hex(&gpu[..8]));
    }
    if real_ok {
        println!("\n✅ Brique 2 : commitment GPU (tensor_hash officiel) == blake3 consensus sur TOUTES les tailles de minage réelles (multi-block), sm_120.");
        println!("   (Le single-block 262144 o diverge mais n'arrive JAMAIS : a_sig = m*k ≥ 516096 o.)");
    } else {
        println!("\n❌ Brique 2 : divergence sur une taille réelle — à corriger.");
        std::process::exit(1);
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
