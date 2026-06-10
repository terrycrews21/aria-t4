//! Validation OFFLINE contre un VRAI job LuckyPool capturé (10/06, job 343ead91_500000).
//! Grind fix-B (131072², rank 256) contre bound = target×h·w·k, puis recompute OFFICIEL
//! du jackpot (parse_plain_proof) et comparaison aux deux interprétations du bound.
//! Tranche : (a) header/job_key OK ? (b) le bound pool = target×h·w·k ou target brut ?
//! Build : cargo build --release --features gpu --bin luckypool_job_check
use ariaminer::gpu_ffi::ResidentCtx;
use ariaminer::official_grind::{
    build_proof_from_hit_fixb, canonical_gpu_config, grind_only_ctx, Workspace,
};
use ariaminer::stratum_to_official::{header_from_template, scaled_bound_le_from_target_be};
use rand::rngs::StdRng;
use rand::SeedableRng;

const HDR_HEX: &str = "00004020df2a4132ba5893f1954bf490d9852173e307dd71dec8f68533d27fc0c9ca31276b5380962696afc7352641624e7cb7565a776af3934f19b2784b5750141f20cacc35296a9cfe0018";
const TARGET_HEX: &str = "000000000000218dcdb37c99ae924f227d028a1dfb9389b52007dd441355475a";

/// a ≤ b sur 32 octets little-endian.
fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    true
}

fn main() {
    unsafe { std::env::set_var("ARIA_FIXB", "1") };
    let (m, n, k) = (131072usize, 131072usize, 4096usize);
    let header = header_from_template(HDR_HEX).expect("parse header LuckyPool");
    println!(
        "  header: version={:#x} ntime={:#x} nbits={:#x}",
        header.version, header.timestamp, header.nbits
    );
    let mut target_be = [0u8; 32];
    hex::decode_to_slice(TARGET_HEX, &mut target_be).unwrap();
    // Pour itérer vite tout en exerçant le VRAI chemin BE, on peut éclaircir le
    // target (ARIA_TEST_TARGET_BYTE0=0x0f → 1ᵉ octet, reste 0xff). Le mécanisme
    // big-endian est identique, juste plus facile à trouver.
    if let Ok(b0) = std::env::var("ARIA_TEST_TARGET_BYTE0") {
        let b0 = u8::from_str_radix(b0.trim_start_matches("0x"), 16).unwrap_or(0x0f);
        target_be = [0xffu8; 32];
        target_be[0] = b0;
        println!("  (test) target éclairci big-endian = {:02x}ff..ff", b0);
    }
    let config = canonical_gpu_config(k as u32);
    let rank = config.rank as usize;
    let be = std::env::var("ARIA_BE").is_ok();
    println!("  config rank={rank} cols={:?}  BE={be}", config.cols_pattern.to_list());
    // En mode BE (LuckyPool) le kernel attend les OCTETS du target big-endian tels
    // quels (il les byteswappe). Sinon (legacy) bound little-endian = target inversé.
    let bound_for_kernel: [u8; 32] = if be {
        target_be
    } else {
        let mut b = target_be; b.reverse(); b
    };
    let factor = 128u64 * k as u64;
    let bound_scaled = scaled_bound_le_from_target_be(&target_be, factor);
    let mut target_le = target_be;
    target_le.reverse();

    let ctx = ResidentCtx::new(m, n, k, 64);
    let mut rng = StdRng::seed_from_u64(0x10C4_9001);

    println!("▶ grind #1 (remplit B) …");
    let (s0, _h0, _jk0) = loop {
        if let Some(t) = grind_only_ctx(&ctx, &mut rng, &header, &config, &bound_for_kernel) {
            break t;
        }
    };
    println!("▶ grind #2 (A varie, B figé) …");
    let (s1, hit, jk) = loop {
        if let Some(t) = grind_only_ctx(&ctx, &mut rng, &header, &config, &bound_for_kernel) {
            break t;
        }
    };
    println!("  S0={s0:#x} S1={s1:#x} — hit trouvé, build preuve (CPU, ~1 min à 131072)…");
    let mut ws = Workspace::new();
    let proof = build_proof_from_hit_fixb(s1, s0, &hit, &jk, m, n, k, rank, &mut ws);

    match zk_pow::ffi::plain_proof::parse_plain_proof(header, &proof) {
        Err(e) => println!("❌ parse_plain_proof ÉCHOUE : {e:?} → header/pipeline KO"),
        Ok((privp, public)) => {
            // Recompute OFFICIEL du jackpot — même chaîne que verify_plain_proof
            // (parse → compute_noise → compute_jackpot → compute_jackpot_hash).
            use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
            use zk_pow::circuit::chip::compute_jackpot;
            use zk_pow::circuit::pearl_noise::compute_noise;
            let compiled = CompiledPublicParams::from(&public);
            let noise = compute_noise(&compiled);
            let jackpot = compute_jackpot(&compiled, &privp.s_a, &privp.s_b, &noise);
            let j = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
            println!("  jackpot OFFICIEL recalculé (bytes) = {}", hex::encode(j));
            // j est l'array de 32 octets du jackpot. Teste les 2 conventions :
            let mut j_rev = j; j_rev.reverse();
            // LA règle LuckyPool : j lu BIG-endian < target_be  ⇔  j_rev (LE) < target_le.
            let be_ok = le_leq(&j_rev, &target_le) && j_rev != target_le;
            println!(
                "  [BE LuckyPool] jackpot big-endian < target : {}",
                if be_ok { "✅✅ OUI → notre grind BE trouve des shares VALIDES LuckyPool !" } else { "❌ NON" }
            );
            // (référence) ce que l'ancienne convention LE aurait donné
            println!("  (réf LE)  j little-endian ≤ target_le  : {}", if le_leq(&j, &target_le) { "oui" } else { "non" });
            println!("  (réf bloc) j ≤ target×h·w·k            : {}", if le_leq(&j, &bound_scaled) { "oui" } else { "non" });
        }
    }
}
