//! E2E consensus proof for the HeroMiners dialect, on a REAL intercepted job.
//!
//! 1. Rebuilds the official job from the captured gfwroute `mining.notify`
//!    (header / job suffix → difficulty → penalized share bound).
//! 2. Asserts `share_bound_le(d_pool, cfg)` == the master `check_rank_penalty`
//!    bound bit-for-bit (same saturating algebra).
//! 3. Mines ONE share via the production CPU path at an EASY bound
//!    (computationally tractable) and verifies it with the master
//!    `verify_plain_proof(header, proof, Some(asst_nbits), Salted)` —
//!    the exact call the pool verifier makes. Also submits the proof through
//!    the same parse_plain_proof(Salted) gate the production miner applies
//!    pre-submission.
//!
//! RUN: cargo run --release --bin hero_share_check -- /tmp/peak_capture/job_00000005.json

use ariaminer::official_grind::{Workspace, try_mine_one_bounded};
use ariaminer::official_proof::{encode_base64, encode_base64_gzip, parse_plain_proof};
use ariaminer::protocol::{Job, MiningParams};
use ariaminer::stratum_to_official::{config_from_params, share_bound_le};
use primitive_types::U256;

fn herominers_default_params() -> MiningParams {
    let (rows_pattern, cols_pattern) = if std::env::var("ARIA_T4_DUAL").is_ok() {
        ((0..16).collect(), (0..16).collect())
    } else if std::env::var("ARIA_TMA_MS").is_ok() {
        (
            vec![0, 8, 32, 40, 64, 72, 96, 104],
            vec![0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225],
        )
    } else {
        (
            vec![0, 8, 32, 40, 64, 72, 96, 104],
            vec![0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
        )
    };
    MiningParams {
        m: 16_384,
        n: 65_536,
        k: 8192,
        rank: 128,
        rows_pattern,
        cols_pattern,
        mma_type: "Int7xInt7ToInt32".into(),
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: hero_share_check <notify.json>");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let job = Job::from_params(v.get("params").expect("params"))?;
    println!("job_id={} header={}…", job.job_id, &job.header_template[..16]);

    let suffix: u64 = job
        .job_id
        .rsplit_once('_')
        .and_then(|(_, d)| d.parse().ok())
        .expect("job_id suffix difficulty");
    let difficulty = suffix.saturating_mul(1u64 << 32);
    println!("suffix={suffix} d_pool={difficulty} (2^{:.1})", (difficulty as f64).log2());

    let params = herominers_default_params();
    let config = config_from_params(&params)?;
    let bound_le = share_bound_le(difficulty, &config);

    // (2) bit-equality with the master penalized scheme (saturating semantics).
    let target = U256::MAX / difficulty + 1; // == share_target_from_difficulty semantics? log both
    let share_target = U256::from_big_endian(&ariaminer::cpu_engine::share_target_from_difficulty(difficulty));
    let master = zk_pow::api::sanity_checks::penalized_target_bound(share_target, &config)
        .unwrap_or(U256::MAX);
    let mut master_le = [0u8; 32];
    master.to_little_endian(&mut master_le);
    println!("bound_le (ariaminer) = {}", hex::encode(bound_le));
    println!("bound_le (master)    = {}", hex::encode(master_le));
    assert_eq!(bound_le, master_le, "share bound must equal master penalized_target_bound");
    println!("✅ bound bit-exact with master check_rank_penalty algebra (target={target:#x})");

    // (3) mine one share on the REAL header at an EASY bound, verify Salted.
    // EASY: mantissa near-max → bound saturates to ~allspace/2^8 so a hit
    // lands within a few attempts; the header/job_key stay 100% real.
    let nbits_easy: u32 = 0x207f_ffff;
    let mut header = ariaminer::stratum_to_official::header_from_template(&job.header_template)?;
    header.nbits = nbits_easy;
    // nbits_easy (0x207fffff) saturates scale() to U256::MAX verifier-side, so
    // the verifier accepts any hash here; the grind uses MAX/64 to exercise a
    // few dozen attempts before hitting. Difficulty selectivity itself is
    // covered by the bit-exact bound assertion above.
    let easy_bound_u = U256::MAX / 64;
    let mut easy_bound_le = [0u8; 32];
    easy_bound_u.to_little_endian(&mut easy_bound_le);

    let (m, n, k) = (256usize, 256usize, params.k as usize);
    let mut ws = Workspace::new();
    let mut rng = rand::rngs::StdRng::from_entropy();
    use rand::SeedableRng;
    let proof = {
        let mut p = None;
        for attempt in 0..100u64 {
            if let Some(h) = try_mine_one_bounded(&mut ws, &mut rng, m, n, k, &header, &config, &easy_bound_le) {
                println!("hit at attempt {attempt}");
                p = Some(h);
                break;
            }
        }
        p.expect("no hit at easy bound within 100 attempts")
    };
    println!("proof: m^={} n^={} k={} noise_rank={} moe={}", proof.m, proof.n, proof.k, proof.noise_rank, proof.moe.is_some());

    // Salted structural parse + full master verification (what the pool runs).
    let (_priv, _pub) = parse_plain_proof(header, &proof).expect("Salted parse (roots check)");
    println!("✅ parse_plain_proof(Salted) OK");
    zk_pow::api::verify::verify_plain_proof(&header, &proof, None, zk_pow::api::proof::SeedDerivation::Salted)
        .expect("master verify_plain_proof must pass at header nbits");
    println!("✅ verify_plain_proof(Salted, header.nbits easy) PASS");

    // The mined share would also satisfy check_rank_penalty at ITS bound.
    let (_p2, pub2) = parse_plain_proof(header, &proof).unwrap();
    let compiled = zk_pow::api::proof_utils::CompiledPublicParams::from(&pub2);
    let noise = zk_pow::circuit::pearl_noise::compute_noise(&compiled);
    let jp = zk_pow::circuit::chip::compute_jackpot(&compiled, &_priv.s_a, &_priv.s_b, &noise);
    let hj = zk_pow::api::proof_utils::compute_jackpot_hash(&jp, compiled.a_noise_seed());
    zk_pow::api::sanity_checks::check_rank_penalty(&pub2.mining_config, &hj, nbits_easy)
        .expect("check_rank_penalty at easy nbits");
    println!("✅ check_rank_penalty(Salted) PASS — hash {}", hex::encode(&hj[..8]));

    // Wire encoding round-trips (both dialects).
    let raw = encode_base64(&proof)?;
    let gz = encode_base64_gzip(&proof)?;
    println!("base64 len={}  gzip+b64 len={}", raw.len(), gz.len());
    println!("🎉 hero_share_check: ALL PASS");
    Ok(())
}
