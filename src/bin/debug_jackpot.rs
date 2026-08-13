use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use ariaminer::stratum_to_official::header_from_template;
use ariaminer::official_proof::parse_plain_proof;
use zk_pow::ffi::plain_proof::PlainProof;
use zk_pow::api::proof_utils::{CompiledPublicParams, compute_jackpot_hash};
use zk_pow::circuit::pearl_noise::compute_noise;
use zk_pow::circuit::chip::compute_jackpot;
use serde_json::Value;
use primitive_types::U256;

fn main() {
    let notify_str = std::fs::read_to_string("/tmp/share_1129_notify.json").unwrap();
    let share_str = std::fs::read_to_string("/tmp/share_1129_submit.json").unwrap();

    let notify: Value = serde_json::from_str(&notify_str).unwrap();
    let share: Value = serde_json::from_str(&share_str).unwrap();

    let hdr_hex = notify["params"]["header"].as_str().unwrap();
    let target_hex = notify["params"]["target"].as_str().unwrap();
    let job_id = notify["params"]["job_id"].as_str().unwrap();
    let proof_b64 = share["params"]["plain_proof"].as_str().unwrap();

    println!("Job ID: {}", job_id);
    println!("Target Hex: {}", target_hex);

    let header = header_from_template(hdr_hex).unwrap();
    let p_bytes = B64.decode(proof_b64.trim()).unwrap();
    let p: PlainProof = bincode::deserialize(&p_bytes).unwrap();

    let (private_params, public_params) = parse_plain_proof(header, &p).unwrap();
    let compiled = CompiledPublicParams::from(&public_params);
    let noise = compute_noise(&compiled);
    let jackpot = compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);
    let hash_jackpot = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());

    let hash_jackpot_u256 = U256::from_little_endian(&hash_jackpot);
    println!("Hash Jackpot LE Hex: {}", hex::encode(hash_jackpot));
    println!("Hash Jackpot U256:   {}", hash_jackpot_u256);

    let target_bytes = hex::decode(target_hex).unwrap();
    let mut target_be = [0u8; 32];
    target_be.copy_from_slice(&target_bytes);

    // Compute factor inariaminer:
    let rank = (public_params.rank() as u64).max(1);
    let dpl = public_params.dot_product_length() as u64;
    let tile_size = (public_params.mining_config.rows_pattern.size() as u64) * (public_params.mining_config.cols_pattern.size() as u64);
    let factor = tile_size * dpl;
    let bound_le = ariaminer::stratum_to_official::scaled_bound_le_from_target_be(&target_be, factor);

    println!("\nHASH JACKPOT u32 words (LE):");
    let mut cv_u32 = [0u32; 8];
    for i in 0..8 {
        cv_u32[i] = u32::from_le_bytes(hash_jackpot[i*4..(i+1)*4].try_into().unwrap());
        println!("  cv[{}] = 0x{:08x}", i, cv_u32[i]);
    }

    println!("\nBOUND LE u32 words (LE):");
    let mut bound_u32 = [0u32; 8];
    for i in 0..8 {
        bound_u32[i] = u32::from_le_bytes(bound_le[i*4..(i+1)*4].try_into().unwrap());
        println!("  bound[{}] = 0x{:08x}", i, bound_u32[i]);
    }

    println!("\nCUDA KERNEL POWCHECK SIMULATION (i=7 down to 0):");
    let mut gpu_found = true;
    for i in (0..8).rev() {
        if cv_u32[i] > bound_u32[i] {
            println!("  i={}: cv[0x{:08x}] > bound[0x{:08x}] -> REJECT", i, cv_u32[i], bound_u32[i]);
            gpu_found = false;
            break;
        }
        if cv_u32[i] < bound_u32[i] {
            println!("  i={}: cv[0x{:08x}] < bound[0x{:08x}] -> ACCEPT", i, cv_u32[i], bound_u32[i]);
            break;
        }
        println!("  i={}: cv == bound, continuing...", i);
    }

    println!("\nCUDA GPU SIMULATION RESULT: found={}", gpu_found);

    // Now compute official extract_difficulty_bound for nbits vs share target:
    let target_u256 = U256::from_big_endian(&target_bytes);
    let pool_bound = target_u256 * U256::from(factor);
    println!("Pool Target U256:   {}", target_u256);
    println!("Pool Bound U256:    {}", pool_bound);

    if hash_jackpot_u256 <= pool_bound {
        println!(">>> MATCH: hash_jackpot <= pool_bound! Pool SHOULD have accepted! <<<");
    } else {
        println!(">>> MISMATCH: hash_jackpot > pool_bound! Pool rejected because hash exceeded pool bound! <<<");
    }
}
