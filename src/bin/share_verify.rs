use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use ariaminer::stratum_to_official::header_from_template;
use ariaminer::official_proof::parse_plain_proof;
use zk_pow::ffi::plain_proof::PlainProof;
use zk_pow::api::verify::verify_plain_proof;
use serde_json::Value;

fn main() {
    let notify_str = std::fs::read_to_string("/tmp/rejected_1115_notify.json").unwrap();
    let share_str = std::fs::read_to_string("/tmp/rejected_1115_share.json").unwrap();

    let notify: Value = serde_json::from_str(&notify_str).unwrap();
    let share: Value = serde_json::from_str(&share_str).unwrap();

    let hdr_hex = notify["params"]["header"].as_str().unwrap();
    let target_hex = notify["params"]["target"].as_str().unwrap();
    let job_id = notify["params"]["job_id"].as_str().unwrap();

    let proof_b64 = share["params"]["plain_proof"].as_str().unwrap();

    println!("Verifying Job ID: {}", job_id);
    println!("Header Hex: {}", hdr_hex);
    println!("Target Hex: {}", target_hex);

    let header = header_from_template(hdr_hex).unwrap();
    println!("Parsed Header version={:#x} nbits={:#x} timestamp={}", header.version, header.nbits, header.timestamp);

    let p_bytes = B64.decode(proof_b64.trim()).unwrap();
    let p: PlainProof = bincode::deserialize(&p_bytes).unwrap();

    println!("Proof m={} n={} k={} rank={}", p.m, p.n, p.k, p.noise_rank);
    println!("A root: {:?}", hex::encode(p.a.proof.root));
    println!("B root: {:?}", hex::encode(p.bt.proof.root));

    match parse_plain_proof(header, &p) {
        Ok((r_a, r_b)) => println!("parse_plain_proof OK! r_a={:?}, r_b={:?}", r_a, r_b),
        Err(e) => println!("parse_plain_proof ERR: {:?}", e),
    }

    match verify_plain_proof(&header, &p, None, zk_pow::api::proof::SeedDerivation::Salted) {
        Ok(_) => println!("verify_plain_proof OK!"),
        Err(e) => println!("verify_plain_proof ERR: {:?}", e),
    }
}
