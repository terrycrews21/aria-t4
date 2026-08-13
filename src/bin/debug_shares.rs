use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use ariaminer::stratum_to_official::header_from_template;
use ariaminer::official_proof::parse_plain_proof;
use zk_pow::ffi::plain_proof::PlainProof;

fn main() {
    let hdr_hex = "00004020f4effd8f8384abe0f0fa2762f86af5b903a7a922175e6bb8416c2315466a2c82c7926ca016161864dccae74fd5f05563e0fba1c4b0470b077e5eaff84b09c056ad477c6a3eb00018";
    let header = header_from_template(hdr_hex).unwrap();

    let s1_b64 = std::fs::read_to_string("/tmp/share1.b64").unwrap();
    let s2_b64 = std::fs::read_to_string("/tmp/share2.b64").unwrap();

    let p1: PlainProof = bincode::deserialize(&B64.decode(s1_b64.trim()).unwrap()).unwrap();
    let p2: PlainProof = bincode::deserialize(&B64.decode(s2_b64.trim()).unwrap()).unwrap();

    let (r_a1, r_b1) = parse_plain_proof(header, &p1).unwrap();
    let (r_a2, r_b2) = parse_plain_proof(header, &p2).unwrap();

    println!("Share 1 hash_jackpot: {:x?}", r_b1.hash_jackpot);
    println!("Share 1 hash_jackpot hex: {}", hex::encode(r_b1.hash_jackpot));

    println!("\nShare 2 hash_jackpot: {:x?}", r_b2.hash_jackpot);
    println!("Share 2 hash_jackpot hex: {}", hex::encode(r_b2.hash_jackpot));
}
