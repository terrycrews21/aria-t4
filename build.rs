// ariaminer v0.4.0 — build.rs : compile le kernel GPU (libpearl_gpu.a) quand la feature `gpu`
// est active, et le linke. Sans `gpu`, ne fait rien (le miner CPU reste buildable sans CUDA).
use std::env;
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_GPU").is_err() {
        return; // feature gpu OFF → pas de CUDA
    }
    let out = env::var("OUT_DIR").unwrap();
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-12.8".into());
    let nvcc = format!("{}/bin/nvcc", cuda);
    // chemins includes CUTLASS + csrc (overridables par env)
    let cutlass = env::var("CUTLASS_INCLUDE").unwrap_or_else(|_|
        "/mnt/aria/mining-data/pearl/source/miner/pearl-gemm/third_party/cutlass/include".into());
    let csrc = env::var("PEARL_CSRC").unwrap_or_else(|_|
        "/mnt/aria/mining-data/pearl/source/miner/pearl-gemm/csrc".into());

    let shim = "cuda/shim"; // shim TORCH_CHECK no-op (pour tensor_hash officiel)
    // (fichier source, objet) — chaque .cu devient un objet dans libpearl_gpu.a
    let units = [
        ("cuda/pearl_gpu_lib.cu", "pearl_gpu_lib.o"),       // GEMM IMMA + jackpot + pow
        ("cuda/pearl_commit_lib.cu", "pearl_commit_lib.o"), // commitment (tensor_hash officiel)
        ("cuda/pearl_noise_lib.cu", "pearl_noise_lib.o"),   // noise structuré (port spec)
        ("cuda/pearl_resident_lib.cu", "pearl_resident_lib.o"), // gen+commit+stir résident
    ];
    let lib = format!("{}/libpearl_gpu.a", out);
    for (src, obj_name) in units {
        let obj = format!("{}/{}", out, obj_name);
        let status = Command::new(&nvcc)
            .args(["-arch=sm_120a", "-O3", "-std=c++17",
                   "-Icuda", &format!("-I{}", shim), &format!("-I{}", cutlass), &format!("-I{}", csrc),
                   "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
                   "-c", src, "-o", &obj])
            .status().expect("nvcc introuvable");
        assert!(status.success(), "compilation CUDA échouée: {src}");
        Command::new("ar").args(["crus", &lib, &obj]).status().expect("ar");
    }

    println!("cargo:rustc-link-search=native={}", out);
    println!("cargo:rustc-link-lib=static=pearl_gpu");
    println!("cargo:rustc-link-search=native={}/lib64", cuda);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_kernel.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_fold.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_commit_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_noise_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_resident_lib.cu");
}
