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
        ("cuda/pearl_gpu_lib.cu", "pearl_gpu_lib.o"),       // GEMM IMMA + jackpot 8×16 + pow (v1.0)
        ("cuda/pearl_gpu_2x64_lib.cu", "pearl_gpu_2x64_lib.o"), // GEMM IMMA + jackpot 2×64 + pow (v1.1 AlphaPool)
        ("cuda/pearl_commit_lib.cu", "pearl_commit_lib.o"), // commitment (tensor_hash officiel)
        ("cuda/pearl_noise_lib.cu", "pearl_noise_lib.o"),   // noise structuré (port spec)
        ("cuda/pearl_resident_lib.cu", "pearl_resident_lib.o"), // gen+commit+stir résident
        ("cuda/pearl_gpu_proof_lib.cu", "pearl_gpu_proof_lib.o"), // preuve Merkle FULL GPU (v2.0)
    ];
    let lib = format!("{}/libpearl_gpu.a", out);
    // v0.6.3-beta : build FATBIN multi-archi → un seul binaire pour RTX 30 (sm_86),
    // 40 (sm_89) et 50 (sm_120a). Le path TMA (Blackwell) compile partout (CuTe trappe
    // ses intrinsics SM90 sur les archis sans TMA, jamais appelés) ; le path cp.async
    // (portable) sert Ampere/Ada via dispatch runtime (cudaDeviceProp.major).
    // Override : ARIA_CUDA_ARCHES="86,89,120a" (défaut). "120a" seul = build mono-5080 (perf dev).
    let arches = env::var("ARIA_CUDA_ARCHES").unwrap_or_else(|_| "86,89,120a".into());
    let mut gencode: Vec<String> = Vec::new();
    for a in arches.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        gencode.push("-gencode".into());
        gencode.push(format!("arch=compute_{a},code=sm_{a}"));
    }
    // Flags nvcc additionnels (ex : ARIA_NVCC_EXTRA=-DARIA_DUMP_TRANSCRIPT pour jackpot_diff)
    let extra: Vec<String> = env::var("ARIA_NVCC_EXTRA").map(|v| v.split_whitespace().map(String::from).collect()).unwrap_or_default();
    for (src, obj_name) in units {
        let obj = format!("{}/{}", out, obj_name);
        let mut args: Vec<String> = gencode.clone();
        args.extend(extra.iter().cloned());
        args.extend(["-O3", "-std=c++17",
                   "-Icuda", &format!("-I{}", shim), &format!("-I{}", cutlass), &format!("-I{}", csrc),
                   "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
                   "-c", src, "-o", &obj].iter().map(|s| s.to_string()));
        let status = Command::new(&nvcc).args(&args)
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
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_2x64_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_kernel_2x64.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_kernel_tma.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_kernel_cpasync_ms.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_kernel_sm75.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_kernel_sm75_dual.cuh");
    println!("cargo:rerun-if-env-changed=ARIA_CUDA_ARCHES");
    println!("cargo:rerun-if-changed=cuda/pearl_fold.cuh");
    println!("cargo:rerun-if-changed=cuda/pearl_commit_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_noise_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_resident_lib.cu");
    println!("cargo:rerun-if-changed=cuda/pearl_gpu_proof_lib.cu");
}
