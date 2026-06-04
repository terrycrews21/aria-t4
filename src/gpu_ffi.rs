//! FFI vers `libpearl_gpu` (kernel GPU CuTe : GEMM IMMA + fold jackpot + pow-check blake3).
//! Le CPU fournit a_eff/b_eff (signal + noise déjà appliqués via `add_noise_into_fast`),
//! la clé pow (`a_noise_seed`) et le bound ; le GPU renvoie les candidats (hash ≤ bound).
use std::os::raw::c_int;

unsafe extern "C" {
    fn pearl_gpu_grind(
        a_eff: *const i8, b_eff: *const i8, m: c_int, n: c_int, k: c_int,
        pow_key: *const u32, pow_bound: *const u32,
        hit_rows: *mut c_int, hit_cols: *mut c_int, max_hits: c_int,
    ) -> c_int;

    fn pearl_gpu_tensor_hash(
        data: *const u8, len: u32, key: *const u8, out: *mut u8,
    ) -> c_int;

    fn pearl_gpu_noise(
        seed_label: *const u8, key: *const u8, m: c_int, k: c_int, noise_out: *mut i8,
    ) -> c_int;

    fn pearl_gpu_commit_seeds(
        setup_seed: u64, m: c_int, n: c_int, k: c_int, job_key: *const u8,
        a_seed_out: *mut u8, b_seed_out: *mut u8,
    ) -> c_int;

    fn pearl_gpu_resident_grind(
        setup_seed: u64, m: c_int, n: c_int, k: c_int,
        job_key: *const u8, pow_bound_le: *const u8,
        hit_rows: *mut c_int, hit_cols: *mut c_int, max_hits: c_int,
    ) -> c_int;

    fn pearl_resident_create(m: c_int, n: c_int, k: c_int, max_hits: c_int) -> *mut core::ffi::c_void;
    fn pearl_resident_grind_ctx(
        ctx: *mut core::ffi::c_void, setup_seed: u64,
        job_key: *const u8, pow_bound_le: *const u8,
        hit_rows: *mut c_int, hit_cols: *mut c_int,
    ) -> c_int;
    fn pearl_resident_destroy(ctx: *mut core::ffi::c_void);

    fn pearl_resident2_create(m: c_int, n: c_int, k: c_int) -> *mut core::ffi::c_void;
    fn pearl_resident2_grind_batch(
        ctx: *mut core::ffi::c_void, base_seed: u64, job_key: *const u8, bound_le: *const u8,
        num: c_int, out_idx: *mut c_int, out_rows: *mut c_int, out_cols: *mut c_int, max_out: c_int,
    ) -> c_int;
    fn pearl_resident2_destroy(ctx: *mut core::ffi::c_void);
}

/// Contexte DOUBLE-BUFFER batché : 2 jeux de buffers + 2 streams, prologue(N+1)
/// overlap grind(N). `grind_batch` pipeline `num` setups en un appel.
pub struct ResidentCtx2 {
    ptr: *mut core::ffi::c_void,
    m: usize, n: usize, k: usize,
}
impl ResidentCtx2 {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        let ptr = unsafe { pearl_resident2_create(m as c_int, n as c_int, k as c_int) };
        assert!(!ptr.is_null(), "pearl_resident2_create a échoué (VRAM ?)");
        Self { ptr, m, n, k }
    }
    /// Pipeline `num` setups (seed = base_seed+i). Renvoie les hits : (setup_index, rows, cols).
    pub fn grind_batch(&self, base_seed: u64, job_key: &[u8; 32], bound_le: &[u8; 32], num: usize)
        -> Vec<(u64, Vec<usize>, Vec<usize>)> {
        let max_out = 128usize;
        let mut idx = vec![0i32; max_out];
        let mut rows = vec![0i32; max_out * 128];
        let mut cols = vec![0i32; max_out * 128];
        let got = unsafe {
            pearl_resident2_grind_batch(self.ptr, base_seed, job_key.as_ptr(), bound_le.as_ptr(),
                num as c_int, idx.as_mut_ptr(), rows.as_mut_ptr(), cols.as_mut_ptr(), max_out as c_int)
        };
        let nret = (got.max(0) as usize).min(max_out);
        let mut out = Vec::with_capacity(nret);
        for s in 0..nret {
            let mut r: Vec<usize> = rows[s*128..s*128+128].iter().map(|&x| x as usize).collect();
            let mut c: Vec<usize> = cols[s*128..s*128+128].iter().map(|&x| x as usize).collect();
            r.sort_unstable(); r.dedup(); c.sort_unstable(); c.dedup();
            out.push((base_seed + idx[s] as u64, r, c));
        }
        out
    }
    pub fn dims(&self) -> (usize, usize, usize) { (self.m, self.n, self.k) }
}
impl Drop for ResidentCtx2 {
    fn drop(&mut self) { unsafe { pearl_resident2_destroy(self.ptr) } }
}

/// Contexte résident PERSISTANT : alloue tous les buffers GPU 1× ; chaque `grind`
/// réutilise (gen→commit→noise→grind), zéro alloc/free par appel. C'est ça qui
/// permet au GPU de tourner à 100 %.
pub struct ResidentCtx {
    ptr: *mut core::ffi::c_void,
    m: usize, n: usize, k: usize, max_hits: usize,
}
impl ResidentCtx {
    pub fn new(m: usize, n: usize, k: usize, max_hits: usize) -> Self {
        let ptr = unsafe { pearl_resident_create(m as c_int, n as c_int, k as c_int, max_hits as c_int) };
        assert!(!ptr.is_null(), "pearl_resident_create a échoué (VRAM ?)");
        Self { ptr, m, n, k, max_hits }
    }
    /// Un grind résident. Renvoie (nb_candidats, hits).
    pub fn grind(&self, setup_seed: u64, job_key: &[u8; 32], bound_le: &[u8; 32]) -> (i32, Vec<Hit>) {
        let mut hr = vec![0i32; self.max_hits * 128];
        let mut hc = vec![0i32; self.max_hits * 128];
        let found = unsafe {
            pearl_resident_grind_ctx(self.ptr, setup_seed, job_key.as_ptr(), bound_le.as_ptr(),
                hr.as_mut_ptr(), hc.as_mut_ptr())
        };
        let nret = (found.max(0) as usize).min(self.max_hits);
        let mut hits = Vec::with_capacity(nret);
        for s in 0..nret {
            let mut rows: Vec<usize> = hr[s*128..s*128+128].iter().map(|&x| x as usize).collect();
            let mut cols: Vec<usize> = hc[s*128..s*128+128].iter().map(|&x| x as usize).collect();
            rows.sort_unstable(); rows.dedup(); cols.sort_unstable(); cols.dedup();
            hits.push(Hit { rows, cols });
        }
        (found, hits)
    }
    pub fn dims(&self) -> (usize, usize, usize) { (self.m, self.n, self.k) }
}
impl Drop for ResidentCtx {
    fn drop(&mut self) { unsafe { pearl_resident_destroy(self.ptr) } }
}

/// Pipeline résident COMPLET sur GPU : gen A,B (setup_seed) → commit → stir →
/// noise → grind. Zéro upload a_eff/b_eff. Sur win, le CPU regénère les strips
/// via `int7_at(setup_seed,..)`. Renvoie (nb_candidats, hits rows×cols globaux).
pub fn resident_grind(
    setup_seed: u64, m: usize, n: usize, k: usize,
    job_key: &[u8; 32], bound_le: &[u8; 32], max_hits: usize,
) -> (i32, Vec<Hit>) {
    let mut hr = vec![0i32; max_hits * 128];
    let mut hc = vec![0i32; max_hits * 128];
    let found = unsafe {
        pearl_gpu_resident_grind(
            setup_seed, m as c_int, n as c_int, k as c_int,
            job_key.as_ptr(), bound_le.as_ptr(),
            hr.as_mut_ptr(), hc.as_mut_ptr(), max_hits as c_int,
        )
    };
    let nret = (found.max(0) as usize).min(max_hits);
    let mut hits = Vec::with_capacity(nret);
    for s in 0..nret {
        let mut rows: Vec<usize> = hr[s * 128..s * 128 + 128].iter().map(|&x| x as usize).collect();
        let mut cols: Vec<usize> = hc[s * 128..s * 128 + 128].iter().map(|&x| x as usize).collect();
        rows.sort_unstable(); rows.dedup();
        cols.sort_unstable(); cols.dedup();
        hits.push(Hit { rows, cols });
    }
    (found, hits)
}

/// Génère A,B int7 (setup_seed) sur GPU, commit (tensor_hash keyé job_key), stir
/// → (a_seed, b_seed). Doit == `compute_commitment_hash` CPU sur les mêmes A,B.
pub fn commit_seeds(setup_seed: u64, m: usize, n: usize, k: usize, job_key: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
    let mut a_seed = [0u8; 32];
    let mut b_seed = [0u8; 32];
    let rc = unsafe {
        pearl_gpu_commit_seeds(setup_seed, m as c_int, n as c_int, k as c_int,
            job_key.as_ptr(), a_seed.as_mut_ptr(), b_seed.as_mut_ptr())
    };
    assert_eq!(rc, 0, "pearl_gpu_commit_seeds a échoué (rc={rc})");
    (a_seed, b_seed)
}

/// Signal int7 reproductible — MÊME formule que le kernel GPU `int7_at`
/// (splitmix64). Pour regénérer côté CPU les strips gagnants sur un hit.
pub fn int7_at(seed: u64, sel: u32, i: u32, j: u32) -> i8 {
    let mut x = seed ^ ((sel as u64) << 62) ^ ((i as u64) << 32) ^ (j as u64);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x & 0x7F) as i64 - 64) as i8
}

/// Noise structuré Pearl sur GPU pour les lignes 0..m.
/// `seed_label` = "A_tensor"/"B_tensor" paddé 32o ; `key` = a/b_noise_seed.
/// Doit être bit-exact avec `compute_noise_for_indices(...).a` (rows 0..m).
pub fn noise(seed_label: &[u8; 32], key: &[u8; 32], m: usize, k: usize) -> Vec<i8> {
    let mut out = vec![0i8; m * k];
    let rc = unsafe {
        pearl_gpu_noise(seed_label.as_ptr(), key.as_ptr(), m as c_int, k as c_int, out.as_mut_ptr())
    };
    assert_eq!(rc, 0, "pearl_gpu_noise a échoué (rc={rc})");
    out
}

/// Commitment GPU : blake3 keyed tree de `data` via le `tensor_hash` officiel.
/// Doit être bit-exact avec `pearl_blake3::blake3_digest(data, Some(key))`.
/// `data.len()` doit être > 131072 (contrainte du kernel officiel).
pub fn tensor_hash(data: &[u8], key: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let rc = unsafe {
        pearl_gpu_tensor_hash(data.as_ptr(), data.len() as u32, key.as_ptr(), out.as_mut_ptr())
    };
    assert_eq!(rc, 0, "pearl_gpu_tensor_hash a échoué (rc={rc})");
    out
}

/// Un candidat trouvé par le GPU : la tuile gagnante = `rows` (8 lignes globales, triées)
/// × `cols` (16 colonnes globales, triées). Directement utilisable par `make_proof`.
#[derive(Clone, Debug)]
pub struct Hit {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
}

/// Lance le grind GPU sur les matrices effectives.
/// `a_eff`: m·k int8 row-major. `b_eff`: n·k int8 row-major (= Bᵀ). m,n mult. 128, k mult. 64.
/// Renvoie `(nb_candidats, hits)` (hits tronqué à `max_hits`).
pub fn grind(
    a_eff: &[i8], b_eff: &[i8], m: usize, n: usize, k: usize,
    pow_key: &[u32; 8], pow_bound: &[u32; 8], max_hits: usize,
) -> (i32, Vec<Hit>) {
    assert_eq!(a_eff.len(), m * k);
    assert_eq!(b_eff.len(), n * k);
    let mut hr = vec![0i32; max_hits * 128];
    let mut hc = vec![0i32; max_hits * 128];
    let found = unsafe {
        pearl_gpu_grind(
            a_eff.as_ptr(), b_eff.as_ptr(), m as c_int, n as c_int, k as c_int,
            pow_key.as_ptr(), pow_bound.as_ptr(),
            hr.as_mut_ptr(), hc.as_mut_ptr(), max_hits as c_int,
        )
    };
    let nret = (found.max(0) as usize).min(max_hits);
    let mut hits = Vec::with_capacity(nret);
    for s in 0..nret {
        let mut rows: Vec<usize> = hr[s * 128..s * 128 + 128].iter().map(|&x| x as usize).collect();
        let mut cols: Vec<usize> = hc[s * 128..s * 128 + 128].iter().map(|&x| x as usize).collect();
        rows.sort_unstable(); rows.dedup();
        cols.sort_unstable(); cols.dedup();
        hits.push(Hit { rows, cols });
    }
    (found, hits)
}
