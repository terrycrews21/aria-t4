//! FFI vers `libpearl_gpu` (kernel GPU CuTe : GEMM IMMA + fold jackpot + pow-check blake3).
//! Le CPU fournit a_eff/b_eff (signal + noise déjà appliqués via `add_noise_into_fast`),
//! la clé pow (`a_noise_seed`) et le bound ; le GPU renvoie les candidats (hash ≤ bound).
use std::os::raw::c_int;

unsafe extern "C" {
    fn pearl_gpu_device_major() -> c_int;

    fn pearl_gpu_grind(
        a_eff: *const i8, b_eff: *const i8, m: c_int, n: c_int, k: c_int,
        pow_key: *const u32, pow_bound: *const u32,
        hit_rows: *mut c_int, hit_cols: *mut c_int, max_hits: c_int,
    ) -> c_int;

    fn pearl_gpu_grind_2x64(
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
    fn pearl_resident_create_2x64(m: c_int, n: c_int, k: c_int, max_hits: c_int) -> *mut core::ffi::c_void;
    fn pearl_resident_grind_ctx(
        ctx: *mut core::ffi::c_void, setup_seed: u64,
        job_key: *const u8, pow_bound_le: *const u8,
        hit_rows: *mut c_int, hit_cols: *mut c_int,
    ) -> c_int;
    fn pearl_resident_grind2_ctx(
        ctx: *mut core::ffi::c_void, setup_seed: u64, next_seed: u64, has_next: c_int,
        job_key: *const u8, pow_bound_le: *const u8,
        hit_rows: *mut c_int, hit_cols: *mut c_int,
    ) -> c_int;
    fn pearl_resident_destroy(ctx: *mut core::ffi::c_void);
    fn pearl_resident_set_timing(ctx: *mut core::ffi::c_void, on: c_int);
    fn pearl_resident_last_times(ctx: *mut core::ffi::c_void, prologue_ms: *mut f32, grind_ms: *mut f32);
    fn pearl_resident_last_times4(ctx: *mut core::ffi::c_void, genc_ms: *mut f32, noise_ms: *mut f32, grind_ms: *mut f32);

    fn pearl_proof_create(max_rows: c_int, k: c_int, max_leafdata: c_int) -> *mut core::ffi::c_void;
    fn pearl_proof_build(
        ctx: *mut core::ffi::c_void, setup_seed: u64, sel: c_int, rows: c_int, k: c_int,
        job_key: *const u8, leaf_indices: *const i64, n_leaves: c_int,
        out_leaf_data: *mut u8, out_layers: *mut u8, out_layer_lens: *mut i64,
        out_num_layers: *mut c_int, out_root: *mut u8,
    ) -> c_int;
    fn pearl_proof_build_tree(
        ctx: *mut core::ffi::c_void, setup_seed: u64, sel: c_int, rows: c_int, k: c_int,
        job_key: *const u8, out_layers: *mut u8, out_layer_lens: *mut i64,
        out_num_layers: *mut c_int, out_root: *mut u8,
    ) -> c_int;
    fn pearl_proof_gather(
        ctx: *mut core::ffi::c_void, leaf_indices: *const i64, n_leaves: c_int, out_leaf_data: *mut u8,
    ) -> c_int;
    fn pearl_proof_destroy(ctx: *mut core::ffi::c_void);

    fn pearl_resident2_create(m: c_int, n: c_int, k: c_int) -> *mut core::ffi::c_void;
    fn pearl_resident2_grind_batch(
        ctx: *mut core::ffi::c_void, base_seed: u64, job_key: *const u8, bound_le: *const u8,
        num: c_int, out_idx: *mut c_int, out_rows: *mut c_int, out_cols: *mut c_int, max_out: c_int,
    ) -> c_int;
    fn pearl_resident2_destroy(ctx: *mut core::ffi::c_void);
    fn pearl_resident2_set_grind_blocks(ctx: *mut core::ffi::c_void, g: c_int);
    fn pearl_gpu_sm_count() -> c_int;
}

/// CUDA compute-capability major of the active device (7=Turing, 8=Ampere/Ada,
/// 9=Hopper, 10=Blackwell datacenter, 12=Blackwell consumer).
pub fn device_major() -> i32 { unsafe { pearl_gpu_device_major() as i32 } }

/// Nombre de SM du GPU 0 (pour calibrer le headroom du grind persistant).
pub fn gpu_sm_count() -> usize { (unsafe { pearl_gpu_sm_count() }).max(0) as usize }

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
    /// v0.5.0 : grille du grind persistant. 0 = grille pleine ; >0 = G blocs grid-stride
    /// (occupation bridée → le prologue du slot suivant peut tourner en concurrence).
    pub fn set_grind_blocks(&self, g: usize) { unsafe { pearl_resident2_set_grind_blocks(self.ptr, g as c_int) } }
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
    /// Variante **2×64** (AlphaPool, v1.1) : pipeline résident identique mais le grind
    /// final folde en tuile 2 lignes {r,r+32} × 64 cols. `grind`/`dims`/`Drop` partagés.
    pub fn new_2x64(m: usize, n: usize, k: usize, max_hits: usize) -> Self {
        let ptr = unsafe { pearl_resident_create_2x64(m as c_int, n as c_int, k as c_int, max_hits as c_int) };
        assert!(!ptr.is_null(), "pearl_resident_create_2x64 a échoué (VRAM ?)");
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
    /// v0.6.2 : grind avec OVERLAP PROLOGUE — le prologue de `next_seed` est préfetché
    /// pendant ce grind (chemin LuckyPool ARIA_TMA_MS+ARIA_FIXB ; sinon identique à `grind`).
    pub fn grind2(&self, setup_seed: u64, next_seed: u64, job_key: &[u8; 32], bound_le: &[u8; 32]) -> (i32, Vec<Hit>) {
        let mut hr = vec![0i32; self.max_hits * 128];
        let mut hc = vec![0i32; self.max_hits * 128];
        let found = unsafe {
            pearl_resident_grind2_ctx(self.ptr, setup_seed, next_seed, 1,
                job_key.as_ptr(), bound_le.as_ptr(), hr.as_mut_ptr(), hc.as_mut_ptr())
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

    /// Active/désactive la mesure per-phase (cudaEvents). Hors chemin chaud.
    pub fn set_timing(&self, on: bool) { unsafe { pearl_resident_set_timing(self.ptr, on as c_int) } }
    /// Derniers temps GPU (ms) du dernier `grind` : (prologue gen+commit+noise, grind GEMM).
    pub fn last_times(&self) -> (f32, f32) {
        let (mut p, mut g) = (0f32, 0f32);
        unsafe { pearl_resident_last_times(self.ptr, &mut p, &mut g) };
        (p, g)
    }
    /// Split fin du dernier grind : (gen+commit+stir, noise, grind) en ms.
    pub fn last_times4(&self) -> (f32, f32, f32) {
        let (mut gc, mut no, mut gr) = (0f32, 0f32, 0f32);
        unsafe { pearl_resident_last_times4(self.ptr, &mut gc, &mut no, &mut gr) };
        (gc, no, gr)
    }
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

/// Arbre Merkle keyé d'une matrice signal, construit ENTIÈREMENT sur GPU et rapatrié
/// (toutes les couches + les chunks de feuilles demandés). Le CPU n'a plus qu'à faire le
/// walk d'indices (= `MerkleTree::get_multileaf_proof`), aucun hachage. `layers` = octets,
/// nœud `j` de la couche `l` = `layers[(off(l)+j)*32 .. +32]` où `off(l)=Σ layer_lens[0..l]`.
pub struct ProofRaw {
    pub leaf_data: Vec<u8>,       // n_leaves * 1024
    pub layers: Vec<u8>,          // total_nodes * 32
    pub layer_lens: Vec<usize>,   // nœuds par couche (layer_lens[0] = total_leaves)
    pub root: [u8; 32],
}

/// Contexte de preuve GPU PERSISTANT : buffers alloués 1× (signal + couches), réutilisés à
/// chaque `build`. Un par thread builder (chacun son stream) → preuves en parallèle.
pub struct ProofGpuCtx {
    ptr: *mut core::ffi::c_void,
    max_rows: usize,
    k: usize,
    max_leafdata: usize,
}
unsafe impl Send for ProofGpuCtx {}
impl ProofGpuCtx {
    pub fn new(max_rows: usize, k: usize, max_leafdata: usize) -> Self {
        let ptr = unsafe { pearl_proof_create(max_rows as c_int, k as c_int, max_leafdata as c_int) };
        assert!(!ptr.is_null(), "pearl_proof_create a échoué (VRAM ?)");
        Self { ptr, max_rows, k, max_leafdata }
    }

    /// Construit l'arbre de la matrice `sel` (0=A, 1=Bᵀ), `rows`×`k`, depuis `setup_seed` clé
    /// `job_key`, et renvoie les couches + les chunks bruts des `leaf_indices` (triés/uniques).
    pub fn build(&self, setup_seed: u64, sel: u32, rows: usize, k: usize,
                 job_key: &[u8; 32], leaf_indices: &[usize]) -> ProofRaw {
        assert_eq!(k, self.k);
        assert!(rows <= self.max_rows);
        assert!(!leaf_indices.is_empty() && leaf_indices.len() <= self.max_leafdata);
        let num_leaves = rows * k / 1024;
        let li64: Vec<i64> = leaf_indices.iter().map(|&x| x as i64).collect();
        let mut leaf_data = vec![0u8; leaf_indices.len() * 1024];
        let mut layers = vec![0u8; (2 * num_leaves + 8) * 32];
        let mut layer_lens = vec![0i64; 64];
        let mut num_layers: c_int = 0;
        let mut root = [0u8; 32];
        let rc = unsafe {
            pearl_proof_build(
                self.ptr, setup_seed, sel as c_int, rows as c_int, k as c_int,
                job_key.as_ptr(), li64.as_ptr(), li64.len() as c_int,
                leaf_data.as_mut_ptr(), layers.as_mut_ptr(), layer_lens.as_mut_ptr(),
                &mut num_layers as *mut c_int, root.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "pearl_proof_build a échoué (rc={rc})");
        let lens: Vec<usize> = layer_lens[..num_layers as usize].iter().map(|&x| x as usize).collect();
        let total_nodes: usize = lens.iter().sum();
        layers.truncate(total_nodes * 32);
        ProofRaw { leaf_data, layers, layer_lens: lens, root }
    }

    /// v0.4.2 — bâtit l'arbre de `sel` (rows×k) UNE fois (signal + couches résidents dans le ctx),
    /// renvoie couches + lens + racine SANS gather. À 131072 on appelle ça 1×/setup puis [`gather`]
    /// par hit (les ~256 hits du setup partagent la même matrice → arbre amorti).
    pub fn build_tree(&self, setup_seed: u64, sel: u32, rows: usize, k: usize,
                      job_key: &[u8; 32]) -> (Vec<u8>, Vec<usize>, [u8; 32]) {
        assert_eq!(k, self.k);
        assert!(rows <= self.max_rows);
        let num_leaves = rows * k / 1024;
        let mut layers = vec![0u8; (2 * num_leaves + 8) * 32];
        let mut layer_lens = vec![0i64; 64];
        let mut num_layers: c_int = 0;
        let mut root = [0u8; 32];
        let rc = unsafe {
            pearl_proof_build_tree(self.ptr, setup_seed, sel as c_int, rows as c_int, k as c_int,
                job_key.as_ptr(), layers.as_mut_ptr(), layer_lens.as_mut_ptr(),
                &mut num_layers as *mut c_int, root.as_mut_ptr())
        };
        assert_eq!(rc, 0, "pearl_proof_build_tree a échoué (rc={rc})");
        let lens: Vec<usize> = layer_lens[..num_layers as usize].iter().map(|&x| x as usize).collect();
        let total_nodes: usize = lens.iter().sum();
        layers.truncate(total_nodes * 32);
        (layers, lens, root)
    }

    /// Ramasse les chunks bruts des feuilles `leaf_indices` depuis le signal résident
    /// (= dernier [`build_tree`] sur ce ctx). Négligeable → 1 appel par hit.
    pub fn gather(&self, leaf_indices: &[usize]) -> Vec<u8> {
        assert!(!leaf_indices.is_empty() && leaf_indices.len() <= self.max_leafdata);
        let li64: Vec<i64> = leaf_indices.iter().map(|&x| x as i64).collect();
        let mut leaf_data = vec![0u8; leaf_indices.len() * 1024];
        let rc = unsafe {
            pearl_proof_gather(self.ptr, li64.as_ptr(), li64.len() as c_int, leaf_data.as_mut_ptr())
        };
        assert_eq!(rc, 0, "pearl_proof_gather a échoué (rc={rc})");
        leaf_data
    }
}
impl Drop for ProofGpuCtx {
    fn drop(&mut self) { unsafe { pearl_proof_destroy(self.ptr) } }
}

/// Grind GPU **2×64** (AlphaPool) — fold jackpot sur tuile 2 lignes {r,r+32} × 64 cols.
/// Après dédup : `rows` = 2 indices, `cols` = 64 indices. Sinon identique à [`grind`].
pub fn grind_2x64(
    a_eff: &[i8], b_eff: &[i8], m: usize, n: usize, k: usize,
    pow_key: &[u32; 8], pow_bound: &[u32; 8], max_hits: usize,
) -> (i32, Vec<Hit>) {
    assert_eq!(a_eff.len(), m * k);
    assert_eq!(b_eff.len(), n * k);
    let mut hr = vec![0i32; max_hits * 128];
    let mut hc = vec![0i32; max_hits * 128];
    let found = unsafe {
        pearl_gpu_grind_2x64(
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
