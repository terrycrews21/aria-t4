//! Mouchard — télémétrie LIVE haute précision du mineur GPU Pearl.
//!
//! But (approche A du chantier perf) : instrumenter le binaire qui marche le mieux
//! (151 TH/s pool) pour SAVOIR, au lieu de tâtonner, où fuient les TH/s entre nous
//! (151 live) et le pro alpha (174.78 live), sachant que le plafond physique du GEMM
//! pur est 217. Le juge de vérité reste **le TH/s crédité par la pool** ; le mouchard
//! éclaire la décomposition (prologue vs grind vs hôte vs build-preuve) + l'état HW.
//!
//! Contraintes :
//! - **Zéro perturbation par défaut** : tout est gated par `ARIA_MOUCHARD=1`. Sans la
//!   variable, `enabled()==false` → aucune mesure, aucun coût → le 151 reste intact.
//!   (On valide en A/B que le mouchard ON ne baisse pas le TH/s pool.)
//! - **Sans verrou sur le chemin chaud** : accumulateurs atomiques (add/max), lus+remis
//!   à zéro par fenêtre via `swap`. Le sampler HW tourne sur son propre thread.
//! - Sortie : 1 ligne JSONL structurée par fenêtre (~10 s) dans `ARIA_MOUCHARD_LOG`
//!   (déf `~/pearl-rnd-logs/mouchard.jsonl`) → exploitable ensuite par l'optimiseur (B).

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[inline]
fn ms_to_ns(ms: f32) -> u64 {
    if ms.is_finite() && ms > 0.0 { (ms as f64 * 1.0e6) as u64 } else { 0 }
}

/// Collecteur de télémétrie. Un seul, partagé (`Arc`) entre le grind, les builders
/// et le reporter.
pub struct Mouchard {
    enabled: bool,
    log_path: String,

    // --- accumulateurs par fenêtre (remis à 0 à chaque flush via swap) ---
    setups: AtomicU64,        // nb de grinds dans la fenêtre
    genc_ns: AtomicU64,       // Σ gen+commit+stir (ns GPU)
    noise_ns: AtomicU64,      // Σ noise (ns GPU)
    grind_ns: AtomicU64,      // Σ GEMM+fold+powcheck (ns GPU)
    wall_ns: AtomicU64,       // Σ wall-time hôte par setup (ns) — révèle les bulles
    grind_ns_max: AtomicU64,  // pic grind (ns)
    found_total: AtomicU64,   // Σ candidats (hits bruts) trouvés
    builds: AtomicU64,        // nb de preuves construites (builders)
    build_ns: AtomicU64,      // Σ temps build_proof_from_hit (ns)
    build_ns_max: AtomicU64,  // pic build (ns)
    drops: AtomicU64,         // hits lâchés (canal builder saturé)

    // --- snapshot HW (maj par le sampler, lu au flush) ---
    hw_sm_util: AtomicU64,    // %
    hw_mem_util: AtomicU64,   // %
    hw_power_mw: AtomicU64,   // milliwatts
    hw_sm_clk: AtomicU64,     // MHz
    hw_mem_clk: AtomicU64,    // MHz
    hw_temp: AtomicU64,       // °C
    hw_have: AtomicBool,      // au moins un échantillon ?
}

impl Mouchard {
    /// Construit depuis l'environnement. `ARIA_MOUCHARD=1` active la collecte.
    pub fn new() -> Self {
        let enabled = std::env::var("ARIA_MOUCHARD").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
        let log_path = std::env::var("ARIA_MOUCHARD_LOG").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{home}/pearl-rnd-logs/mouchard.jsonl")
        });
        Self::with(enabled, log_path)
    }

    /// Instance désactivée (pour l'auto-tune : pas de pollution des fenêtres réelles).
    pub fn disabled() -> Self { Self::with(false, String::new()) }

    fn with(enabled: bool, log_path: String) -> Self {
        Mouchard {
            enabled, log_path,
            setups: AtomicU64::new(0), genc_ns: AtomicU64::new(0), noise_ns: AtomicU64::new(0),
            grind_ns: AtomicU64::new(0), wall_ns: AtomicU64::new(0), grind_ns_max: AtomicU64::new(0),
            found_total: AtomicU64::new(0), builds: AtomicU64::new(0), build_ns: AtomicU64::new(0),
            build_ns_max: AtomicU64::new(0), drops: AtomicU64::new(0),
            hw_sm_util: AtomicU64::new(0), hw_mem_util: AtomicU64::new(0), hw_power_mw: AtomicU64::new(0),
            hw_sm_clk: AtomicU64::new(0), hw_mem_clk: AtomicU64::new(0), hw_temp: AtomicU64::new(0),
            hw_have: AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool { self.enabled }

    /// Un grind GPU mesuré. `*_ms` viennent de `ResidentCtx::last_times4` (cudaEvents),
    /// `wall_ns` du chrono hôte autour de l'appel, `found` = candidats bruts.
    #[inline]
    pub fn record_setup(&self, genc_ms: f32, noise_ms: f32, grind_ms: f32, wall_ns: u64, found: u32) {
        if !self.enabled { return; }
        self.setups.fetch_add(1, Ordering::Relaxed);
        self.genc_ns.fetch_add(ms_to_ns(genc_ms), Ordering::Relaxed);
        self.noise_ns.fetch_add(ms_to_ns(noise_ms), Ordering::Relaxed);
        let g = ms_to_ns(grind_ms);
        self.grind_ns.fetch_add(g, Ordering::Relaxed);
        self.grind_ns_max.fetch_max(g, Ordering::Relaxed);
        self.wall_ns.fetch_add(wall_ns, Ordering::Relaxed);
        self.found_total.fetch_add(found as u64, Ordering::Relaxed);
    }

    /// Une preuve construite hors chemin grind (thread builder).
    #[inline]
    pub fn record_build(&self, ns: u64) {
        if !self.enabled { return; }
        self.builds.fetch_add(1, Ordering::Relaxed);
        self.build_ns.fetch_add(ns, Ordering::Relaxed);
        self.build_ns_max.fetch_max(ns, Ordering::Relaxed);
    }

    /// Un hit lâché parce que le canal builder était saturé (perte de share potentielle).
    #[inline]
    pub fn record_drop(&self) {
        if !self.enabled { return; }
        self.drops.fetch_add(1, Ordering::Relaxed);
    }

    /// Démarre le sampler HW (nvidia-smi toutes les ~1 s) sur un thread détaché.
    /// No-op si désactivé.
    pub fn start_hw_sampler(self: &Arc<Self>) {
        if !self.enabled { return; }
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            loop {
                if let Some((sm, mem, pw, sclk, mclk, temp)) = sample_nvidia_smi() {
                    me.hw_sm_util.store(sm, Ordering::Relaxed);
                    me.hw_mem_util.store(mem, Ordering::Relaxed);
                    me.hw_power_mw.store(pw, Ordering::Relaxed);
                    me.hw_sm_clk.store(sclk, Ordering::Relaxed);
                    me.hw_mem_clk.store(mclk, Ordering::Relaxed);
                    me.hw_temp.store(temp, Ordering::Relaxed);
                    me.hw_have.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
        });
    }

    /// Lit + remet à zéro les accumulateurs de la fenêtre, croise avec le HW et la
    /// vérité-terrain pool (`display_th`, `setups_s`, `shares_total`), écrit une ligne
    /// JSONL et renvoie un résumé compact pour stdout. À appeler par le reporter.
    pub fn flush_window(&self, up_s: u64, display_th: f64, setups_s: f64, shares_total: u64) -> Option<String> {
        if !self.enabled { return None; }
        let n = self.setups.swap(0, Ordering::Relaxed);
        let genc = self.genc_ns.swap(0, Ordering::Relaxed);
        let noise = self.noise_ns.swap(0, Ordering::Relaxed);
        let grind = self.grind_ns.swap(0, Ordering::Relaxed);
        let wall = self.wall_ns.swap(0, Ordering::Relaxed);
        let gmax = self.grind_ns_max.swap(0, Ordering::Relaxed);
        let found = self.found_total.swap(0, Ordering::Relaxed);
        let builds = self.builds.swap(0, Ordering::Relaxed);
        let bns = self.build_ns.swap(0, Ordering::Relaxed);
        let bmax = self.build_ns_max.swap(0, Ordering::Relaxed);
        let drops = self.drops.swap(0, Ordering::Relaxed);

        let nf = n.max(1) as f64;
        // moyennes par setup, en ms
        let genc_ms = genc as f64 / nf / 1e6;
        let noise_ms = noise as f64 / nf / 1e6;
        let grind_ms = grind as f64 / nf / 1e6;
        let wall_ms = wall as f64 / nf / 1e6;
        let gpu_ms = genc_ms + noise_ms + grind_ms;
        // bulle hôte = temps mur non couvert par le GPU (lancement, FFI, rng, copies)
        let bubble_ms = (wall_ms - gpu_ms).max(0.0);
        let grind_max_ms = gmax as f64 / 1e6;
        // part du temps GPU passée DANS le grind utile (tensor cores) vs prologue
        let grind_share = if gpu_ms > 0.0 { grind_ms / gpu_ms } else { 0.0 };
        let build_avg_ms = if builds > 0 { bns as f64 / builds as f64 / 1e6 } else { 0.0 };
        let build_max_ms = bmax as f64 / 1e6;

        let (have_hw, sm, mem, pw_w, sclk, mclk, temp) = (
            self.hw_have.load(Ordering::Relaxed),
            self.hw_sm_util.load(Ordering::Relaxed),
            self.hw_mem_util.load(Ordering::Relaxed),
            self.hw_power_mw.load(Ordering::Relaxed) as f64 / 1000.0,
            self.hw_sm_clk.load(Ordering::Relaxed),
            self.hw_mem_clk.load(Ordering::Relaxed),
            self.hw_temp.load(Ordering::Relaxed),
        );
        // efficience : TH/s par watt (×1000 pour lisibilité = GH/J ~ TH/s/kW)
        let gh_per_w = if have_hw && pw_w > 1.0 { display_th / pw_w } else { 0.0 };

        let json = format!(
            "{{\"up_s\":{up_s},\"win_setups\":{n},\"display_th\":{display_th:.3},\"setups_s\":{setups_s:.1},\"shares\":{shares_total},\
\"genc_ms\":{genc_ms:.4},\"noise_ms\":{noise_ms:.4},\"grind_ms\":{grind_ms:.4},\"grind_max_ms\":{grind_max_ms:.4},\
\"gpu_ms\":{gpu_ms:.4},\"wall_ms\":{wall_ms:.4},\"bubble_ms\":{bubble_ms:.4},\"grind_share\":{grind_share:.4},\
\"found_per_setup\":{:.3},\"builds\":{builds},\"build_avg_ms\":{build_avg_ms:.3},\"build_max_ms\":{build_max_ms:.3},\"drops\":{drops},\
\"hw\":{},\"sm_util\":{sm},\"mem_util\":{mem},\"power_w\":{pw_w:.1},\"sm_clk\":{sclk},\"mem_clk\":{mclk},\"temp\":{temp},\"th_per_w\":{gh_per_w:.4}}}",
            found as f64 / nf,
            if have_hw { "true" } else { "false" },
        );
        self.append_line(&json);

        // résumé stdout (1 ligne lisible)
        Some(format!(
            "🔎 pro {genc_ms:.3}+{noise_ms:.3}ms · grind {grind_ms:.3}ms ({:.0}%) · bulle {bubble_ms:.3}ms · build {build_avg_ms:.2}ms×{builds} drop{drops} · SM {sm}% {pw_w:.0}W {sclk}MHz {temp}°C · {gh_per_w:.3} TH/s/W",
            grind_share * 100.0,
        ))
    }

    fn append_line(&self, line: &str) {
        if self.log_path.is_empty() { return; }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.log_path) {
            let _ = writeln!(f, "{line}");
        }
    }
}

impl Default for Mouchard {
    fn default() -> Self { Self::new() }
}

/// Un échantillon HW via `nvidia-smi` (GPU 0). Renvoie (sm_util%, mem_util%, power_mW,
/// sm_clk MHz, mem_clk MHz, temp °C). `None` si nvidia-smi indisponible/illisible.
fn sample_nvidia_smi() -> Option<(u64, u64, u64, u64, u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i", "0",
        ])
        .output().ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?;
    let f: Vec<f64> = line.split(',').map(|x| x.trim().parse::<f64>().unwrap_or(0.0)).collect();
    if f.len() < 6 { return None; }
    Some((f[0] as u64, f[1] as u64, (f[2] * 1000.0) as u64, f[3] as u64, f[4] as u64, f[5] as u64))
}
