# ARIAMiner

**Open-source, 0% dev-fee CPU miner for Pearl (PRL).**

ARIAMiner turns Pearl's GEMM Int7×Int7 proof-of-work into real CPU throughput —
an amortized grind that does one BLAKE3 setup and sweeps thousands of matmul
tiles per attempt. No GPU required, no hidden fees: the miner takes **0%**.

- **0% dev-fee** — the binary mines 100% to *your* wallet.
- **CPU only** — runs on any modern x86-64 box, no CUDA / no GPU.
- **One portable binary** — auto-detects and uses **AVX-512** (AMD Zen 4/5,
  Intel with AVX-512) or **AVX-VNNI / AVX2** (Intel Core Ultra / Arrow Lake) at
  runtime, with a scalar fallback.
- **Register-blocked kernel** — 16 independent accumulators hide the `vpdpbusd`
  latency; the per-attempt noise/commitment setup is fused and amortized over a
  large tile batch.

## Algorithm

Each attempt computes a sparse **GEMM Int7×Int7 → Int32** "jackpot" over
noise-perturbed secret strips, then runs a **keyed BLAKE3** hash of that
transcript against the network/share target. The heavy matmul is what makes the
work ASIC-resistant; BLAKE3 provides the difficulty test. The kernel output is
**bit-identical** to the Pearl reference implementation, so shares validate
normally.

## Quick start

### Option 1 — prebuilt binary

Download the latest build from the pool page, then point it at a pool:

```bash
chmod +x ariaminer-linux-x86_64
./ariaminer-linux-x86_64 \
  --pool pearl.ariabrain.com:3334 \
  --wallet <YOUR_PRL_ADDRESS> \
  --worker my-cpu-01
```

### Option 2 — build from source

Requires a recent Rust toolchain (`rustup`, stable).

```bash
git clone https://github.com/stefancrypto68/ariaminer
cd ariaminer
cargo build --release
./target/release/ariaminer \
  --pool pearl.ariabrain.com:3334 \
  --wallet <YOUR_PRL_ADDRESS> \
  --worker my-cpu-01
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pool` | *(required)* | Pool endpoint `host:port` |
| `--wallet` | *(required)* | Your Pearl receive address (`prl1...`) |
| `--worker` | `aria` | Worker / rig label |
| `--threads` | all logical cores | Number of grind threads |
| `--batch` | `4096` | Amortization batch (rows/cols drawn per setup) |
| `--password` | `x` | Stratum password (`x;d=N` for static difficulty) |

**Tuning tips**

- `--batch 4096` is the sweet spot on most CPUs (the per-setup noise generation
  amortizes over `batch² / 128` tiles). Smaller batches under-feed the kernel;
  larger ones plateau and start thrashing cache.
- The cache-blocking panel size defaults to a value tuned for a large L3. On
  CPUs with smaller L3 you can experiment with the `ARIA_PANEL_RG` environment
  variable, but the default is fine on most chips.
- On a desktop, leaving one thread free (`--threads <cores-1>`) often gives a
  smoother system and equal or better hashrate.

## Where to mine

Point it at any Pearl pool that speaks the Pearl Stratum v1 dialect. One such
pool is **AriaPool** — `pearl.ariabrain.com:3334` (1% pool fee), with a live
dashboard at <https://pool.ariabrain.com/pearl.html>.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome.
