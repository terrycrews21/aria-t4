# ARIAMiner

**Open-source, 0% dev-fee CPU miner for Pearl (PRL).**

ARIAMiner mines Pearl's **GEMM Int7×Int7** proof-of-work on your CPU and submits
real, node-validated `PlainProof` shares — not shortcuts. No GPU required, no
hidden fees: the miner takes **0%**.

- **0% dev-fee** — the binary mines 100% to *your* wallet.
- **Canonical proofs** — every accepted share is a genuine proof that passes the
  node-side `verify_plain_proof`, so a pool can assemble the block and
  `submitblock` on a hit. Output is **bit-identical** to the Pearl reference.
- **CPU only** — runs on any modern x86-64 box, no CUDA / no GPU.
- **One portable binary** — auto-detects and uses **AVX-512 VNNI** (AMD Zen 4/5,
  Intel with AVX-512) or **AVX-VNNI** (Intel Core Ultra / Arrow Lake) at runtime,
  with a scalar fallback on older CPUs.

## Algorithm

Each attempt draws noise-perturbed secret strips, computes a sparse
**GEMM Int7×Int7 → Int32** "jackpot" over them, and runs a **keyed BLAKE3** hash
of that transcript against the share / network target. The Int7 matrices are
multiplied on the hardware int8 datapath (`vpdpbusd`), which is bit-exact to the
reference, so shares validate normally. The heavy matmul is what makes the work
hard to accelerate cheaply; BLAKE3 provides the difficulty test.

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
| `--password` | `x` | Stratum password (`x;d=N` for static difficulty) |

**Tip:** on a desktop, leaving one or two threads free
(`--threads <cores-1>`) often gives a smoother system and equal hashrate.

## Where to mine

Point it at any Pearl pool that speaks the Pearl Stratum v1 dialect. One such
pool is **AriaPool** — `pearl.ariabrain.com:3334` (1% pool fee), with a live
dashboard at <https://pool.ariabrain.com/pearl.html>.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome.
