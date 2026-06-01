# ARIAMiner — HiveOS Flight Sheet (clean)

Create a **Flight Sheet** in HiveOS with these exact values. Fields map 1:1 to
the integration scripts (`h-config.sh`).

## Wallet (add it once, in Wallets)
- **Coin:** any (HiveOS has no PRL — pick e.g. `CustomCoin`; it isn't used, the
  real wallet/pool come from the miner config below).
- **Address:** your Pearl address — `prl1…………`
- **Name:** e.g. `pearl-prl`

## Flight Sheet
| Field | Value |
|-------|-------|
| **Coin** | (the CustomCoin above) |
| **Wallet** | the `pearl-prl` wallet you added |
| **Pool** | `Configure in miner` |
| **Miner** | `Custom` |

### Setup Miner Config (Custom)
| Field | Value |
|-------|-------|
| **Miner name** | `ariaminer` |
| **Installation URL** | `https://pool.ariabrain.com/downloads/ariaminer-hiveos-0.2.0.tar.gz` |
| **Hash algorithm** | `gemm-int7` |
| **Wallet and worker template** | `%WAL%` |
| **Pool URL** | `pearl.ariabrain.com:3334` |
| **Pass** | `x` |
| **Extra config arguments** | `--threads 23` *(optional; default = all cores)* |
| **User config** | *(leave empty)* |

## Apply
Save the flight sheet and assign it to your rig(s). ARIAMiner downloads, starts,
and reports **hashrate / accepted shares / uptime** to the HiveOS dashboard
automatically (via the miner's `--stats-port`). Auto-reconnect is built in — if
the pool restarts, the rig reconnects on its own.

## Tips
- `%WAL%` resolves to your Pearl address; `%WORKER%`/the rig name is passed
  automatically as the worker label (no need to add it to the template).
- Leave a thread or two free on busy rigs: `--threads <cores-1>`.
- Hashrate is difficulty-weighted (same TH/s convention as the pool); it climbs
  for a few minutes after start while vardiff settles, then stabilises.
