# ARIAMiner on HiveOS

ARIAMiner runs as a HiveOS **Custom Miner**. This folder holds the integration
scripts (`h-manifest.conf`, `h-config.sh`, `h-run.sh`, `h-stats.sh`); the ready
package (scripts + the `ariaminer` binary) is the tarball linked from the pool
page.

## Install (rig side)
1. In HiveOS, create a **Flight Sheet** with a **Custom** miner.
2. **Installation URL**: the package tarball, e.g.
   `https://pool.ariabrain.com/downloads/ariaminer-hiveos-0.2.0.tar.gz`
3. Fill the fields:
   - **Miner name**: `ariaminer`
   - **Wallet and worker template**: your `prl1...` address
   - **Pool URL**: `pearl.ariabrain.com:3334`
   - **Pass**: `x` (optional)
   - **Extra config arguments** (optional): e.g. `--threads 23`
4. Apply the flight sheet. HiveOS shows the rig's hashrate (TH/s), accepted
   shares and uptime in its dashboard.

## How the stats reach HiveOS
`ariaminer` exposes a tiny JSON endpoint via `--stats-port` (default `4068`,
set automatically by `h-config.sh`). `h-stats.sh` polls
`http://127.0.0.1:4068/` and reports the hashrate (same TH/s convention as the
pool), accepted/rejected shares and uptime to HiveOS.

## Notes
- The hashrate is difficulty-weighted (like the pool); it converges upward for a
  few minutes after start while the pool's vardiff finds your rig's level.
- Auto-reconnect is built in: if the pool restarts, the miner reconnects on its
  own — no rig babysitting needed.
