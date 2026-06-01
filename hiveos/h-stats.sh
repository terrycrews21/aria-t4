#!/usr/bin/env bash
# Report ariaminer stats to HiveOS by reading its JSON --stats-port endpoint.
. `dirname $0`/h-manifest.conf

local resp=$(curl -s --connect-timeout 2 --max-time 3 "http://127.0.0.1:${CUSTOM_API_PORT}/" 2>/dev/null)
[[ -z $resp ]] && return

local hs=$(echo "$resp" | jq -r '.hashrate_hs // 0')
local acc=$(echo "$resp" | jq -r '.accepted // 0')
local rej=$(echo "$resp" | jq -r '.rejected // 0')
local up=$(echo "$resp"  | jq -r '.uptime_s // 0')

# HiveOS wants total hashrate in kH/s (khs) and a stats JSON. hs_units="hs"
# tells HiveOS the per-device hs[] array is already in H/s.
khs=$(awk "BEGIN{printf \"%.3f\", ${hs}/1000}")
stats=$(jq -nc \
  --argjson hs "[${hs}]" \
  --argjson ar "[${acc},${rej}]" \
  --argjson up "${up}" \
  --arg ver "$CUSTOM_VERSION" \
  '{hs:$hs, hs_units:"hs", ar:$ar, uptime:$up, ver:$ver, algo:"gemm-int7", bus_numbers:[0]}')
