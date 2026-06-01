#!/usr/bin/env bash
# Build the ariaminer command line from the HiveOS flight sheet.
# Flight sheet mapping:
#   Wallet/template  -> CUSTOM_TEMPLATE  (your prl1... address)
#   Pool URL         -> CUSTOM_URL       (host:port, e.g. pearl.ariabrain.com:3334)
#   Pass             -> CUSTOM_PASS      (optional, default x)
#   Extra config     -> CUSTOM_USER_CONFIG (e.g. --threads 23)

[[ -z $CUSTOM_TEMPLATE ]] && echo "ERROR: empty wallet (set it in the flight sheet template)" && return 1
[[ -z $CUSTOM_URL ]] && echo "ERROR: empty pool URL" && return 1

# Strip any stratum+tcp:// prefix HiveOS may prepend.
POOL=$(echo "$CUSTOM_URL" | sed -E 's#^[a-z+]+://##')
WALLET="$CUSTOM_TEMPLATE"
PASS="${CUSTOM_PASS:-x}"
WORKER="${WORKER_NAME:-hive}"
PORT="${CUSTOM_API_PORT:-4068}"

echo "--pool $POOL --wallet $WALLET --worker $WORKER --password $PASS --stats-port $PORT $CUSTOM_USER_CONFIG" > $CUSTOM_CONFIG_FILENAME
