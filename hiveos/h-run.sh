#!/usr/bin/env bash
cd `dirname $0`
. h-manifest.conf

ARGS=""
[[ -f $CUSTOM_CONFIG_FILENAME ]] && ARGS=$(< $CUSTOM_CONFIG_FILENAME)

# Run in foreground; HiveOS captures stdout to the miner log.
exec ./ariaminer $ARGS
