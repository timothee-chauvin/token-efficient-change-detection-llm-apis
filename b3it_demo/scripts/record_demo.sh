#!/bin/sh
# Re-record the README demo: asciinema cast -> animated SVG (assets/demo.svg).
# Run from b3it_demo/. Requires uv and npx.
set -e
cast=$(mktemp --suffix=.cast)
script -qc "stty rows 44 cols 100; \
  B3IT_DEMO_AUTO_PAUSE_SECONDS=3 \
  uvx asciinema rec --overwrite -c 'uv run b3it-demo < /dev/null' $cast" /dev/null
npx --yes svg-term-cli --in "$cast" --out assets/demo.svg --window
rm "$cast"
