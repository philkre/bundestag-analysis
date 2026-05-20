#!/usr/bin/env bash
# Render coalition-only MP networks for each Bundestag period.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RENDERER="$SCRIPT_DIR/renderer/render.js"

render_period() {
  local period="$1"
  local parties="$2"
  local title="$3"
  local out_dir="$SCRIPT_DIR/output/$period"

  if [ ! -f "$out_dir/nodes.csv" ]; then
    echo "Skipping $period — no nodes.csv found"
    return
  fi

  local parliament
  parliament=$(echo "$period" | sed 's/bundestag_/Bundestag /; s/_/ – /')

  echo "Rendering $period coalition: $parties"
  node "$RENDERER" \
    --out-dir "$out_dir" \
    --parliament "$parliament" \
    --title "$title" \
    --filter-parties "$parties" \
    --img-suffix "coalition" \
    --min-weight 0.15 \
    --top-edges-per-mp 2
}

render_period "bundestag_2005_2009" "CDU/CSU,SPD"                           "Grand Coalition 2005–09"
render_period "bundestag_2009_2013" "CDU/CSU,FDP"                           "CDU/CSU–FDP Coalition 2009–13"
render_period "bundestag_2013_2017" "CDU/CSU,SPD"                           "Grand Coalition 2013–17"
render_period "bundestag_2017_2021" "CDU/CSU,SPD"                           "Grand Coalition 2017–21"
render_period "bundestag_2021_2025" "SPD,BÜNDNIS 90/DIE GRÜNEN,FDP"        "Traffic Light Coalition 2021–25"
render_period "bundestag_2025_2029" "CDU/CSU,SPD"                           "Grand Coalition 2025–29"

echo ""
echo "Done. Images written to output/img/"
