#!/usr/bin/env bash
# Re-run the comparison from an existing NLO export and extracted S-event LHE.
# See docs/audits/delta_noemission/report.txt for event generation/cards.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 PYTHIA_INSTALL TT_EXPORT S_EVENTS_LHE OUTPUT_DIR" >&2
  exit 2
fi
pythia_install=$(realpath "$1")
process_export=$(realpath "$2")
s_events=$(realpath "$3")
study_output=$(realpath -m "$4")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$study_output"
xml_dir="$pythia_install/share/Pythia8/xmldoc"
c++ -O2 -std=c++17 "$script_dir/pythia_probe.cc" -I"$pythia_install/include" -L"$pythia_install/lib" -Wl,-rpath,"$pythia_install/lib" -lpythia8 -o "$study_output/probe"
c++ -O2 -std=c++17 -shared -fPIC "$script_dir/pythia_pdf.cc" -I"$pythia_install/include" -L"$pythia_install/lib" -Wl,-rpath,"$pythia_install/lib" -lpythia8 -o "$study_output/pdf.so"
"$study_output/probe" "$xml_dir" shower "$s_events" "$study_output/branch_exact" 300 72883 13 > "$study_output/main.log" 2>&1
"$study_output/probe" "$xml_dir" shower "$s_events" "$study_output/branch_exact_pdf15" 200 72884 15 > "$study_output/pdf15.log" 2>&1
"$study_output/probe" "$xml_dir" shower "$s_events" "$study_output/branch_exact_isr" 200 72885 13 isr > "$study_output/isr.log" 2>&1
"$study_output/probe" "$xml_dir" shower "$s_events" "$study_output/branch_exact_fsr" 200 72886 13 fsr > "$study_output/fsr.log" 2>&1
for prefix in branch_exact branch_exact_pdf15 branch_exact_isr branch_exact_fsr; do
  python3 "$script_dir/pythia_audit.py" "$study_output/$prefix" > "$study_output/$prefix.audit.log"
done
python3 "$script_dir/analyse.py" "$study_output/branch_exact" --pdf-library "$study_output/pdf.so" --xmldir "$xml_dir" --native-pdf-export "$process_export" --probes 60 --output "$study_output/results"
python3 "$script_dir/analyse.py" "$study_output/branch_exact_pdf15" --pdf-library "$study_output/pdf.so" --xmldir "$xml_dir" --native-pdf-export "$process_export" --probes 40 --output "$study_output/results/matched_pdf15"
python3 "$script_dir/controls.py" "$study_output/branch_exact" "$study_output/results"
