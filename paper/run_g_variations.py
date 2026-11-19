#!/usr/bin/env python3
"""Rerun the four G-width variations and two common-G comparisons, sequentially.

Fresh exports use the fixed matching source and preserve the previous samples.
Use --prepare-only to create the exports without starting integrations.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 3
CARD_NAMES = ("param_card.dat", "run_card.dat", "FKS_params.dat", "shower_card.dat",
              "FO_analyse_card.dat", "amcatnlo_configuration.txt")
MATCHING_SOURCES = (
    "Template/NLO/SubProcesses/cluster.f",
    "Template/NLO/SubProcesses/driver_mintMC.f",
    "Template/NLO/SubProcesses/fks_singular.f",
    "Template/NLO/SubProcesses/montecarlocounter.f",
    "Template/NLO/SubProcesses/genps_fks.f",
    "Template/NLO/SubProcesses/genps_fks_helpers.f",
    "Template/NLO/SubProcesses/genps_fks_radiation.f",
    "Template/NLO/SubProcesses/fks_phase_space_data.f",
    "Template/NLO/SubProcesses/initial_recoil.f",
    "Template/NLO/SubProcesses/resonance_recoil.f",
    "Template/NLO/SubProcesses/boostwdir2.f",
    "Template/NLO/SubProcesses/fks_Sij.f",
    "Template/NLO/SubProcesses/FKSParams.f90",
    "Template/NLO/SubProcesses/scale_module.f90",
    "Template/NLO/SubProcesses/mc_native_runtime.f",
    "Template/NLO/SubProcesses/fks_powers.inc",
    "Template/NLO/MCatNLO/PY8Analyzer/py8an_HwU_pp_tj_recoil.f",
    "Template/NLO/MCatNLO/include/HEPMC_ST_RECOIL.INC",
    "Template/NLO/MCatNLO/include/HEPEVTSize.h",
    "Template/NLO/MCatNLO/include/LHEFRead.h",
    "Template/NLO/MCatNLO/Scripts/MCatNLO_MadFKS_PYTHIA8.Script",
    "Template/NLO/MCatNLO/shower_template.sh",
)
OUTPUT_NAMES = ("summary.txt", "events.lhe.gz", "plot_PYTHIA8_1_0.HwU",
                "plot_PYTHIA8_1_0.pdf")
SHOWER_LOG = "MCatNLO/RUN_PYTHIA8_1/mcatnlo_run.log"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_fixed_clustering(path):
    """Reject exports missing the coloured core mother momentum fix."""
    if not re.search(r"^\s*p_inter\(:,\s*0,\s*0\)\s*=\s*pcl\(:,\s*3\)",
                     path.read_text(), re.MULTILINE | re.IGNORECASE):
        raise RuntimeError(f"Missing normalization fix in {path}")


def generation_inputs(row):
    """Record the physics inputs that must agree before reusing a run."""
    check_fixed_clustering(ROOT / MATCHING_SOURCES[0])
    source = Path(row["source_process"])
    names = list(MATCHING_SOURCES) + [
        str(source / "Cards" / name) for name in CARD_NAMES + ("proc_card_mg5.dat",)]
    return {"schema": SCHEMA, "normalization_fix": "coloured-core-momentum-2026-10-25",
            "numerical_fix": "local-recoil-geometry-and-endpoints-2026-10-25",
            "run_spec": row, "source_sha256": {name: sha256(ROOT / name) for name in names}}


def prepared_files(directory):
    # Core count is an execution setting, so it may change after preparation.
    names = ["Cards/" + name for name in CARD_NAMES
             if name != "amcatnlo_configuration.txt"]
    names += [str(Path(name).relative_to("Template/NLO")) for name in MATCHING_SOURCES]
    hashes = {name: sha256(directory / name) for name in names}
    # Keep the shower/library selection protected while allowing a prepared
    # export to run with a different number of workers.
    config = (directory / "Cards/amcatnlo_configuration.txt").read_text()
    config = re.sub(r"^[ \t]*(?:run_mode|nb_core|automatic_html_opening)[ \t]*=.*(?:\n|$)",
                    "", config, flags=re.MULTILINE)
    hashes["Cards/amcatnlo_configuration.txt (without execution settings)"] = hashlib.sha256(
        config.strip().encode()).hexdigest()
    return hashes


def shower_jobs(nevents, cores):
    """MG5 requires fewer than 100 equal chunks and an exact divisor of N."""
    nevents, cores = int(nevents), int(cores)
    if nevents < 1 or cores < 1:
        raise ValueError("Event and core counts must be positive")
    return next(jobs for jobs in range(min(nevents, cores, 99), 0, -1)
                if nevents % jobs == 0)


def shower_logs(directory):
    """Return every expected worker log, including the serial naming case."""
    card = (directory / "Cards/shower_card.dat").read_text()
    match = re.search(r"^[ \t]*nsplit_jobs[ \t]*=[ \t]*(\d+)\b", card, re.MULTILINE)
    if not match or not 1 <= int(match[1]) < 100:
        raise RuntimeError(f"Invalid shower job count in {directory / 'Cards/shower_card.dat'}")
    jobs = int(match[1])
    serial = directory / SHOWER_LOG
    return ([serial] if jobs == 1 else
            [serial.with_name(f"mcatnlo_run_{i}.log") for i in range(1, jobs + 1)])


def configure_cores(directory, cores):
    run = (directory / "Cards/run_card.dat").read_text()
    match = re.search(r"^[ \t]*(\d+)[ \t]*=[ \t]*nevents\b", run, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Missing event count in {directory / 'Cards/run_card.dat'}")
    card = directory / "Cards/shower_card.dat"
    for name, value in (("nevents", -1), ("nsplit_jobs", shower_jobs(match[1], cores)),
                        ("combine_td", "T")):
        replace(card, rf"^({name}[ \t]*=[ \t]*)[^!\n]*(.*)$", value)
    with (directory / "Cards/amcatnlo_configuration.txt").open("a") as stream:
        stream.write(f"\nrun_mode = 2\nnb_core = {cores}\nautomatic_html_opening = False\n")


def preflight_pythia(row):
    config = ROOT / row["source_process"] / "Cards/amcatnlo_configuration.txt"
    settings = re.findall(r"^[ \t]*pythia8_path[ \t]*=[ \t]*([^#\n]+)",
                          config.read_text(), re.MULTILINE)
    if not settings:
        raise RuntimeError(f"Missing pythia8_path in {config}")
    prefix = Path(settings[-1].strip())
    if not prefix.is_absolute():
        prefix = ROOT / prefix
    tool = prefix / "bin/pythia8-config"
    libraries = [prefix / "lib" / ("libpythia8" + suffix)
                 for suffix in (".a", ".so", ".dylib")]
    if not tool.is_file() or not any(path.is_file() for path in libraries):
        raise RuntimeError(f"Missing Pythia installation: {prefix}. Restore the configured build first.")
    try:
        result = subprocess.run([str(tool), "--xmldoc"], check=True, capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot query the configured Pythia installation: {prefix}") from exc
    xml = Path(result.stdout.strip())
    if xml.resolve() != (prefix / "share/Pythia8/xmldoc").resolve():
        raise RuntimeError(f"Pythia XML path {xml} does not belong to configured prefix {prefix}")
    settings_file = xml / "SpacelikeShowers.xml"
    if not settings_file.is_file():
        raise RuntimeError(f"Missing Pythia shower settings: {settings_file}")
    text = settings_file.read_text()
    for key, setting in (("PythiaLocalCollinearPatch", "IFquarkCollinearCorrection"),
                         ("PythiaLocalSoftPatch", "IFtopSoftCorrection")):
        if row[key] == "True" and f'name="SpaceShower:{setting}"' not in text:
            raise RuntimeError(f"Configured Pythia build {prefix} lacks {setting}")
    return prefix


def source_migration_provenance(row, directory, inputs):
    """Document the one audited local-only correction for the global sample."""
    path = directory / "campaign_source_migration.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    source = "Template/NLO/SubProcesses/resonance_recoil.f"
    before = "cfb85cf75959fee528c89c13a349bbd1ad97e47bbda7e188b0800951f4df05b8"
    after = "ab6cdf35cbecbb8c876675ece29a933eee7e4c1a7865226ca0456930934dc566"
    expected = dict(inputs, source_sha256=dict(inputs.get("source_sha256", {})))
    expected["source_sha256"][source] = before
    original = record.get("original_generation_inputs")
    fks = (directory / "Cards/FKS_params.dat").read_text()
    settings = [re.search(rf"^#{name}[ \t]*\n[ \t]*(\S+)", fks, re.MULTILINE)
                for name in ("FKSPythiaLocalRecoil", "FKSFinalRecoil")]
    exported = directory / "SubProcesses/resonance_recoil.f"
    if (record.get("schema") != SCHEMA or record.get("run_id") != "C_GLB"
            or row.get("run_id") != "C_GLB" or row.get("FKSPythiaLocalRecoil") != "0"
            or inputs.get("run_spec") != row
            or inputs.get("source_sha256", {}).get(source) != after
            or original != expected or original == inputs
            or any(setting is None or setting[1] != "0" for setting in settings)
            or sha256(exported) != before):
        raise RuntimeError(f"Unverified compatible source migration: {path}")
    return {"generation_inputs": original, "source_migration_sha256": sha256(path)}


def completion_record(row, directory, inputs):
    events = directory / "Events" / row["run_id"]
    outputs = {name: sha256(events / name) for name in OUTPUT_NAMES}
    outputs.update({str(path.relative_to(directory)): sha256(path)
                    for path in shower_logs(directory)})
    return {"schema": SCHEMA, "inputs": inputs,
            "cards_sha256": prepared_files(directory),
            "output_sha256": outputs,
            **source_migration_provenance(row, directory, inputs)}


def pending_shower(row, directory, inputs):
    """Resume only event files whose generation has already been verified."""
    events = directory / "Events" / row["run_id"]
    path = events / "campaign_shower_pending.json"
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    outputs = record.get("output_sha256", {})
    if (record.get("schema") != SCHEMA or record.get("inputs") != inputs
            or set(outputs) not in ({"events.lhe", "summary.txt"},
                                    {"events.lhe.gz", "summary.txt"})):
        raise RuntimeError(f"Unverified pending shower: {path}")
    for name, digest in outputs.items():
        output = events / name
        if not output.is_file() or sha256(output) != digest:
            raise RuntimeError(f"Changed pending shower input: {output}")
    return True


def pending_generation(row, directory, inputs):
    """Use retained integration grids only after an explicit audited checkpoint."""
    path = directory / "campaign_generation_pending.json"
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    retained = record.get("retained_sha256", {})
    status = "SubProcesses/job_status.pkl"
    if (record.get("schema") != SCHEMA or record.get("inputs") != inputs
            or not isinstance(retained, dict) or status not in retained):
        raise RuntimeError(f"Unverified pending generation: {path}")
    for name, digest in retained.items():
        relative = Path(name)
        output = directory / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not output.is_file() or sha256(output) != digest):
            raise RuntimeError(f"Changed retained generation input: {output}")
    # MG5 itself loads this saved job list for --only_generation. Inspect it
    # only after its recorded digest has been checked.
    with (directory / status).open("rb") as stream:
        jobs = pickle.load(stream)
    if not isinstance(jobs, list) or len(jobs) != 4:
        raise RuntimeError(f"Wrong retained generation jobs: {path}")
    required = {status}
    channels = set()
    count = 0
    for job in jobs:
        process = job.get("p_dir", "") if isinstance(job, dict) else ""
        dirname = Path("SubProcesses") / process / "GF1.0"
        if (not re.fullmatch(r"P[0-9A-Za-z_]+", process)
                or job.get("split") != 0 or job.get("mint_mode") != 2
                or job.get("channel") != "1.0" or job.get("run_mode") != "all"
                or Path(job.get("dirname", "")).resolve() != (directory / dirname).resolve()
                or not isinstance(job.get("nevents"), int) or job["nevents"] < 0):
            raise RuntimeError(f"Wrong retained generation job: {path}")
        channels.add(process)
        count += job["nevents"]
        required.update(str(dirname / name) for name in
                        ("mint_grids", "grid.MC_integer", "res_1.dat", "res_1"))
    if len(channels) != 4 or count != int(row["nevents"]) or not required <= retained.keys():
        raise RuntimeError(f"Wrong retained generation count or grids: {path}")
    return True


def archive_generation(row, directory):
    """Preserve every partial event chunk before MG5 recreates generation jobs."""
    chunks = sorted((directory / "SubProcesses").glob("P*/GF*_*"))
    events = directory / "Events" / row["run_id"]
    paths = chunks + ([events] if events.exists() else [])
    attempt = 1
    while any(path.with_name(f"INTERRUPTED_{path.name}_attempt_{attempt}").exists()
              for path in paths):
        attempt += 1
    archived = []
    for path in paths:
        target = path.with_name(f"INTERRUPTED_{path.name}_attempt_{attempt}")
        shutil.move(str(path), str(target))
        archived.append(target)
    return archived


def archive_showers(row, directory):
    """Preserve interrupted outputs before restarting the recorded shower."""
    mc = directory / "MCatNLO"
    events = directory / "Events" / row["run_id"]
    runs = sorted(mc.glob("RUN_PYTHIA8_*"))
    plots = sorted(events.glob("plot_PYTHIA8_*"))
    attempt = 1
    while ((events / "interrupted_showers" / f"attempt_{attempt}").exists()
           or any(mc.glob(f"INTERRUPTED_RUN_PYTHIA8_*_attempt_{attempt}"))):
        attempt += 1
    archived = []
    for path in runs:
        # Keep relative event-file and XML symlinks at their original depth.
        target = path.with_name(f"INTERRUPTED_{path.name}_attempt_{attempt}")
        shutil.move(str(path), str(target))
        archived.append(target)
    for path in plots:
        target = events / "interrupted_showers" / f"attempt_{attempt}" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        archived.append(target)
    return archived


def replace(path, pattern, value):
    text, count = re.subn(pattern, lambda m: m[1] + str(value) + m[2],
                          path.read_text(), flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one matching setting in {path}: {pattern}")
    path.write_text(text)


def set_run_card(path, name, value):
    replace(path, rf"^([ \t]*)[^!\n=]+?([ \t]*=[ \t]*{name}\b.*)$", value)


def set_fks_card(path, name, value):
    # The reference cards predate the common-profile validation control.
    if not re.search(rf"^#{name}[ \t]*$", path.read_text(), re.MULTILINE):
        with path.open("a") as stream:
            stream.write(f"\n#{name}\n{value}\n")
    else:
        replace(path, rf"^(#{name}[ \t]*\n)[^\n]*(\n)", value)


def prepare(row, directory, cores, log):
    preflight_pythia(row)
    source = ROOT / row["source_process"]
    commands = (source / "Cards/proc_card_mg5.dat").read_text()
    commands, count = re.subn(r"^output\b.*$", lambda m: f"output {directory}",
                              commands, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("Expected one output command in the process card")
    command_file = directory.with_suffix(".mg5")
    command_file.write_text(commands + "\nquit\n")
    subprocess.run([sys.executable, str(ROOT / "bin/mg5_aMC"), str(command_file)],
                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in CARD_NAMES:
        shutil.copy2(source / "Cards" / name, directory / "Cards" / name)

    card = directory / "Cards/run_card.dat"
    for name in ("nevents", "req_acc", "iseed", "shower_scale_factor",
                 "reweight_scale", "store_rwgt_info"):
        set_run_card(card, name, row[name])
    for name, value in (("reweight_PDF", "False"), ("parton_shower", "PYTHIA8"),
                        ("ickkw", "0"), ("MCatNLO_DELTA", "False")):
        set_run_card(card, name, value)
    for name in ("rw_rscale", "rw_fscale"):
        if row[name]:
            set_run_card(card, name, row[name].replace(";", ", "))
    card = directory / "Cards/FKS_params.dat"
    for name, value in (("MCExplicitKLSum", ".true."),
                        ("MCSubtractionAtFixedFlow", ".true."),
                        ("FKSExplicitSum", ".true." if row["FKSPythiaLocalRecoil"] == "1" else ".false."),
                        ("FKSISRMapping", "0"), ("FKSFinalRecoil", "0")):
        set_fks_card(card, name, value)
    for name in ("FKSPythiaLocalRecoil", "PythiaLocalCollinearPatch", "PythiaLocalSoftPatch"):
        val = row[name]
        set_fks_card(card, name, f".{val.lower()}." if val in ("True", "False") else val)
    if row["PythiaLocalGWidth"]:
        set_fks_card(card, "PythiaLocalGWidth", row["PythiaLocalGWidth"] + "d0")
    set_fks_card(card, "PythiaLocalGProfile", 1 if row["profile"] == "w_forced" else 0)
    for name in ("xicut", "deltaI", "deltaO", "xiScut", "deltaS", "xiBSVcut"):
        replace(directory / "SubProcesses/fks_powers.inc",
                rf"^(\s*parameter\s*\(\s*{name}\s*=)[^)]*(\).*)$", row[name] + "d0")
    card = directory / "Cards/shower_card.dat"
    for name, value in (("nevents", "-1"), ("nsplit_jobs", shower_jobs(row["nevents"], cores)),
                        ("combine_td", "T"), ("rnd_seed", row["rnd_seed"])):
        replace(card, rf"^({name}[ \t]*=[ \t]*)[^!\n]*(.*)$", value)
    # Use the master analysis and matching HEPEVT layout in each fresh export.
    mc = ROOT / "Template/NLO/MCatNLO"
    for name in ("PY8Analyzer/py8an_HwU_pp_tj_recoil.f", "include/HEPMC_ST_RECOIL.INC",
                 "include/HEPEVTSize.h", "shower_template.sh"):
        shutil.copy2(mc / name, directory / "MCatNLO" / name)
    configure_cores(directory, cores)
    (directory / "run_spec.json").write_text(json.dumps(row, indent=2) + "\n")
    check_fixed_clustering(directory / "SubProcesses/cluster.f")


def check_output(row, directory):
    """A generation summary alone does not mean the shower succeeded."""
    marker = directory / "NUMERICAL_FIX.json"
    if marker.exists() and row["run_id"] in json.loads(marker.read_text()).get("invalid_runs", []):
        raise RuntimeError(f"Superseded MC@NLO sample: {directory}. "
                           "Use a fresh export after the local-recoil numerical fixes.")
    marker = directory / "NORMALIZATION_FIX.json"
    if marker.exists() and row["run_id"] in json.loads(marker.read_text()).get("invalid_runs", []):
        raise RuntimeError(f"Superseded MC@NLO sample: {directory}. "
                           "Use a fresh --output-dir after the core clustering fix.")
    sys.path.insert(0, str(ROOT))
    from madgraph.various.histograms import HwUList
    events = directory / "Events" / row["run_id"]
    for name in OUTPUT_NAMES:
        p = events / name
        if not p.is_file() or not p.stat().st_size:
            raise RuntimeError(f"Missing completed generation/shower output: {p}")
    count = re.search(r"Number of events generated:\s*(\d+)",
                      (events / "summary.txt").read_text())
    if not count or int(count[1]) != int(row["nevents"]):
        raise RuntimeError(f"Wrong or missing generated event count in {events / 'summary.txt'}")
    logs = shower_logs(directory)
    total = int(row["nevents"])
    if total % len(logs):
        raise RuntimeError(f"Shower job count does not divide {total} events: {directory}")
    expected = total // len(logs)
    accepted_total = 0
    for shower_log in logs:
        if not shower_log.is_file():
            raise RuntimeError(f"Missing completed shower log: {shower_log}")
        accepted = re.findall(r"^\s*\|\s*sum\s*\|\s*(\d+)\s+(\d+)\s+(\d+)\s*\|",
                              shower_log.read_text(), re.MULTILINE)
        if not accepted or int(accepted[-1][2]) != expected:
            raise RuntimeError(f"Shower did not accept all {expected} worker events: {shower_log}")
        accepted_total += int(accepted[-1][2])
    if accepted_total != total:
        raise RuntimeError(f"Shower did not accept all {total} events: {directory}")
    data = [hist for hist in HwUList(str(events / "plot_PYTHIA8_1_0.HwU"))
            if not hist.title.upper().startswith("AUX")
            and not hist.type.upper().startswith("AUX")]
    if len(data) != 75 or len({(hist.title, hist.type) for hist in data}) != 75:
        raise RuntimeError(f"Expected 75 analysis histograms in {events}")
    if any(not math.isfinite(value) for hist in data for bin_ in hist.bins
           for value in bin_.wgts.values()):
        raise RuntimeError(f"Non-finite histogram weights in {events}")
    if row["reweight_scale"] == "True":
        expected = set(itertools.product(map(float, row["rw_rscale"].split(";")),
                                         map(float, row["rw_fscale"].split(";"))))
        # The nominal (1,1) entry may be represented by 'central' alone.
        for hist in data:
            labels = hist.bins[0].wgts
            scales = {tuple(k[-2:]) for k in labels if isinstance(k, tuple)
                      and k[0] in ("scale", "scale_adv")}
            if "central" not in labels or not (expected - {(1., 1.)}) <= scales:
                raise RuntimeError(f"Missing hard-scale weights in {events}: {hist.title}")
    return events


def run_campaign(counts, output_name, description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cores", type=int, default=64,
                        help="cores for integration and parallel showers (default: 64)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / output_name)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="export and configure processes without generating events")
    parser.add_argument("--only", nargs="+", metavar="RUN_ID",
                        help="run only the named entries from this campaign")
    args = parser.parse_args(argv)
    if args.cores < 1:
        parser.error("--cores must be positive")
    output = args.output_dir.resolve()
    if any(c.isspace() for c in str(output)):
        parser.error("use an output path without whitespace for MG5 commands")
    with (ROOT / "paper/proposed_runs.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["group"] in counts]
    for group, count in counts.items():
        if sum(row["group"] == group for row in rows) != count:
            parser.error(f"expected {count} {group} rows in proposed_runs.csv")
    if args.only:
        unknown = set(args.only) - {row["run_id"] for row in rows}
        if unknown:
            parser.error("unknown run IDs: " + ", ".join(sorted(unknown)))
        rows = [row for row in rows if row["run_id"] in args.only]
    for row in rows:
        name = row["run_id"]
        directory = output / name
        done = directory / "Events" / name / "campaign_done.json"
        print(f"{name}: {row['source_process']}, G={row['profile']}, "
              f"h={row['PythiaLocalGWidth'] or 'inactive'}, Q={row['shower_scale_factor']}, "
              f"events={row['nevents']}, seeds={row['iseed']}/{row['rnd_seed']}, "
              f"scale weights={row['reweight_scale']}", flush=True)
        print(f"  Output: {directory}", flush=True)
        print(f"  Shower: {shower_jobs(row['nevents'], args.cores)} jobs, combined histograms", flush=True)
        if args.dry_run:
            continue
        inputs = generation_inputs(row)
        resume_shower = False
        resume_generation = False
        manifest_path = directory / "campaign_manifest.json"
        if done.is_file():
            stored = json.loads(done.read_text())
            if not manifest_path.is_file() or stored.get("inputs") != inputs:
                sys.exit(f"Unverified or changed inputs for {name}: use a fresh --output-dir.")
            if json.loads(manifest_path.read_text()).get("inputs") != inputs:
                sys.exit(f"Prepared inputs changed for {name}: use a fresh --output-dir.")
            check_output(row, directory)
            if stored != completion_record(row, directory, inputs):
                sys.exit(f"Completed files changed for {name}: inspect {done.parent}.")
            print(f"  Already finished: {done.parent}", flush=True)
            continue
        if directory.exists():
            if not manifest_path.is_file():
                sys.exit(f"Unverified export exists: {directory}. Use a fresh --output-dir.")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("inputs") != inputs or manifest.get("prepared_sha256") != prepared_files(directory):
                sys.exit(f"Prepared inputs changed: {directory}. Use a fresh --output-dir.")
            if (directory / "Events" / name).exists():
                resume_shower = pending_shower(row, directory, inputs)
            if not resume_shower:
                resume_generation = pending_generation(row, directory, inputs)
            if ((directory / "Events" / name).exists()
                    and not resume_shower and not resume_generation):
                sys.exit(f"Incomplete run exists: {directory / 'Events' / name}. Inspect it before retrying.")
            check_fixed_clustering(directory / "SubProcesses/cluster.f")
            preflight_pythia(row)
            configure_cores(directory, args.cores)
            manifest_path.write_text(json.dumps({"inputs": inputs,
                "prepared_sha256": prepared_files(directory)}, indent=2) + "\n")
            print(f"  Using verified prepared export: {directory}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"{name}.log"
        if not directory.exists():
            print(f"  Preparing; log: {log_path}", flush=True)
            with log_path.open("w") as log:
                prepare(row, directory, args.cores, log)
            manifest_path.write_text(json.dumps({"inputs": inputs,
                "prepared_sha256": prepared_files(directory)}, indent=2) + "\n")
        if args.prepare_only:
            print(f"  Prepared: {directory}", flush=True)
            continue
        print(f"  Running; log: {log_path}", flush=True)
        if resume_shower:
            for path in archive_showers(row, directory):
                print(f"  Preserved interrupted shower: {path}", flush=True)
            command = [str(directory / "bin/shower"), name, "-f"]
        else:
            if resume_generation:
                for path in archive_generation(row, directory):
                    print(f"  Preserved interrupted generation: {path}", flush=True)
            command = [str(directory / "bin/generate_events"), "aMC@NLO", "-f",
                       f"--name={name}"]
            if resume_generation:
                command.append("--only_generation")
        with log_path.open("a") as log:
            subprocess.run(command, cwd=directory,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        check_output(row, directory)
        # MG5 may rewrite cards into its canonical format while launching.
        # The completion records describe the files actually left by that run.
        manifest_path.write_text(json.dumps({"inputs": inputs,
            "prepared_sha256": prepared_files(directory)}, indent=2) + "\n")
        done.write_text(json.dumps(completion_record(row, directory, inputs), indent=2) + "\n")
        print(f"  Finished: {done.parent}", flush=True)


if __name__ == "__main__":
    run_campaign({"G": 4, "P": 2}, "g_runs_fixed", __doc__)
