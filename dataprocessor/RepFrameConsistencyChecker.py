"""
Cross-check the representative-frame information across the four artifacts of
CycPeptMPDB-4D, for both solvent environments (Water, Hexane).

For every peptide the representative conformation must be described
consistently by:

  1. log        — the GROMACS clustering log (`*_<env>.log`) defines the
                  representative frame as the "middle structure" time of
                  cluster 1.
  2. trajectory — that time must be one of the 100 frames in `*_Traj.pdb`.
  3. RepFrame   — the `<env>_RepFrame` column in the CSV must equal the
                  1-based index of that frame (frame 1 = 20.3 ns).
  4. structure  — the extracted `*_Str.pdb` must actually BE that frame; its
                  TITLE time must match the log's middle time.

The log is the single source of truth: RepFrame and the representative
structure are both derived from it. This script flags any peptide where the
trajectory, RepFrame, or structure disagree with the log.

Usage:
    python dataprocessor/RepFrameConsistencyChecker.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths (anchored to this file, never relative "../" strings) ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"

sys.path.insert(0, str(SCRIPT_DIR))
from ExtractMiddleConf import get_cluster_middle_time

# ── Trajectory frame grid: 100 frames, 20.3 → 50.0 ns, 0.3 ns apart ───────────
FIRST_FRAME_NS = 20.3
FRAME_STEP_NS = 0.3
ENVIRONMENTS = [
    ("Water", "H2O", "Water_RepFrame"),
    ("Hexane", "Hexane", "Hexane_RepFrame"),
]


def check_dataset(csv_path: Path, data_dir: Path) -> bool:
    """Cross-check every peptide in both environments. Return True if all consistent."""
    df = pd.read_csv(csv_path, low_memory=False).set_index("CycPeptMPDB_ID")
    all_consistent = True
    for env, suffix, repframe_col in ENVIRONMENTS:
        problems = check_environment(df, data_dir, env, suffix, repframe_col)
        report(env, len(df), problems)
        all_consistent &= not any(problems.values())
    return all_consistent


def check_environment(df: pd.DataFrame, data_dir: Path, env: str,
                      suffix: str, repframe_col: str) -> dict:
    """Return a dict of {problem_kind: [offending entries]} for one environment."""
    log_dir = data_dir / env / "Logs"
    traj_dir = data_dir / env / "Trajectories"
    str_dir = data_dir / env / "Structures"

    problems = {
        "missing_files": [],         # log / trajectory / structure absent
        "no_cluster_middle": [],     # log has no cluster-1 middle time
        "rep_time_not_in_traj": [],  # log middle time is not a trajectory frame
        "repframe_mismatch": [],     # RepFrame column != log frame  (id, log, csv)
        "stale_structure": [],       # *_Str.pdb is a different frame than the log
    }

    for cyc_id, row in df.iterrows():
        tag = f"{row.Source}_{cyc_id}_{suffix}"
        log_path = log_dir / f"{tag}.log"
        traj_path = traj_dir / f"{tag}_Traj.pdb"
        str_path = str_dir / f"{tag}_Str.pdb"

        if not (log_path.exists() and traj_path.exists() and str_path.exists()):
            problems["missing_files"].append(cyc_id)
            continue

        middle_ns = get_cluster_middle_time(str(log_path))
        if middle_ns is None:
            problems["no_cluster_middle"].append(cyc_id)
            continue
        log_frame = time_to_frame(middle_ns)

        if round(middle_ns, 1) not in trajectory_frame_times(traj_path):
            problems["rep_time_not_in_traj"].append(cyc_id)
        if int(row[repframe_col]) != log_frame:
            problems["repframe_mismatch"].append((cyc_id, log_frame, int(row[repframe_col])))
        if time_to_frame(structure_time_ns(str_path)) != log_frame:
            problems["stale_structure"].append(cyc_id)

    return problems


def report(env: str, n_total: int, problems: dict) -> None:
    """Print a per-environment summary; list the (small) offending sets in full."""
    print(f"\n=== {env}: {n_total} peptides ===")
    print(f"  missing log/traj/structure : {len(problems['missing_files'])}")
    print(f"  log has no cluster-1 middle : {len(problems['no_cluster_middle'])}")
    print(f"  rep-frame time not in traj  : {len(problems['rep_time_not_in_traj'])} "
          f"{problems['rep_time_not_in_traj']}")
    print(f"  RepFrame != log frame       : {len(problems['repframe_mismatch'])} "
          f"{problems['repframe_mismatch']}")
    print(f"  stale structure             : {len(problems['stale_structure'])} "
          f"{problems['stale_structure']}")


# ── Low-level helpers ─────────────────────────────────────────────────────────
def time_to_frame(t_ns: float) -> int:
    """1-based trajectory frame index for a time in ns (frame 1 = 20.3 ns)."""
    return round((t_ns - FIRST_FRAME_NS) / FRAME_STEP_NS) + 1


def trajectory_frame_times(traj_path: Path) -> set:
    """Set of frame times (ns, rounded to 0.1) from a multi-model trajectory PDB."""
    text = traj_path.read_text()
    return {round(float(t) / 1000.0, 1) for t in re.findall(r"t=\s*([\d.]+)", text)}


def structure_time_ns(str_path: Path) -> float:
    """Time (ns) of the single frame in a *_Str.pdb, read from its TITLE record."""
    for line in str_path.read_text().splitlines():
        if line.startswith("TITLE"):
            match = re.search(r"t=\s*([\d.]+)", line)
            assert match is not None, f"No t= in TITLE of {str_path}"
            return float(match.group(1)) / 1000.0
    raise ValueError(f"No TITLE record found in {str_path}")


if __name__ == "__main__":
    CSV_PATH = DATA_DIR / "CycPeptMPDB-4D.csv"

    consistent = check_dataset(CSV_PATH, DATA_DIR)
    print("\nALL CONSISTENT" if consistent else "\nINCONSISTENCIES FOUND (see above)")
    sys.exit(0 if consistent else 1)
