#!/usr/bin/env python3
"""
Batch MD-vs-NMR NOE distance comparison across cyclic peptides.

For each peptide:
  - If a wwPDB NEF/NMR-STAR (.str) file is available, use it as source of truth
    for atom names and upper bounds.
  - Otherwise, parse the raw XPLOR .mr file directly.
  - In either case, cross-reference the .mr for the S/M/W/vW category (from the
    depositing group's comments).
  - Compute per-restraint MD <r^-6>^{-1/6} average, plus 5th & 95th percentiles,
    across the 100-frame trajectory.
  - Output a combined CSV with a `source` column indicating provenance.

Unlike noe_md_comparison.py, this parses NMR-STAR (.str) restraints and handles
AMBIGUOUS multi-atom restraints (r^-6-averaged over members). It does NOT parse
CYANA .upl, so the CYANA-only peptides (no .str) yield no rows here.

Paths are script-relative:
    restraints    <script_dir>/CHCl3/{PID}.mr and {pid}_nmr-data.str   (this folder)
    trajectories  <Data>/CycPeptMPDB_4D/June2026/CHCl3/{PID}_CHCl3_Traj.pdb
    output        <repo>/csvs/all_NOE_MD_comparison_str.csv
"""

import re, os, sys, csv, glob
from pathlib import Path
import numpy as np
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent               # dataprocessor/NMR_Analysis
REPO_ROOT = SCRIPT_DIR.parent.parent                       # repo root
DATA_DIR = REPO_ROOT.parent / "Data" / "CycPeptMPDB_4D"    # sibling Data dir
RESTRAINT_DIR = SCRIPT_DIR / "CHCl3"                        # {PID}.mr / {pid}_nmr-data.str
TRAJ_DIR = DATA_DIR / "June2026" / "CHCl3"                  # {PID}_CHCl3_Traj.pdb
OUT_CSV = REPO_ROOT / "csvs" / "all_NOE_MD_comparison_str.csv"

PEPTIDES = ['7L9D', '7L96', '7L98', '7UBG', '7UZL', '8CWA']

# ------------------ trajectory ------------------
def parse_traj(path):
    frames=[]; cur=None
    with open(path) as f:
        for l in f:
            if l.startswith('MODEL'):
                cur={}; continue
            if l.startswith('ENDMDL'):
                if cur is not None: frames.append(cur); cur=None
                continue
            if not l.startswith('ATOM'): continue
            a=l[12:16].strip(); r=int(l[22:26])
            try:
                x=float(l[30:38]); y=float(l[38:46]); z=float(l[46:54])
            except ValueError:
                continue
            cur[(r,a)]=np.array([x,y,z])
        if cur is not None: frames.append(cur)
    return frames

# ------------------ .str ------------------
def parse_str(path):
    """Return list of dicts: {'frame_type', 'id', 'upper', 'members', 'auth_key'}."""
    with open(path) as f: text=f.read()
    out=[]
    for name, body in re.findall(r'save_(\S+)\n(.*?)\nsave_', text, re.DOTALL):
        if 'Gen_dist_constraint' not in body: continue
        m=re.search(r'_Gen_dist_constraint_list\.Constraint_type\s+(\S+)', body)
        if not m: continue
        ct = m.group(1).strip("'")
        loop = re.search(r'loop_\s*\n((?:_Gen_dist_constraint\.\S+\s*\n)+)(.*?)stop_', body, re.DOTALL)
        if not loop: continue
        hdr, data = loop.groups()
        cols = re.findall(r'_Gen_dist_constraint\.(\S+)', hdr)
        rows_by_id = defaultdict(list)
        for ln in data.strip().split('\n'):
            if not ln.strip() or ln.lstrip().startswith('#'): continue
            parts = ln.split()
            if len(parts) != len(cols): continue
            d=dict(zip(cols,parts))
            rows_by_id[d['ID']].append(d)
        for cid, rows in rows_by_id.items():
            r0=rows[0]
            members=[]
            for r in rows:
                try:
                    members.append({
                        'canon': ((int(r['Comp_index_ID_1']), r['Atom_ID_1']),
                                  (int(r['Comp_index_ID_2']), r['Atom_ID_2'])),
                    })
                except (ValueError, KeyError):
                    continue
            if not members: continue
            out.append({
                'frame_type': ct,
                'id': cid,
                'upper': float(r0['Distance_upper_bound_val']),
                'members': members,
                'auth_key': (int(r0['Auth_seq_ID_1']), r0['Auth_atom_name_1'],
                             int(r0['Auth_seq_ID_2']), r0['Auth_atom_name_2']),
            })
    return out

# ------------------ .mr ------------------
NOE_RE = re.compile(
    r'assi\s*\(\s*resi\s+(\d+)\s+and\s+name\s+(\S+?)\s*\)\s*'
    r'\(\s*resi\s+(\d+)\s+and\s+name\s+(\S+?)\s*\)\s+'
    r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(?:!\s*(.*))?', re.IGNORECASE)

def _cat_from_intensity(n):
    if n is None: return None
    if n>=9: return 'S'
    if n>=6: return 'M'
    if n>=3: return 'W'
    return 'vW'

def parse_mr(path):
    """Return dict keyed by (r1,a1,r2,a2,upper) -> {category, intensity, comment}."""
    out={}; in_hb=False
    with open(path) as f:
        for l in f:
            if 'H-Bond' in l or 'H-BOND' in l: in_hb=True
            if in_hb: continue
            if not l.lstrip().lower().startswith('assi'): continue
            m=NOE_RE.match(l.strip())
            if not m: continue
            r1,a1,r2,a2,lo,tol,up,com = m.groups()
            com=(com or '').strip()
            ltr=re.search(r'\b(S|M|W|vw|VW|Vw|vW)\b', com)
            letter = ltr.group(1).upper() if ltr else None
            if letter=='VW': letter='vW'
            num=re.search(r'(?:\()?(\d{1,2})(?:\))?', com)
            intensity=int(num.group(1)) if num else None
            cat=letter if letter else _cat_from_intensity(intensity)
            out[(int(r1),a1,int(r2),a2,float(up))] = dict(category=cat, intensity=intensity, comment=com)
    return out

def lookup_mr_category(mr_map, auth_key, upper):
    r1,a1,r2,a2 = auth_key
    for k in [(r1,a1,r2,a2,upper), (r2,a2,r1,a1,upper)]:
        if k in mr_map: return mr_map[k]
    # Fallback: match without upper (rare)
    for k,v in mr_map.items():
        if (k[0],k[1],k[2],k[3])==(r1,a1,r2,a2) or (k[0],k[1],k[2],k[3])==(r2,a2,r1,a1):
            return v
    return dict(category=None, intensity=None, comment='')

# ------------------ .mr-only heuristic mapping ------------------
def expand_atom_spec(rnum, spec, sample_frame):
    """When only .mr is available: map original XPLOR name to trajectory atoms."""
    residue_atoms = [a for (rn,a) in sample_frame if rn==rnum]
    if not residue_atoms: return []
    # HN -> H
    if spec=='HN':
        if 'H' in residue_atoms: return ['H']
        if 'HN' in residue_atoms: return ['HN']
        return []
    # Prochiral CH2 rename Wüthrich -> IUPAC:  HB1/HB2 -> HB2/HB3 (only if residue has HB2/HB3, no HB1)
    prochiral_test = {'HB': (('HB1','HB2','HB3'),{'HB1':'HB2','HB2':'HB3'}),
                      'HG': (('HG1','HG2','HG3'),{'HG1':'HG2','HG2':'HG3'}),
                      'HD': (('HD1','HD2','HD3'),{'HD1':'HD2','HD2':'HD3'})}
    for pfx,(triple,rename) in prochiral_test.items():
        h1,h2,h3 = triple
        residue_has_ch2 = (h1 not in residue_atoms) and (h2 in residue_atoms) and (h3 in residue_atoms)
        if residue_has_ch2 and spec in rename:
            return [rename[spec]]
    # Direct match
    if spec in residue_atoms: return [spec]
    # Pseudo-atom prefix
    if spec.endswith('*'):
        pfx = spec[:-1]
        m=[a for a in residue_atoms if a.startswith(pfx)]
        if m: return m
    return []

# ------------------ distance math ------------------
def r6_avg(d):
    d=np.asarray(d, dtype=float)
    if len(d)==0: return None
    return (np.mean(d**-6.0))**(-1.0/6.0)

def per_frame_effective(members, frame):
    """Given a list of {'canon':((r1,a1),(r2,a2))}, compute per-frame <r^-6>^{-1/6}
       over all listed pairs."""
    ds=[]
    for m in members:
        (r1,a1),(r2,a2) = m['canon']
        p1=frame.get((r1,a1)); p2=frame.get((r2,a2))
        if p1 is None or p2 is None: continue
        ds.append(np.linalg.norm(p1-p2))
    return r6_avg(ds)

# ------------------ per-peptide analysis ------------------
def analyze_peptide(peptide_id):
    traj_path = os.path.join(TRAJ_DIR, f'{peptide_id}_CHCl3_Traj.pdb')
    if not os.path.exists(traj_path): return None, f'traj missing'
    frames = parse_traj(traj_path)
    sample = frames[0]

    mr_path = os.path.join(RESTRAINT_DIR, f'{peptide_id}.mr')
    mr_map = parse_mr(mr_path) if os.path.exists(mr_path) else {}

    # Try both cases of .str filename
    str_path = None
    for cand in [f'{peptide_id}_nmr-data.str', f'{peptide_id.lower()}_nmr-data.str']:
        p = os.path.join(RESTRAINT_DIR, cand)
        if os.path.exists(p): str_path = p; break

    rows=[]
    if str_path:
        source = os.path.basename(str_path)
        restraints = parse_str(str_path)
        for r in restraints:
            if r['frame_type']=='hydrogen': continue
            cat_info = lookup_mr_category(mr_map, r['auth_key'], r['upper'])
            frame_effds = [d for d in (per_frame_effective(r['members'], fr) for fr in frames) if d is not None]
            if not frame_effds:
                rows.append(dict(peptide=peptide_id, source=source, restraint_source='.str',
                                 frame_type=r['frame_type'], str_id=r['id'],
                                 category=cat_info['category'], intensity=cat_info['intensity'],
                                 mr_pair=f"r{r['auth_key'][0]}:{r['auth_key'][1]}-r{r['auth_key'][2]}:{r['auth_key'][3]}",
                                 canon_pairs=';'.join([f"{m['canon'][0][0]}:{m['canon'][0][1]}-{m['canon'][1][0]}:{m['canon'][1][1]}" for m in r['members']]),
                                 upper=r['upper'], md_r6=None, md_p5=None, md_p95=None,
                                 dev=None, violates=None, status='no_atoms'))
                continue
            arr=np.array(frame_effds)
            md=r6_avg(arr); p5,p95=np.percentile(arr,[5,95])
            rows.append(dict(peptide=peptide_id, source=source, restraint_source='.str',
                             frame_type=r['frame_type'], str_id=r['id'],
                             category=cat_info['category'], intensity=cat_info['intensity'],
                             mr_pair=f"r{r['auth_key'][0]}:{r['auth_key'][1]}-r{r['auth_key'][2]}:{r['auth_key'][3]}",
                             canon_pairs=';'.join([f"{m['canon'][0][0]}:{m['canon'][0][1]}-{m['canon'][1][0]}:{m['canon'][1][1]}" for m in r['members']]),
                             upper=r['upper'], md_r6=md, md_p5=p5, md_p95=p95,
                             dev=md-r['upper'], violates=md>r['upper'], status='ok'))
    else:
        source = os.path.basename(mr_path)
        # Fall back to .mr with heuristic expansion
        for (r1,a1,r2,a2,upper), info in mr_map.items():
            a1_traj = expand_atom_spec(r1, a1, sample)
            a2_traj = expand_atom_spec(r2, a2, sample)
            mr_pair = f"r{r1}:{a1}-r{r2}:{a2}"
            if not a1_traj or not a2_traj:
                rows.append(dict(peptide=peptide_id, source=source, restraint_source='.mr',
                                 frame_type='NOE', str_id='-',
                                 category=info['category'], intensity=info['intensity'],
                                 mr_pair=mr_pair,
                                 canon_pairs=f"a1={a1_traj}; a2={a2_traj}",
                                 upper=upper, md_r6=None, md_p5=None, md_p95=None,
                                 dev=None, violates=None, status='atom_missing'))
                continue
            members=[{'canon':((r1,t1),(r2,t2))} for t1 in a1_traj for t2 in a2_traj]
            frame_effds=[d for d in (per_frame_effective(members, fr) for fr in frames) if d is not None]
            if not frame_effds:
                rows.append(dict(peptide=peptide_id, source=source, restraint_source='.mr',
                                 frame_type='NOE', str_id='-',
                                 category=info['category'], intensity=info['intensity'],
                                 mr_pair=mr_pair,
                                 canon_pairs=';'.join([f"{r1}:{t1}-{r2}:{t2}" for t1 in a1_traj for t2 in a2_traj]),
                                 upper=upper, md_r6=None, md_p5=None, md_p95=None,
                                 dev=None, violates=None, status='no_frames'))
                continue
            arr=np.array(frame_effds)
            md=r6_avg(arr); p5,p95=np.percentile(arr,[5,95])
            rows.append(dict(peptide=peptide_id, source=source, restraint_source='.mr',
                             frame_type='NOE', str_id='-',
                             category=info['category'], intensity=info['intensity'],
                             mr_pair=mr_pair,
                             canon_pairs=';'.join([f"{r1}:{t1}-{r2}:{t2}" for t1 in a1_traj for t2 in a2_traj]),
                             upper=upper, md_r6=md, md_p5=p5, md_p95=p95,
                             dev=md-upper, violates=md>upper, status='ok'))
    return rows, 'ok'

# ------------------ main ------------------
all_rows=[]
for pid in PEPTIDES:
    rows, status = analyze_peptide(pid)
    if rows is None:
        print(f'{pid}: {status}')
        continue
    ok = sum(1 for r in rows if r['status']=='ok')
    viol = sum(1 for r in rows if r['status']=='ok' and r['violates'])
    skipped = sum(1 for r in rows if r['status']!='ok')
    print(f'{pid}: {len(rows)} restraints  ok={ok}  viol={viol}  skipped={skipped}  source={rows[0]["source"] if rows else "?"}')
    all_rows.extend(rows)

# Write CSV
with open(OUT_CSV,'w',newline='') as f:
    if not all_rows: sys.exit(1)
    cols = ['peptide','source','restraint_source','frame_type','str_id','category','intensity',
            'mr_pair','canon_pairs','upper','md_r6','md_p5','md_p95','dev','violates','status']
    w=csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in all_rows:
        row = dict(r)
        for k in ('upper','md_r6','md_p5','md_p95','dev'):
            if row.get(k) is not None: row[k]=f"{row[k]:.3f}"
        w.writerow(row)

print(f'\nWrote {OUT_CSV}')

# Per-peptide, per-category quick summary
print()
print(f'{"Peptide":<7} {"Cat":>4} {"n":>3} {"viol":>4} {"mean(Δ)":>8} {"worst(Δ)":>9}')
from statistics import mean
by = defaultdict(list)
for r in all_rows:
    if r['status']!='ok': continue
    by[(r['peptide'], r['category'])].append(r)
for (pid, cat), lst in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    devs=[r['dev'] for r in lst]
    viol=sum(1 for r in lst if r['violates'])
    print(f'{pid:<7} {str(cat):>4} {len(lst):>3} {viol:>4} {mean(devs):>+8.2f} {max(devs):>+9.2f}')
