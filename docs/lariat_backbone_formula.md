# Lariat Backbone Ring Formula

How `OmegaComputer` finds the backbone **macrocycle** of Lariat peptides so it can
compute ω (omega) dihedral angles. This is about the *length of the backbone ring* —
**not** chirality.

## Summary

All 1,396 Lariat peptides in CycPeptMPDB-4D are from 2021_Kelly. The backbone-ring atom
count is:

```
cycle_length = Monomer_Length_in_Main_Chain * 3 + n_extra
```

where `n_extra` counts backbone-extending residues (each adds one atom to the ring). For
Lariats that is the **TNH** residue — exactly one per Kelly Lariat, so `+1`. The naive
`Monomer_Length_in_Main_Chain * 3` (and, worse, using the full `Monomer_Length`) is wrong.

## Lariat distribution

- Total: **1,396** (100% `2021_Kelly`)
- `Monomer_Length` (ML): always **10**
- `Monomer_Length_in_Main_Chain` (ML_MC): **7** (78.6%), **8** (15.3%), **9** (6.1%)

## Why TNH adds one atom

`TNH` (Threonine with an N-terminal hook) closes the macrocycle through its **side chain**.
In the ring it contributes `CA → CB → OG1 → C` (4 atoms) instead of the standard
`N → CA → C` (3 atoms) — so `+1` per TNH.

## Ring length by ML_MC (verified against structures)

| ML_MC | ring atoms = ML_MC×3 + 1 | example verified |
|:-----:|:-----------------------:|------------------|
| 7 | 22 | 2021_Kelly_5670 |
| 8 | 25 | 2021_Kelly_6769 |
| 9 | 28 | 2021_Kelly_6983 |

Example — `2021_Kelly_5670` (ML_MC=7): TNH contributes 4 atoms (`CA, CB, OG1, C`) + 6
standard residues × 3 = **22 atoms**.

## Derivation

```
cycle_length = (ML_MC - n_TNH) * 3  +  n_TNH * 4
             = ML_MC * 3  +  n_TNH
```

With `n_TNH = 1` (one TNH per Kelly Lariat) → `ML_MC * 3 + 1`.

## How the code implements it

`OmegaComputer.count_extra_backbone_atoms(pdb)` reads residue names from the PDB and counts
**`BHF`** (β-homo amino acid) and **`TNH`** (Lariat hook) residues — each adds 1 to the ring
length. They are folded together into a single `n_extra`, not separate parameters:

```python
loop_len = residue_len * 3 + n_extra        # in find_backbone_cycle()
# residue_len = row.Monomer_Length_in_Main_Chain
# n_extra     = count_extra_backbone_atoms(pdb)   # count of BHF + TNH residues
```

Reading residue names is fast (one PDB pass), unambiguous, and needs no extra graph search.

## CSV field meanings

- **`Monomer_Length` (ML)** — every residue: N-cap + macrocycle + C-terminal tail. Kelly
  Lariats: always 10.
- **`Monomer_Length_in_Main_Chain` (ML_MC)** — residues in the macrocyclic ring only. This
  is the correct value for the ring formula. Kelly Lariats: 7/8/9.
- **ML − ML_MC** — the N-cap (`ac-`) + tail residues, which sit *outside* the ring:
  ML_MC=7 → diff 3 (cap + 2 tail), ML_MC=8 → diff 2, ML_MC=9 → diff 1 (cap only).
