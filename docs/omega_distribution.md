# ω (peptide-bond) distribution — note for Figure 6b

The CycPeptMPDB-4D ω distribution is broader than CREMP (trans-peak std ≈ 12° vs ≈ 4°;
21% of trans bonds sit >15° from planar in 4D, essentially 0% in CREMP). This is expected,
not an artifact, for two reasons:

1. **Composition — N-substituted bonds.** CycPeptMPDB-4D is rich in N-substituted
   (N-methylated / N-benzyl) residues — a standard permeability motif — and these bonds are
   intrinsically more flexible because N-substitution reduces amide planarity/rigidity.
   Per-residue ω widths bear this out: N-substituted residues (MLE, MPH, NBZ, MAL, NPR) give
   trans-std ≈ 13–15°, versus ≈ 11° for standard residues (LEU, ALA, PRO, PHE). N-methyl-Leu
   (MLE) alone is the second-most-common peptide bond in the set (~12,300 instances), so the
   aggregate is broad largely by composition. **CREMP, by contrast, contains only natural
   (proteinogenic) amino acids** — none of these flexible N-substituted bonds — so its ω is
   intrinsically narrow. The comparison therefore partly reflects a difference in chemistry,
   not only method.

2. **Method — classical MD vs QM-refined conformers.** CycPeptMPDB-4D ω comes from
   finite-temperature classical MD (AMBER99SB), which samples thermal fluctuation of a softer
   amide torsion; CREMP is a QM-refined conformer ensemble. Classical MD gives inherently
   broader ω (~2× the crystallographic ~6°) even for standard residues, which accounts for the
   ~11° baseline seen above.

The breadth is spread uniformly across residue types (no single monomer dominates), and the
non-standard β-amino acid BHF is unremarkable (trans-std 12.4°), ruling out a backbone-detection
artifact. Making the 4D curve as sharp as CREMP would understate the real flexibility of the
N-substituted bonds this dataset is built from.

(cis bonds are largely absent from the 4D curve here because cis conformations were removed
upstream; CREMP retains them, ~5.5%.)
