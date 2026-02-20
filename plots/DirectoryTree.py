import matplotlib.pyplot as plt

text = r"""
CycPeptMPDB-4D/
│
├── Water/                      (5,152 peptides)
│   ├── Trajectories/           *.pdb trajectories (100 frames each)
│   │   ├── XXX_YYY_H2O_Traj.pdb
│   │   └── ...
│   ├── Structures/             representative conformations
│   │   ├── XXX_YYY_H2O_Str.pdb
│   │   └── ...
│   └── Logs/                   clustering logs
│       ├── XXX_YYY_H2O.log
│       └── ...
├── Hexane/                     (5,152 peptides)
│   ├── Trajectories/
│   │   ├── XXX_YYY_Hexane_Traj.pdb
│   │   └── ...
│   ├── Structures/
│   │   ├── XXX_YYY_Hexane_Str.pdb
│   │   └── ...
│   └── Logs/
│       ├── XXX_YYY_Hexane.log
│       └── ...
├── CHCl3/                      (validation set, 6 peptides)
│   ├── 7L98_CHCl3_Traj.pdb
│   └── ...
└── CycPeptMPDB-4D.csv          metadata & molecular descriptors
"""

fig = plt.figure(figsize=(7, 0.01), facecolor="white")   # tiny height → autoscale to text
plt.text(0.0, 1.0, text,
         ha='left', va='top',
         family='monospace', fontsize=13, color='black')
plt.axis('off')

fig.tight_layout(pad=0)
plt.savefig("directory_tree.png", dpi=600, facecolor='white',
            bbox_inches='tight', pad_inches=0)
plt.savefig("directory_tree.pdf", dpi=600, facecolor='white',
            bbox_inches='tight', pad_inches=0)
plt.show()
