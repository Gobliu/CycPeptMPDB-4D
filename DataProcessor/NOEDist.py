import numpy as np
from OmegaComputor import read_pdb


def noe_dict_checker(pdb_path, noe_list):
    c_list = read_pdb(pdb_path)
    for idx1, idx2 in noe_list:
        dist_list = []
        for conf in c_list:
            assert conf[idx1-1][-1] == 'H' and conf[idx2-1][-1] == 'H', f'Target atom {idx1} or {idx2} is not H'
            p0 = np.asarray([float(i) for i in conf[idx1-1][:3]])
            p1 = np.asarray([float(i) for i in conf[idx2-1][:3]])
            dist = np.linalg.norm(p0 - p1)
            dist_list.append(dist)
            # print(dist, conf[idx1-1][:3], conf[idx1-1][:3])
        r_eff = (np.mean(np.asarray(dist_list) ** -6)) ** (-1 / 6)
        print(f'atom {idx1} atom {idx2} dist {r_eff}')


if __name__ == '__main__':
    pdb_path = '../../../Data/trjs/7l9d_100frames.pdb'
    noe_list = [[42, 46], [48, 33], [48, 67], [75, 79]]   # 7l9d
    # noe_list = [[4, 2], [4, 21], [2, 21], [23, 21], [42, 59], [42, 40], [40, 59], [61, 80], [61, 79],
    #             [112, 2], [112, 21]]    # 7l96
    # noe_list = [[31, 49], [31, 45], [78, 82]]   # 7l98
    # noe_list = [[12, 29], [25, 29], [25, 39], [29, 39], [29, 41], [29, 77], [29, 96], [41, 73], [41, 77],
    #             [43, 77], [73, 96], [73, 129], [77, 96], [96, 116], [96, 117], [125, 129]]    # 7uzl
    # noe_list = [[12, 29], [25, 39], [29, 39], [29, 129], [77, 129], [79, 96], [96, 129],
    #             [98, 125], [98, 129], [98, 136], [103, 125]]  # 7uzl
    # noe_list = [[2, 73], [2, 121], [2, 125], [2, 127], [6, 73], [61, 73], [73, 121], [73, 125], [121, 125]]     # 8cwa
    c_list = read_pdb(pdb_path)
    print(len(c_list), len(c_list[0]))
    noe_dict_checker(pdb_path, noe_list)
    # for idx, line in enumerate(c_list[0]):
    #     print(idx+1, line)
