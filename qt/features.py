
import numpy as np
from protlearn import features
from protlearn.features import ctd
from protlearn.features import qso

def CTDC(seq):
    seqs = [seq]
    ctd_arr, ctd_desc = features.ctdc(seqs)
    ctd_arr = ctd_arr[0]
    ctd_list = np.array(ctd_arr).tolist()
    return ctd_list

def AAC(seq):
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    res = []
    for x in std:
        tmp = seq.count(x) / len(seq)
        res.append(tmp)
    return res

def DPC(seq):
    mer2_dict = {}
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    for x in std:
        for y in std:
            tmp = x + y
            mer2_dict[tmp] = 0

    res_dict = {}
    for k, v in mer2_dict.items():
        tmp = seq.count(k)
        res_dict[k] = tmp
    res = list(res_dict.values())
    return res

def GDPC(seq):
    '''
    c (g1: IMLVAG), aromatic (g2: WYF), positive charge (g3: HRK), negative charge (g4: ED) and uncharged (g5:
    QNPCTS)
    '''

    g1 = "IMLVAG"
    g2 = "WYF"
    g3 = "HRK"
    g4 = "ED"
    g5 = "QNPCTS"

    res = []
    for gx in [g1, g2, g3, g4, g5]:
        for gy in [g1, g2, g3, g4, g5]:
            tmp = 0
            for x in gx:
                for y in gy:
                    tmp = tmp + seq.count(x + y)
            tmp = tmp / len(seq)
            res.append(tmp)
    return res


def GTPC(seq):
    '''
    c (g1: IMLVAG), aromatic (g2: WYF), positive charge (g3: HRK), negative charge (g4: ED) and uncharged (g5:
    QNPCTS)
    '''

    g1 = "IMLVAG"
    g2 = "WYF"
    g3 = "HRK"
    g4 = "ED"
    g5 = "QNPCTS"

    res = []
    for gx in [g1, g2, g3, g4, g5]:
        for gy in [g1, g2, g3, g4, g5]:
            for gz in [g1, g2, g3, g4, g5]:
                tmp = 0
                for x in gx:
                    for y in gy:
                        for z in gz:
                            tmp = tmp + seq.count(x + y + z)
                tmp = tmp / len(seq)
                res.append(tmp)
    return res


def CTF(seq):
    '''
    seven groups:
    g1 = "AVG"
    g2 = "FLIP"
    g3 = "TSYM"
    g4 = "HQNW"
    g5 = "RK"
    g6 = "DE"
    g7 = "C"
    '''
    g1 = "AVG"
    g2 = "FLIP"
    g3 = "TSYM"
    g4 = "HQNW"
    g5 = "RK"
    g6 = "DE"
    g7 = "C"

    res = []
    for gx in [g1, g2, g3, g4, g5, g6, g7]:
        for gy in [g1, g2, g3, g4, g5, g6, g7]:
            for gz in [g1, g2, g3, g4, g5, g6, g7]:
                tmp = 0
                for x in gx:
                    for y in gy:
                        for z in gz:
                            tmp = tmp + seq.count(x + y + z)
                tmp = tmp / len(seq)
                res.append(tmp)
    return res


def get_QSO(seq):
    seqs = [seq * 3]
    sw, g, desc = features.qso(seqs, d=3, remove_zero_cols=False)
    sw = np.array(sw)[0].tolist()
    g = np.array(g)[0].tolist()
    res = sw + g
    return res


def get_PAAC(seq):
    seqs = [seq * 3]
    paac_comp, desc = features.paac(seqs, lambda_=3, remove_zero_cols=False)
    paac_comp = np.array(paac_comp)[0].tolist()
    return paac_comp


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def AAC(seq):
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    res = []
    for x in std:
        tmp = seq.count(x) / len(seq)
        res.append(tmp)
    return res

def DPC(seq):
    mer2_dict = {}
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    for x in std:
        for y in std:
            tmp = x + y
            mer2_dict[tmp] = 0

    res_dict = {}
    for k, v in mer2_dict.items():
        tmp = seq.count(k)
        res_dict[k] = tmp
    res = list(res_dict.values())
    return res

def GDPC(seq):
    '''
    c (g1: IMLVAG), aromatic (g2: WYF), positive charge (g3: HRK), negative charge (g4: ED) and uncharged (g5:
    QNPCTS)
    '''

    g1 = "IMLVAG"
    g2 = "WYF"
    g3 = "HRK"
    g4 = "ED"
    g5 = "QNPCTS"

    res = []
    for gx in [g1, g2, g3, g4, g5]:
        for gy in [g1, g2, g3, g4, g5]:
            tmp = 0
            for x in gx:
                for y in gy:
                    tmp = tmp + seq.count(x + y)
            tmp = tmp / len(seq)
            res.append(tmp)
    return res

def GTPC(seq):
    '''
    c (g1: IMLVAG), aromatic (g2: WYF), positive charge (g3: HRK), negative charge (g4: ED) and uncharged (g5:
    QNPCTS)
    '''

    g1 = "IMLVAG"
    g2 = "WYF"
    g3 = "HRK"
    g4 = "ED"
    g5 = "QNPCTS"

    res = []
    for gx in [g1, g2, g3, g4, g5]:
        for gy in [g1, g2, g3, g4, g5]:
            for gz in [g1, g2, g3, g4, g5]:
                tmp = 0
                for x in gx:
                    for y in gy:
                        for z in gz:
                            tmp = tmp + seq.count(x + y + z)
                tmp = tmp / len(seq)
                res.append(tmp)
    return res


def BPF(seq):
    std = ["A", "I", "L", "V", "F", "W", "Y", "N", "C", "Q", "M", "S", "T", "R", "H", "K", "D", "E", "G", "P"]
    seq = seq[0:10]
    res = []
    for x in seq:
        tmp = [0] * 20
        tmp[std.index(x)] = 1
        res.extend(tmp)
    while len(res) <= 400:
        res.append(0)
    return res


def QSO(seq):
    seqs = [seq]
    sw, g, desc = qso(seqs, d=1, remove_zero_cols=False)
    sw = list(sw[0])
    g = list(g[0])
    return sw + g

def CTD(seq):
    seqs = [seq]
    ctd_arr, ctd_desc = ctd(seqs)
    ctd_arr = list(ctd_arr[0])
    return ctd_arr

def CTF(seq):
    '''
    seven groups:
    g1 = "AVG"
    g2 = "FLIP"
    g3 = "TSYM"
    g4 = "HQNW"
    g5 = "RK"
    g6 = "DE"
    g7 = "C"
    '''
    g1 = "AVG"
    g2 = "FLIP"
    g3 = "TSYM"
    g4 = "HQNW"
    g5 = "RK"
    g6 = "DE"
    g7 = "C"

    res = []
    for gx in [g1, g2, g3, g4, g5, g6, g7]:
        for gy in [g1, g2, g3, g4, g5, g6, g7]:
            for gz in [g1, g2, g3, g4, g5, g6, g7]:
                tmp = 0
                for x in gx:
                    for y in gy:
                        for z in gz:
                            tmp = tmp + seq.count(x + y + z)
                tmp = tmp / len(seq)
                res.append(tmp)
    return res


def ATC(seq):
    seqs = [seq]
    atoms, bonds = features.atc(seqs)
    atoms = atoms[0]
    bonds = bonds[0]
    atoms = np.array(atoms).tolist()
    bonds = np.array(bonds).tolist()
    res = atoms + bonds
    return res

def Hydrophobicity(seq):
    '''
    I	4.5
    V	4.2
    L	3.8
    F	2.8
    C	2.5
    M	1.9
    A	1.8
    G	-0.4
    T	-0.7
    S	-0.8
    W	-0.9
    Y	-1.3
    P	-1.6
    H	-3.2
    E	-3.5
    Q	-3.5
    D	-3.5
    N	-3.5
    K	-3.9
    R	-4.5
    '''
    std_Hyd = {
        "I": "4.5  ",
        "V": "4.2  ",
        "L": "3.8  ",
        "F": "2.8  ",
        "C": "2.5  ",
        "M": "1.9  ",
        "A": "1.8  ",
        "G": "-0.4",
        "T": "-0.7",
        "S": "-0.8",
        "W": "-0.9",
        "Y": "-1.3",
        "P": "-1.6",
        "H": "-3.2",
        "E": "-3.5",
        "Q": "-3.5",
        "D": "-3.5",
        "N": "-3.5",
        "K": "-3.9",
        "R": "-4.5",
    }

    seq = seq[0:50]
    res = np.zeros(50)
    for i, x in enumerate(seq):
        res[i] = std_Hyd[x]
    return list(res)


def Charge(seq):
    '''
    G	5.97
    A	6
    S	5.68
    P	6.3
    V	5.96
    T	6.16
    C	5.07
    I	6.02
    L	5.98
    N	5.41
    D	2.77
    Q	5.65
    K	9.74
    E	3.22
    M	5.74
    H	7.59
    F	5.48
    R	10.76
    Y	5.66
    W	5.89

    '''
    std_charge = {

        "G": 5.97,
        "A": 6,
        "S": 5.68,
        "P": 6.3,
        "V": 5.96,
        "T": 6.16,
        "C": 5.07,
        "I": 6.02,
        "L": 5.98,
        "N": 5.41,
        "D": 2.77,
        "Q": 5.65,
        "K": 9.74,
        "E": 3.22,
        "M": 5.74,
        "H": 7.59,
        "F": 5.48,
        "R": 10.76,
        "Y": 5.66,
        "W": 5.89,

    }

    seq = seq[0:50]
    res = np.zeros(50)
    for i, x in enumerate(seq):
        res[i] = std_charge[x]
    return list(res)

def fs_encode(seq):
    x1 = AAC(seq)
    x3 = GDPC(seq)
    x5 = BPF(seq)
    x6 = QSO(seq)
    x7 = CTD(seq)
    x9 = ATC(seq)
    x16 = Charge(seq)
    res = x1 + x3 + x5 + x6 + x7 + x9 + x16
    return res


'''
>wer
LPLLAGLAANFLPKIFCKITRK
'''
# res = fs_encode("LPLLAGLAANFLPKIFCKITRK")
# print(res)



