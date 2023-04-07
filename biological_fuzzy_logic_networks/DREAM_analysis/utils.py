import numpy as np


def inhibitor_mapping(reverse: bool = False):

    inhibitor_mapping = {
        "EGF": np.nan,
        "full": np.nan,
        "iEGFR": "EGFR",
        "iMEK": "MEK12_S221",
        "iPI3K": "PI3K",
        "iPKC": "PKC",
    }

    if reverse:
        return {v: k for k, v in inhibitor_mapping.items()}
    else:
        return inhibitor_mapping


def data_to_nodes_mapping():

    data_to_nodes = {
        "EGF": "EGF",
        "SERUM": "SERUM",
        "b.CATENIN": "b-catenin",
        "cleavedCas": "cleavedCas",
        "p.4EBP1": "4EBP1",
        "p.Akt.Ser473.": "AKT_S473",
        "p.AKT.Thr308.": "AKT_T308",
        "p.AMPK": "AMPK",
        "p.BTK": "BTK",
        "p.CREB": "CREB",
        "p.ERK": "ERK12",
        "p.FAK": "FAK",
        "p.GSK3b": "GSK3B",
        "p.H3": "H3",
        "p.JNK": "JNK",
        "p.MAP2K3": "MAP3Ks",
        "p.MAPKAPK2": "MAPKAPK2",
        "p.MEK": "MEK12_S221",
        "p.MKK3.MKK6": "MKK36",
        "p.MKK4": "MKK4",
        "p.NFkB": "NFkB",
        "p.p38": "p38",
        "p.p53": "p53",
        "p.p90RSK": "p90RSK",
        "p.PDPK1": "PDPK1",
        "p.PLCg2": "PLCg2",
        "p.RB": "RB",
        "p.S6": "S6",
        "p.S6K": "p70S6K",
        "p.SMAD23": "SMAD23",
        "p.SRC": "SRC",
        "p.STAT1": "STAT1",
        "p.STAT3": "STAT3",
        "p.STAT5": "STAT5",
    }

    return data_to_nodes
