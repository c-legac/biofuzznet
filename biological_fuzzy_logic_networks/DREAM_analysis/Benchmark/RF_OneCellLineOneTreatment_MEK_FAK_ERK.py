from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

markers_to_predict = [
    "p.ERK",
]

train_cell_lines = ["BT20"]  # train cell lines
treatments = ["EGF", "iEGFR", "iMEK", "iPI3K", "iPKC", "imTOR"]
test_cell_lines = []  # test cell lines

# All continous features
# cont_features = [
#     "b.CATENIN",
#     "cleavedCas",
#     "CyclinB",
#     "GAPDH",
#     "IdU",
#     "Ki.67",
#     "p.4EBP1",
#     "p.Akt.Ser473.",
#     "p.AKT.Thr308.",
#     "p.AMPK",
#     "p.BTK",
#     "p.CREB",
#     "p.ERK",
#     "p.FAK",
#     "p.GSK3b",
#     "p.H3",
#     # "p.HER2",
#     "p.JNK",
#     "p.MAP2K3",
#     "p.MAPKAPK2",
#     "p.MEK",
#     "p.MKK3.MKK6",
#     "p.MKK4",
#     "p.NFkB",
#     "p.p38",
#     "p.p53",
#     "p.p90RSK",
#     "p.PDPK1",
#     # "p.PLCg2",
#     "p.RB",
#     "p.S6",
#     "p.S6K",
#     "p.SMAD23",
#     "p.SRC",
#     "p.STAT1",
#     "p.STAT3",
#     "p.STAT5",
#     "time",
# ]

# Subnetwork inputs
cont_features = [
    "p.MEK",
    "p.FAK",
]


data = []
for cl in train_cell_lines + test_cell_lines:
    for tr in treatments:
        cl_data = pd.read_csv(
            f"/dccstor/ipc1/CAR/DREAM/DREAMdata/Time_aligned_per_cell_line/CL_incl_test/{cl}_{tr}_time9.csv"
        )
        cl_data = cl_data[cl_data["time"] == 9]

        data.append(cl_data)

data = pd.concat(data)

data["stimulation"] = 1
data.loc[data["time"] == 0, "stimulation"] = 0
data.loc[data["treatment"] == "full", "stimulation"] = 0

data["starvation"] = 1
data.loc[data["treatment"] == "full", "starvation"] = 0

cat_data = pd.DataFrame(data["treatment"])
encoder = OneHotEncoder()
cat_data_encoded = encoder.fit_transform(cat_data)
data[np.concatenate(encoder.categories_, axis=0)] = cat_data_encoded.todense()
# data = data.drop("treatment", axis=1)

cat_features = list(np.concatenate(encoder.categories_, axis=0)) + [
    "starvation",
    "stimulation",
]

train, test = train_test_split(data)

np.save(
    "/dccstor/ipc1/CAR/DREAM/Model/Baseline/RF_OneCellLineOneTreatment_MEK_FAK_ERK_MF_inputs/RF_OneCellLine_train.npy",
    train,
)

np.save(
    "/dccstor/ipc1/CAR/DREAM/Model/Baseline/RF_OneCellLineOneTreatment_MEK_FAK_ERK_MF_inputs/RF_OneCellLine_test.npy",
    test,
)

for marker_to_predict in markers_to_predict:
    print(marker_to_predict)
    sel_features = [f for f in cont_features if f is not marker_to_predict]
    features = sel_features + cat_features
    rf = RandomForestRegressor()
    rf.fit(train[features], train[marker_to_predict])
    pred = rf.predict(test[features])

    np.save(
        f"/dccstor/ipc1/CAR/DREAM/Model/Baseline/RF_OneCellLineOneTreatment_MEK_FAK_ERK_MF_inputs/{marker_to_predict}.npy",
        pred,
    )
    print(rf.score(test[features], test[marker_to_predict]))
