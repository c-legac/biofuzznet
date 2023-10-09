from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

markers_to_predict = [
    "p.RB",
    "p.ERK",
    "p.JNK",
    "cleavedCas",
    "p.p38",
    "p.MKK3.MKK6",
    "p.MAPKAPK2",
    "p.p90RSK",
    "p.p53",
    "p.CREB",
    "p.H3",
    "p.MEK",
]

train_cell_lines = [
    "BT474",
    "CAL148",
    "HBL100",
    "MCF7",
    "MDAMB157",
    "T47D",
    "ZR7530",
]  # train cell lines
test_cell_lines = [
    "AU565",
    "EFM19",
    "HCC2218",
    "LY2",
    "MACLS2",
    "MDAMB436",
]  # test cell lines

cont_features = [
    "b.CATENIN",
    "cleavedCas",
    "CyclinB",
    "GAPDH",
    "IdU",
    "Ki.67",
    "p.4EBP1",
    "p.Akt.Ser473.",
    "p.AKT.Thr308.",
    "p.AMPK",
    "p.BTK",
    "p.CREB",
    "p.ERK",
    "p.FAK",
    "p.GSK3b",
    "p.H3",
    # "p.HER2",
    "p.JNK",
    "p.MAP2K3",
    "p.MAPKAPK2",
    "p.MEK",
    "p.MKK3.MKK6",
    "p.MKK4",
    "p.NFkB",
    "p.p38",
    "p.p53",
    "p.p90RSK",
    "p.PDPK1",
    # "p.PLCg2",
    "p.RB",
    "p.S6",
    "p.S6K",
    "p.SMAD23",
    "p.SRC",
    "p.STAT1",
    "p.STAT3",
    "p.STAT5",
    "time",
]

data = []
for cl in train_cell_lines + test_cell_lines:
    cl_data = pd.read_csv(
        f"/dccstor/ipc1/CAR/DREAM/DREAMdata/Time_aligned_per_cell_line/CL_incl_test/{cl}.csv"
    )
    cl_data = cl_data[cl_data["time"] == 9]
    cl_data = (
        cl_data.groupby(["cell_line", "treatment", "time"])
        .sample(n=500, replace=True)
        .reset_index(drop=False)
    )

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

train = data[data["cell_line"].isin(train_cell_lines)]
test = data[data["cell_line"].isin(test_cell_lines)]

for marker_to_predict in markers_to_predict:
    print(marker_to_predict)
    sel_features = [f for f in cont_features if f is not marker_to_predict]
    features = sel_features + cat_features
    rf = RandomForestRegressor()
    rf.fit(train[features], train[marker_to_predict])
    pred = rf.predict(test[features])

    np.save(f"/dccstor/ipc1/CAR/DREAM/Model/Baseline/{marker_to_predict}.npy", pred)
    print(rf.score(test[features], test[marker_to_predict]))
