from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

markers_to_predict = [
    "p.ERK",
]

train_cell_lines = ["BT20"]  # train cell lines
treatments = ["EGF", "iEGFR", "iMEK", "iPI3K", "iPKC", "imTOR"]
test_cell_lines = []  # test cell lines

# Subnetwork inputs
cont_features = [
    "p.MEK",
    "p.FAK",
]


data = []
for cl in train_cell_lines + test_cell_lines:
    for tr in treatments:  # First time I loop over treatments :p
        cl_data = pd.read_csv(
            f"/dccstor/ipc1/CAR/DREAM/DREAMdata/Time_aligned_per_cell_line/CL_incl_test/{cl}_{tr}_time9.csv"
        )
        cl_data = cl_data[cl_data["time"] == 9]

        data.append(cl_data)

data = pd.concat(data)

for treatment in treatments:  # Second time I loop over treatments, whoopsie
    print(treatment)

    tr_data = data[data["treatment"] == treatment]
    train, test = train_test_split(tr_data)

    np.save(
        f"/dccstor/ipc1/CAR/DREAM/Model/Baseline/RF_OneCellLineOneTreatment_MEK_FAK_ERK_MF_inputs/RF_OneCellLine_{treatment}_train.npy",
        train,
    )

    np.save(
        f"/dccstor/ipc1/CAR/DREAM/Model/Baseline/RF_OneCellLineOneTreatment_MEK_FAK_ERK_MF_inputs/RF_OneCellLine_{treatment}_test.npy",
        test,
    )

    for marker_to_predict in markers_to_predict:
        print(marker_to_predict)
        features = [f for f in cont_features if f is not marker_to_predict]
        rf = RandomForestRegressor()
        rf.fit(train[features], train[marker_to_predict])
        pred = rf.predict(test[features])

        np.save(
            f"/dccstor/ipc1/CAR/DREAM/Model/Baseline/RF_OneCellLineOneTreatment_MEK_FAK_ERK_MF_inputs/{treatment}_{marker_to_predict}.npy",
            pred,
        )
        print(rf.score(test[features], test[marker_to_predict]))
