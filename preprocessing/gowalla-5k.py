import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import json

from joblib import Parallel, delayed

from tqdm import tqdm
import pickle as pickle

from utils import split_dataset


def applyParallelPD(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.concat(df_ls)


def _get_time(df):
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["weekday"] = df["started_at"].dt.weekday
    return df

def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return df_ls

def getValidSequence(input_df, previous_day):
    valid_user_ls = applyParallel(
        input_df.groupby("user_id"),
        getValidSequenceUser,
        previous_day=previous_day,
        n_jobs=-1,
    )
    return [item for sublist in valid_user_ls for item in sublist]

    # This function works for a single user's check-ins to identify the valid staypoints based on a given number of previous_day lookback. 
    # It ensures that there is enough activity (at least 3 records) in the past previous_day days for a valid sequence.
def getValidSequenceUser(df, previous_day):
    df.reset_index(drop=True, inplace=True)

    valid_id = []
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days

    for index, row in df.iterrows(): # Iterates over each row in the DataFrame, where row represents a specific check-in, and index is its position in the DataFrame.
        # exclude the first records
        if row["diff_day"] < previous_day: # Skips any check-in where the time difference from the earliest check-in (diff_day) is less than previous_day. 
                                           # This ensures that we only consider check-ins that have enough historical data.
            continue

        hist = df.iloc[:index]
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]

        # exclude series which contains too few records
        if len(hist) < 3:
            continue
        valid_id.append(row["id"])

    return valid_id

    # The function enrich_time_info adds time-related features to each staypoint, transforming the raw started_at times into useful features like day of the week, 
    # minute of the day, and the number of days since the first check-in for each user.
def enrich_time_info(sp):
    tqdm.pandas(desc="Time enriching")
    sp = applyParallelPD(sp.groupby("user_id", group_keys=False), _get_time, n_jobs=-1, print_progress=True)
    #     sp.groupby("user_id", group_keys=False).progress_apply(_get_time)
    sp.drop(columns={"started_at"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    #
    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    # final cleaning, reassign ids
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def get_dataset(config):
    print(f"Test Dataset Preprocessing")
    # note: location does not start at 0
    gowalla = pd.read_csv(
        os.path.join(config[f"raw_gowalla"], "test-dataset.txt"),
        sep="\t",               # specifies the file is tab-separated.
        header=None,            # indicates the file doesn't have a header row.
        parse_dates=[1],        # converts the second column (start times) into datetime objects.
        names=["user_id", "started_at", "latitude", "longitude", "location_id"], # defines custom column names for the DataFrame, allowing the code to refer to user IDs, 
                                                                                 # timestamps, and geographical coordinates easily
    )

    gowalla_enriched = enrich_time_info(gowalla)

    # Filter infrequent user
    user_size = gowalla_enriched.groupby(["user_id"]).size()
    valid_users = user_size[user_size > 10].index
    gowalla_enriched = gowalla_enriched.loc[gowalla_enriched["user_id"].isin(valid_users)]

    # Filter infrequent POIs
    poi_size = gowalla_enriched.groupby(["location_id"]).size()
    valid_pois = poi_size[poi_size > 10].index
    gowalla_enriched = gowalla_enriched.loc[gowalla_enriched["location_id"].isin(valid_pois)]

    # split into train vali and test
    train_data, vali_data, test_data = split_dataset(gowalla_enriched)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value", # The encoder is set with handle_unknown="use_encoded_value" and unknown_value=-1, which means that any unseen locations 
                                            # (in the validation or test sets) will be assigned the value -1.
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    
    # add 2 to account for unseen locations and to account for 0 padding
    # Locations that exist in the validation and test sets but are not present in the training data are unseen.

    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2 # After encoding, 2 is added to the resulting values. This is to account for:
                                                                                                   # 1. Unseen locations, which were assigned -1, will now become 1 (representing unseen locations in a meaningful way).
                                                                                                   # 2. Padding Values: In some models, 0 is reserved for padding purposes, so we add 2 to shift all encoded location IDs to avoid conflicts with this padding index.
                                                                                                   # After fitting the encoder on the training data, the transformation is applied back to the location_id field of the train_data. 
                                                                                                   # The reshape(-1, 1) ensures that the data is correctly shaped as a column for the encoder.
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    # the days to consider when generating final_valid_id
    all_ids = gowalla_enriched[["id"]].copy() # This extracts just the id column from gowalla_enriched and creates a new DataFrame all_ids. 
                                              # It serves as a container to later mark which staypoints are valid across the training, validation, and test sets.

    # for each previous_day, get the valid staypoint id
    valid_ids = getValidSequence(train_data, previous_day=7)
    valid_ids.extend(getValidSequence(vali_data, previous_day=7))
    valid_ids.extend(getValidSequence(test_data, previous_day=7))

    all_ids["7"] = 0 # A new column "7" is added to all_ids, initialized to 0. This will later be used to mark which ids (staypoints) are valid.
    all_ids.loc[all_ids["id"].isin(valid_ids), f"7"] = 1 # If an id in all_ids is in the valid_ids list, its corresponding "7" column value is set to 1, 
                                                         # indicating that it's a valid staypoint for the past 7 days.

    # get the final valid staypoint id
    all_ids.set_index("id", inplace=True) # Sets id as the index of all_ids
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values # Only staypoints that are marked as valid across all splits (training, validation, and testing) are selected.
                                                                         # The final valid ids are reset to a regular column and then converted to a NumPy array (final_valid_id), 
                                                                         # containing the staypoint IDs that are valid across all sets.                   

    # filter the user again based on final_valid_id:
    # if an user has no record in final_valid_id, we discard the user
    valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()

    valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))

    gowalla_afterUser = gowalla_enriched.loc[gowalla_enriched["user_id"].isin(valid_users)].copy()

    train_data, vali_data, test_data = split_dataset(gowalla_afterUser) # fter filtering users based on the final valid staypoint ID, this line re-splits the gowalla_afterUser dataset 
                                                                        # into new training, validation, and test sets using the same function (split_dataset).
                                                                        # The idea is to ensure the filtered dataset is divided properly into these three subsets, 
                                                                        # just like the original data before filtering, but now with only the users that passed 
                                                                        # the previous filtering step.
    gowalla_afterUser = test_data


    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    # add 2 to account for unseen locations and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    #  after user filter, we reencode the users, to ensure the user_id is continues
    # we do not need to encode the user_id again in dataloader.py
    # enc = OrdinalEncoder(dtype=np.int64)
    gowalla_afterUser["user_id"] = enc.fit_transform(gowalla_afterUser["user_id"].values.reshape(-1, 1)) + 1

    # normalize the coordinates. implementation following the original paper
    gowalla_afterUser["longitude"] = (
        2
        * (gowalla_afterUser["longitude"] - gowalla_afterUser["longitude"].min())
        / (gowalla_afterUser["longitude"].max() - gowalla_afterUser["longitude"].min())
        - 1
    )
    gowalla_afterUser["latitude"] = (
        2
        * (gowalla_afterUser["latitude"] - gowalla_afterUser["latitude"].min())
        / (gowalla_afterUser["latitude"].max() - gowalla_afterUser["latitude"].min())
        - 1
    )
    gowalla_loc = (
        gowalla_afterUser.groupby(["location_id"])
        .head(1)
        .drop(columns={"id", "user_id", "start_day", "start_min", "weekday"})
    )
    gowalla_loc = gowalla_loc.rename(columns={"location_id": "id"})

    print(
        f"Max user id:{gowalla_afterUser.user_id.max()}, unique user id:{gowalla_afterUser.user_id.unique().shape[0]}"
    )

    # save the valid_ids and dataset
    data_path = f"./data/test/valid_ids_gowalla.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gowalla_afterUser.to_csv(f"./data/test/dataset_gowalla.csv", index=False)
    gowalla_loc.to_csv(f"./data/test/locations_gowalla.csv", index=False)

    print("Final user size: ", gowalla_afterUser["user_id"].unique().shape[0])


if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    get_dataset(config=CONFIG)
