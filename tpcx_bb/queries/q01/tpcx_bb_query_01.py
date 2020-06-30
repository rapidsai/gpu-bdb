#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from xbb_tools.utils import benchmark, tpcxbb_argparser, run_dask_cudf_query
from xbb_tools.readers import build_reader


### Implementation Notes:
# `drop_duplicates` and `groupby` by default brings result to single partition
# Have changed `drop_duplicates` behaviour to give `n_workers` partitions
# This can change performence chars at larger scales

### Future Notes:
# Settinng  index + merge using  map_parition can be a work-around if dask native merge is slow


cli_args = tpcxbb_argparser()

# -------- Q1 -----------
q01_i_category_id_IN = [1, 2, 3]
# -- sf1 -> 11 stores, 90k sales in 820k lines
q01_ss_store_sk_IN = [10, 20, 33, 40, 50]
q01_viewed_together_count = 50
q01_limit = 100


item_cols = ["i_item_sk", "i_category_id"]
ss_cols = ["ss_item_sk", "ss_store_sk", "ss_ticket_number"]


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables():
    table_reader = build_reader(
        data_format=cli_args["file_format"],
        basepath=cli_args["data_dir"],
        split_row_groups=cli_args["split_row_groups"],
    )
    
    item_df = table_reader.read("item", relevant_cols=item_cols)
    ss_df = table_reader.read("store_sales", relevant_cols=ss_cols)
    return item_df, ss_df


### Inner Self join to get pairs
#     Select t1.ss_item_sk as item_sk_1 , t2.ss_item_sk as item_sk_2
#     FROM (
#     ...
#     ) t1 Inner Join
#     (
#    ...
#     ) t2
#     ON t1.ss_ticket_number == t2.ss_ticket_number
#     Where
#     t1.ss_item_sk < t2.ss_item_sk


@benchmark(dask_profile=cli_args["dask_profile"])
def get_pairs(
    df,
    col_name="ss_item_sk",
    merge_col="ss_ticket_number",
    pair_col="ss_item_sk",
    output_col_1="item_sk_1",
    output_col_2="item_sk_2",
):
    pair_df = df.merge(df, on=merge_col, suffixes=["_t1", "_t2"])
    pair_df = pair_df[[f"{pair_col}_t1", f"{pair_col}_t2"]]
    pair_df = pair_df[
        pair_df[f"{pair_col}_t1"] < pair_df[f"{pair_col}_t2"]
    ].reset_index(drop=True)
    pair_df = pair_df.rename(
        columns={f"{pair_col}_t1": output_col_1, f"{pair_col}_t2": output_col_2}
    )
    return pair_df


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client):

    item_df, ss_df = read_tables()

    # SELECT DISTINCT ss_item_sk,ss_ticket_number
    # FROM store_sales s, item i
    # -- Only products in certain categories sold in specific stores are considered,
    # WHERE s.ss_item_sk = i.i_item_sk
    # AND i.i_category_id IN ({q01_i_category_id_IN})
    # AND s.ss_store_sk IN ({q01_ss_store_sk_IN})

    f_ss_df = ss_df.loc[ss_df["ss_store_sk"].isin(q01_ss_store_sk_IN)][
        ["ss_item_sk", "ss_ticket_number"]
    ].reset_index(drop=True)

    f_item_df = item_df.loc[item_df["i_category_id"].isin(q01_i_category_id_IN)][
        ["i_item_sk"]
    ].reset_index(drop=True)

    ss_item_join = f_item_df.merge(
        f_ss_df, left_on=["i_item_sk"], right_on=["ss_item_sk"]
    )
    ss_item_join = ss_item_join[["ss_item_sk", "ss_ticket_number"]]

    ## keep to a  single partitions
    ## We only have 41,910,265 rows in the dataframe at sf-10k and dont need to split_out.
    ss_item_join = ss_item_join.drop_duplicates()

    ### do pair inner join
    pair_df = get_pairs(ss_item_join)

    # SELECT item_sk_1, item_sk_2, COUNT(*) AS cnt
    # FROM
    # (
    #    ...
    # )
    # GROUP BY item_sk_1, item_sk_2
    # -- 'frequently'
    # HAVING cnt > {q01_viewed_together_count}
    # ORDER BY cnt DESC, item_sk_1, item_sk_2

    grouped_df = (
        pair_df.groupby(["item_sk_1", "item_sk_2"])
        .size()
        .reset_index()
        .rename(columns={0: "cnt"})
    )
    grouped_df = grouped_df[grouped_df["cnt"] > q01_viewed_together_count].reset_index(
        drop=True
    )

    ### 2017 rows after filteration at sf-100
    ### should scale till sf-100k
    grouped_df = grouped_df.repartition(npartitions=1).persist()
    ## converting to strings because of issue
    # https://github.com/rapidsai/tpcx-bb/issues/36

    grouped_df["item_sk_1"] = grouped_df["item_sk_1"].astype("str")
    grouped_df["item_sk_2"] = grouped_df["item_sk_2"].astype("str")
    grouped_df = grouped_df.map_partitions(
        lambda df: df.sort_values(
            by=["cnt", "item_sk_1", "item_sk_2"], ascending=[False, True, True]
        )
    )
    grouped_df = grouped_df.reset_index(drop=True)
    ### below is just 100 rows so should fit on `cudf` context
    grouped_df = grouped_df.head(q01_limit)
    ### writing to int to ensure same values
    grouped_df["item_sk_1"] = grouped_df["item_sk_1"].astype("int32")
    grouped_df["item_sk_2"] = grouped_df["item_sk_2"].astype("int32")
    return grouped_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)
    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
