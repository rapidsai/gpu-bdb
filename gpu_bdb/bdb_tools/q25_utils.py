#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from bdb_tools.readers import build_reader

q25_date = "2002-01-02"

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
    )

    ss_cols = ["ss_customer_sk", "ss_sold_date_sk", "ss_ticket_number", "ss_net_paid"]
    ws_cols = [
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_order_number",
        "ws_net_paid",
    ]
    datedim_cols = ["d_date_sk", "d_date"]

    ss_ddf = table_reader.read("store_sales", relevant_cols=ss_cols, index=False)
    ws_ddf = table_reader.read("web_sales", relevant_cols=ws_cols, index=False)
    datedim_ddf = table_reader.read("date_dim", relevant_cols=datedim_cols, index=False)

    if c:
        c.create_table("web_sales", ws_ddf, persist=False)
        c.create_table("store_sales", ss_ddf, persist=False)
        c.create_table("date_dim", datedim_ddf, persist=False)

    return ss_ddf, ws_ddf, datedim_ddf
