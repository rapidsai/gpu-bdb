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

# session timeout in secs
q30_session_timeout_inSec = 3600
# query output limit
q30_limit = 40



def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_cols = ["i_category_id", "i_item_sk"]
    item_df = table_reader.read("item", relevant_cols=item_cols)

    wcs_cols = ["wcs_user_sk", "wcs_item_sk", "wcs_click_date_sk", "wcs_click_time_sk"]
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    if c:
        c.create_table('web_clickstreams', wcs_df, persist=False)
        c.create_table('item', item_df, persist=False)

    return item_df

