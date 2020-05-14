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

from abc import ABC, abstractmethod


TABLE_NAMES = [
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "item_marketprices",
    "parquet",
    "product_reviews",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_clickstreams",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
]

### these tables with non string columns should easily fit on 1 gpu
## At sf-100k, product_reviews=200M , customer=31M,
## customer_address=15M, item=5M, item_market_prices=28M
## in most queries apart from nlp ones we dont read string columns
## so these should scale
## see https://github.com/rapidsai/tpcx-bb/issues/66 for size details

SMALL_TABLES = ["customer", "customer_address", "item", "item_marketprices"]
### these tables are not expected to grow with scale factors
## these should fit easily with all columns on a single gpu
## see https://github.com/rapidsai/tpcx-bb/issues/66 for size details

SUPER_SMALL_TABLES = [
    "date_dim",
    "time_dim",
    "web_site",
    "income_band",
    "ship_mode",
    "household_demographics",
    "promotion",
    "web_page",
    "warehouse",
    "reason",
    "store",
]


class Reader(ABC):
    """Base class for TPCx-BB File Readers"""

    @abstractmethod
    def read(self, filepath, **kwargs):
        """"""

    @abstractmethod
    def show_tables(self):
        """"""


class ParquetReader(Reader):
    """Read TPCx-BB Parquet data"""

    def __init__(
        self, basepath, split_row_groups=False,
    ):
        self.table_path_mapping = {
            table: f"{basepath}{table}/*.parquet" for table in TABLE_NAMES
        }
        self.split_row_groups = split_row_groups

    def show_tables(self):
        return self.table_path_mapping.keys()

    def read(self, table, relevant_cols=None, **kwargs):
        import dask_cudf

        filepath = self.table_path_mapping[table]
        # we ignore split_row_groups if gather_statistics=False
        if self.split_row_groups:

            df = dask_cudf.read_parquet(
                filepath,
                columns=relevant_cols,
                split_row_groups=self.split_row_groups,
                gather_statistics=True,
                **kwargs,
            )
        else:
            df = dask_cudf.read_parquet(
                filepath,
                columns=relevant_cols,
                split_row_groups=self.split_row_groups,
                gather_statistics=False,
                **kwargs,
            )

        ## Repartition small tables to a single partition to prevent
        ## distributed merges when possible
        ## Only matters when partition size<3GB

        if (table in SMALL_TABLES) or (table in SUPER_SMALL_TABLES):
            df = df.repartition(npartitions=1)
        return df


class ORCReader(Reader):
    """Read TPCx-BB ORC data"""

    # TODO
    def __init__(self, basepath):
        pass


class CSVReader(Reader):
    """Read TPCx-BB CSV data"""

    # TODO
    def __init__(self, basepath):
        pass


def build_reader(basepath, data_format="parquet", **kwargs):
    assert data_format in ("csv", "parquet", "orc")

    if data_format in ("csv",):
        return CSVReader(basepath=basepath, **kwargs)

    elif data_format in ("parquet",):
        return ParquetReader(basepath=basepath, **kwargs)

    elif data_format in ("orc",):
        return ORCReader(basepath=basepath, **kwargs)
