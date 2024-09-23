# -*- coding: utf-8 -*-
"""
The purpose of this script is to prepare the World Extended Energy Balances
for processing by removing non-numeric inputs and categorising flow names.
The raw IEA data needed for this script needs to be made from the WBIG.ivt file
using the "Beyond 20/20" browser, and saved as "wbig_tj.csv".
The table should be in TJ units and with the dimensions:
[Year, country, product] x [Flow].


EXIOBASE energy accounts procedure.
Copyright (C) 2024 Kajwan Rasul

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from functions import (
    group_IEA_flows_names,
    isnumber,
    new_countries
)

# %%
# * Ensuring that file works when run as a VS
# * interactive session or executed from main folder.
try:
    __IPYTHON__  # Checks if run as interactive.
    project_folder = "../"  # Execute from main folder
except NameError:  # If not run as interactive, then
    project_folder = "./"  # Execute from repo folder
project_path = Path(project_folder)

# %%
# * Load enviroment variables and parameters
# ! ensure that file is updated to user dependent version.
load_dotenv(project_folder + "env/variables.env", override=True)
data_path = Path(os.getenv("DATA_PATH"))
iea_version = os.getenv("IEA_VERSION")
iea_start_year = int(os.getenv("IEA_START_YEAR"))
iea_end_year = int(os.getenv("IEA_END_YEAR"))
iea_unit = os.getenv("IEA_UNIT")

# %%
# Define paths
raw_path = (
    data_path
    / "01_raw"
    / "IEA"
    / "world_energy_balances"
    / f"{iea_version}"
)

# Get net use mask to check for new IEA products and flows.
net_use_mask = pd.read_excel(
    (
        data_path
        / "00_auxiliary"
        / "IEA"
        / "masks"
        / "accounts"
        / "net_use.xlsx"
    ),
    index_col=[0],
    header=[0]
)
iea_products = set(net_use_mask.index)
iea_flows = set(net_use_mask.columns)


# %%
if __name__ == "__main__":
    # * Load data
    df_raw = pd.read_csv(
        raw_path / f"wbig_{iea_unit.lower()}.csv",
        index_col=[0, 1, 2],
        header=[0],
        encoding="cp1252",
    )
    df_raw = pd.concat([df_raw], keys=[iea_unit], axis=1)

    # * Clean it
    df_raw.index.names = ["Year", "Country", "Product"]
    df_raw.columns.names = ["Unit", "Flow"]

    df = group_IEA_flows_names(df_raw)

    # Turning columns into numeric and setting to non-numeric to zero.
    # This step takes 4m 25s...
    # TODO: Figure out a more efficient way to do this.
    df = df[df.applymap(isnumber)].replace([np.nan], 0)
    df = df.apply(pd.to_numeric)

    # Test if flows and products are the same as previous versions
    new_products = set(df.index.get_level_values("Product"))
    new_flows = set(df.columns.get_level_values("Flow"))
    assert new_products == iea_products.union({'Memo: Renewables', 'Total'})
    assert new_flows == iea_flows


    # * Save data in yearly files
    years = np.arange(iea_start_year, iea_end_year+1)
    for year in tqdm(years, desc="Years"):
        save_path = (
            data_path
            / "02_intermediate"
            / "IEA"
            / "world_energy_balances"
            / "01_clean"
            / str(year)
        )
        os.makedirs(save_path, exist_ok=True)
        df.loc[year].to_csv(
            save_path / "balance.tsv",
            sep="\t"
        )
