# -*- coding: utf-8 -*-
"""
The purpose of this script is to transform the WEEB from IEA
into energy accounts. The procedure is generic and have been
in the litterature, see fore example:
    - "Supporting Information for energy accounts" in Stadler et. al (2018)
    - UN Statisticvs Compilers' Manual by UN Department of Economic and Social Affairs
In this step we are only interested in the energy inputs to the IEA flows
as we want to create "energy use" accounts.
For "energy supply" accounts, the procedure should be changed.
Note: This script does not transform the data from territorial to residential princinple.
        That is instead done in 03_disaggregate_and_adjust.py after the data has been disaggregated.
        All bunkers (aviation & marine) fuel use is given at an international level.


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
import country_converter as coco
from dotenv import load_dotenv

# * Prepare country converter
cc = coco.CountryConverter(include_obsolete=True)

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
iea_start_year = int(os.getenv("IEA_START_YEAR"))
iea_end_year = int(os.getenv("IEA_END_YEAR"))
iea_unit = os.getenv("IEA_UNIT")
cc_use_local = bool(os.getenv("CC_USE_LOCAL"))
cc_iea_name = os.getenv("CC_IEA_NAME")

years = np.arange(iea_start_year, iea_end_year+1)

if cc_use_local:
    cc_local_path = os.getenv("CC_LOCAL_PATH")
    cc.data = pd.read_csv(
        cc_local_path,
        sep="\t",
    )

# %%
# * Define paths
intermediate_clean__path = (
    data_path
    / "02_intermediate"
    / "IEA"
    / "world_energy_balances"
    / "01_clean"
)

# %%
# * Run script
if __name__ == "__main__":
    for year in tqdm(years, desc="Years"):
        df_raw = pd.read_csv(
            intermediate_clean__path / str(year) / "balance.tsv",
            index_col=[0, 1],
            header=[0, 1],
            sep="\t"
        )

        df = df_raw.loc[:, iea_unit]

        # * Extract bunkers
        # Extract world bunkers related to consumption
        # These will be allocated using adjustment factors.
        # ? Should we include more bunker fuels?
        products = df.index.get_level_values("Product")
        flows = df.columns.get_level_values("Flow")
        bunkers_products = (
            products.str.contains("Aviation")
            | products.str.contains("Kerosene", case=False)
            | products.str.contains("Fuel oil")
            | products.str.contains("Gas/diesel oil excl. biofuels")
        )
        bunkers_flows = (
            flows.str.contains("World .* bunkers", regex=True)
        )

        bunkers = df.loc[
            pd.IndexSlice["World", bunkers_products],
            pd.IndexSlice[bunkers_flows]
        ]

        bunkers = (
            bunkers
            .stack()
            .reorder_levels(["Country", "Flow", "Product"])
            .to_frame("Values")
        )

        # Getting IEA region list from country converter
        IEA_regions = (
            cc.data
            .loc[:, cc_iea_name]
            .dropna()
            .unique()
        )

        # Filter regions that are of interest.
        df = df.loc[pd.IndexSlice[IEA_regions, :], :]

        # Set consumption bunkers to zero.
        # This should be zero by default.
        df.loc[
            :,
            bunkers_flows
        ] = 0

        # Dropping
        # - total and memo from rows
        IEA_products_filter = ~(
            df.index.get_level_values("Product").str.contains("Memo:")
            | df.index.get_level_values("Product").str.contains("Total")
        )

        df = df.loc[pd.IndexSlice[:, IEA_products_filter], :]

        # * Put data in format needed for IO type accounting
        # Final consumption, electricity & heat, and imports are kept as is.
        # However, the latter two are only used for the MSUT input.
        df_standard = df.loc[
            :,
            (
                df.columns.str.contains("Total final consumption")
                | df.columns.str.contains("Electricity output")
                | df.columns.str.contains("Heat output")
                | df.columns.str.contains("Imports", case=False)
            )
        ]

        # Only negative transfers and transformations are considered.
        # Sign is then switched.
        df_trans = df.loc[
            :,
            (
                df.columns.str.contains("Transfers", case=False)
                | df.columns.str.contains("Transformation processes", case=False)
            )
        ].copy()
        df_trans[df_trans > 0] = 0
        df_trans = df_trans * -1

        # Sign is switched on all other flows
        # Note: some of these are aggregate flows that will be dropped later.
        df_other = df.loc[
            :,
            ~(
                df.columns.str.contains("Total final consumption")
                | df.columns.str.contains("Transfers", case=False)
                | df.columns.str.contains("Transformation processes", case=False)
                | df.columns.str.contains("Imports", case=False)
                | df.columns.str.contains("Electricity output")
                | df.columns.str.contains("Heat output")
                # df.columns.str.contains("Exports", case=False)
                # | df.columns.str.contains("Stock changes", case=False)
                # | df.columns.str.contains("Statistical differences", case=False)
                # | df.columns.str.contains("own use", case=False)
                # | df.columns.str.contains("losses", case=False)
            )
        ].copy()
        df_other = df_other * -1

        # Recombine dataframes and put in long format
        energy = (
            pd.concat(
                [df_standard, df_trans, df_other],
                axis=1
            ).stack()
            .reorder_levels(["Country", "Flow", "Product"])
            .to_frame("Values")
        )
        energy = pd.concat([energy, bunkers], axis=0)

        save_path = (
            data_path
            / "02_intermediate"
            / "IEA"
            / "world_energy_balances"
            / "02_accounting_format"
            / str(year)
        )
        os.makedirs(save_path, exist_ok=True)

        energy.to_csv(
            save_path / "energy.tsv",
            sep="\t"
        )
