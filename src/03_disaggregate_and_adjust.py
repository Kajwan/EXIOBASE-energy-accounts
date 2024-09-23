# -*- coding: utf-8 -*-
"""
This script have two purposes.
First the energy data is regionally disaggregrated
to UN region level using GDP data (from the UN SNAAMA).
Then the data is transformed from the territorial to residential principle
using allocation and adjustment factors from transport models.


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
cc_mrio_name = os.getenv("CC_MRIO_NAME")
mrio_start_year = int(os.getenv("MRIO_START_YEAR"))
mrio_end_year = int(os.getenv("MRIO_END_YEAR"))
cc_use_local = bool(os.getenv("CC_USE_LOCAL"))
cc_iea_name = os.getenv("CC_IEA_NAME")

years = np.arange(mrio_start_year, mrio_end_year+1)

# * Prepare country converter
cc = coco.CountryConverter(include_obsolete=True)
if cc_use_local:
    cc_local_path = os.getenv("CC_LOCAL_PATH")
    cc.data = pd.read_csv(
        cc_local_path,
        sep="\t",
    )

# %%
# * Define paths
intermediate_accounting_path = (
    data_path
    / "02_intermediate"
    / "IEA"
    / "world_energy_balances"
    / "02_accounting_format"
)

gdp_path = (
    data_path
    / "00_auxiliary"
    / "GDP"
)

adjustment_factors_path = (
    data_path
    / "00_auxiliary"
    / "IEA"
    / "adjustment_factors"
)

# %%
# * Load GDP data
gdp_full = pd.read_excel(
    gdp_path / "GDP.xlsx",
    index_col=[0]
)

# * Load allocation (marine & aviation) / adjustment (road transport) factors
aviation_factors_full = (
    pd.read_excel(
        adjustment_factors_path / "aviation.xlsx",
        index_col=[0, 1]
    )
    .rename_axis(["ISO3_country"], axis=1)
    .stack()
    .to_frame("Factor")
)
marine_factors_full = (
    pd.read_excel(
        adjustment_factors_path / "marine.xlsx",
        index_col=[0, 1]
    )
    .rename_axis(["ISO3_country"], axis=1)
    .stack()
    .to_frame("Factor")
)
road_transport_factors_full = (
    pd.read_excel(
        adjustment_factors_path / "road_transport.xlsx",
        index_col=[0, 1]
    )
    .rename_axis(["ISO3_country"], axis=1)
    .stack()
    .to_frame("Factor")
)

# %%
for year in tqdm(years):
    # TODO: Make it optional whether to include bunkers and road adjustments
    # * Read in data
    energy = pd.read_csv(
        intermediate_accounting_path / str(year) / "energy.tsv",
        sep="\t",
        index_col=[0, 1, 2],
        header=[0]
    )

    # Prepare energy data
    energy = energy.reset_index()
    energy = energy.rename(
        {
            "Country": "IEA_country",
            "Product": "IEA_product"
        },
        axis=1
    )
    bunkers = energy[energy["IEA_country"] == "World"]
    energy = energy[~(energy["IEA_country"] == "World")]

    # Select gdp data for year
    gdp = (
        gdp_full[gdp_full.Year == year]
        .drop("Year", axis=1)
        .reset_index(drop=True)
    )

    # Calculate GDP totals and shares
    gdp["IEA_country"] = cc.convert(
        gdp["Country"],
        to=cc_iea_name,
        not_found=None
    )

    gdp["MRIO_country"] = cc.convert(
        gdp["Country"],
        to=cc_mrio_name,
        not_found=None
    )

    gdp["GDP_IEA_region_total"] = (
        gdp
        .groupby("IEA_country")["GDP"]
        .transform(sum)
    )

    gdp["GDP_IEA_region_share"] = gdp["GDP"].div(gdp["GDP_IEA_region_total"])

    # Dropping countries not included in IEA, e.g., Andorra,
    # but might be in MRIO.
    # Note: These regions will have no energy extensions in the MRIO.
    print("Regions not in IEA data:")
    print(gdp[gdp["IEA_country"].isna()]["Country"].values)
    gdp = gdp.dropna()

    gdp["GDP_MRIO_region_total"] = (
        gdp
        .groupby("MRIO_country")["GDP"]
        .transform(sum)
    )

    gdp["GDP_MRIO_region_share"] = (
        gdp["GDP"]
        .div(gdp["GDP_MRIO_region_total"])
    )

    # Merging the energy and GDP data
    energy = (
        energy.merge(
            gdp,
            right_on="IEA_country",
            left_on="IEA_country",
            how="outer"
        )
    )

    # ! GDP: Zanzibar not included in Tanzania
    non_GDP_regions = (
        energy[energy["GDP_IEA_region_share"].isna()]
        .loc[:, "IEA_country"]
        .unique()
    )
    print("IEA data we don't have GDP data on:")
    print(non_GDP_regions)

    energy_tmp = energy.dropna().copy()

    energy_tmp.loc[:, "Values"] = (
        energy_tmp["GDP_IEA_region_share"]
        .mul(energy_tmp["Values"])
    )

    energy_tmp = (
        energy_tmp
        .set_index(
            ["MRIO_country", "ISO3_country", "Flow", "IEA_product"]
        ).loc[:, ["Values"]]
    )

    # * Aviation & marine bunker fuels
    # Disaggregating world bunkers use
    # Aviation
    try:
        aviation_factors = (
            aviation_factors_full
            .loc[year]
            .reset_index()
        )
    except KeyError as e:
        aviation_factors_year = max(
            aviation_factors_full
            .index.get_level_values("Year")
        )
        aviation_factors = (
            aviation_factors_full.loc[
                aviation_factors_year
            ].reset_index()
        )
        print(
            f"No aviation allocation factors exist for year: {year}\n",
            f"Instead using values for year: {aviation_factors_year}"
        )

    try:
        marine_factors = (
            marine_factors_full
            .loc[year]
            .reset_index()
        )
    except KeyError as e:
        marine_factors_year = max(
            marine_factors_full
            .index.get_level_values("Year")
        )
        marine_factors = (
            marine_factors_full.loc[
                marine_factors_year
            ].reset_index()
        )
        print(
            f"No marine allocation factors exist for year: {year}\n",
            f"Instead using values for year: {marine_factors_year}"
        )

    bunkers_tmp = (
        bunkers
        .merge(
            pd.concat(
                [aviation_factors, marine_factors],
                axis=0
            ),
            on=["IEA_product"],
            how="outer"
        )
    )

    # Allocate bunker fuel use
    bunkers_tmp["Values (allocated)"] = (
        bunkers_tmp["Values"]
        .mul(bunkers_tmp["Factor"])
    )
    bunkers_tmp["MRIO_country"] = (
        bunkers_tmp["ISO3_country"]
        .apply(lambda x: cc.convert(x, src="ISO3", to=cc_mrio_name))
    )

    # Remove countries not presented in energy data
    # 20/06/23 - ISO3 codes: ['FRO', 'AND', 'LIE', 'GUF', 'SMR', 'MCO']
    # This losses a tiny fraction of the global bunker fuel use (<1e-4).
    bunkers_tmp = bunkers_tmp[
        bunkers_tmp["ISO3_country"].isin(
            list(
                energy_tmp
                .index
                .get_level_values("ISO3_country")
                .unique()
            )
        )
    ]

    bunkers_tmp = (
        bunkers_tmp
        .pivot(
            index=["MRIO_country", "ISO3_country"],
            columns=["Flow", "IEA_product"],
            values="Values (allocated)"
        )
        .fillna(0)
        .stack(["Flow", "IEA_product"])
        .to_frame("Values")
    )

    # * Adjust road transport
    # Note: these are adjustment factors instead of allocation factors.
    # Therefore they are handled differently.
    try:
        road_transport_factors = (
            road_transport_factors_full
            .loc[year]
            .reset_index()
        )
    except KeyError as e:
        road_transport_factors_year = max(
            road_transport_factors_full
            .index.get_level_values("Year")
        )
        road_transport_factors = (
            road_transport_factors_full.loc[
                road_transport_factors_year
            ]
        ).reset_index()
        print(
            f"No road transport adjustment values exist for year: {year}\n",
            f"Instead using values for year: {road_transport_factors_year}"
        )

    road_transport_factors["MRIO_country"] = (
        road_transport_factors["ISO3_country"]
        .apply(lambda x: cc.convert(x, src="ISO3", to=cc_mrio_name))
    )

    energy_RT = (
        road_transport_factors
        .merge(
            energy_tmp.reset_index(),
            how="left",
            on=["MRIO_country", "ISO3_country", "IEA_product"]
        )
    )

    # Handle negative and positive values separately
    energy_RT_tmp = (
        pd.concat(
            [
                energy_RT[energy_RT.Values >= 0],
                energy_RT[energy_RT.Values < 0]
            ],
            axis=0,
            keys=["Positive", "Negative"],
            names=["Sign"]
        ).droplevel(1, axis=0)  # Removing dummy index
        .reset_index()
    )

    # Get totals to rescale back correctly
    energy_RT_tmp["Values (total)"] = (
        energy_RT_tmp
        .groupby(["Sign", "Flow", "IEA_product"])
        ["Values"]
        .transform(sum)
    )

    energy_RT_tmp["Adjustment"] = (
        energy_RT_tmp["Values"].mul(energy_RT_tmp["Factor"])
    )

    energy_RT_tmp["Values (adjusted)"] = (
        energy_RT_tmp["Values"] + energy_RT_tmp["Adjustment"]
    )

    energy_RT_tmp["Values (adjusted total)"] = (
        energy_RT_tmp
        .groupby(["Sign", "IEA_product", "Flow"])
        ["Values (adjusted)"]
        .transform(sum)
    )

    energy_RT_tmp["Values (normalised)"] = (
        energy_RT_tmp["Values (adjusted)"]
        .div(
            energy_RT_tmp["Values (adjusted total)"]
        ).fillna(0)
    )

    energy_RT_tmp["Values (corrected)"] = (
        energy_RT_tmp["Values (normalised)"]
        .mul(energy_RT_tmp["Values (total)"])
    )

    energy_RT_tmp = (
        energy_RT_tmp
        .set_index(["MRIO_country", "ISO3_country", "Flow", "IEA_product"])
        .loc[:, ["Values (corrected)"]]
        .rename({"Values (corrected)": "Values"}, axis=1)
    )

    # Combining all the energy data and saving results
    energy = bunkers_tmp.combine_first(energy_tmp).fillna(0)

    energy = energy_RT_tmp.combine_first(energy).fillna(0)

    # ? Is the GDP dataframe worth saving for future use?
    save_path = (
        data_path
        / "02_intermediate"
        / "extensions"
        / "energy"
        / "01_disaggregated"
        / str(year)
    )
    os.makedirs(save_path, exist_ok=True)

    energy.to_csv(
        save_path / "energy.tsv",
        sep="\t"
    )
