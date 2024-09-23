# -*- coding: utf-8 -*-
"""
The purpose of this script is to transform the energy input data
produced in the previous script into energy use accounts.
A large set of auxilliary data and monetary supply and use tables are needed for that.
The procedure is described in detail in the paper, but the general structure is:
1) Load supply and use tables.
2) Create proxies.
3) Allocate energy to sectors using the proxies.
4) Save the data at desired level of resolution.
Note: Saving the data at full resolution requires a lof of memory.

# TODO: Delete interim variables to save memory


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
import country_converter as coco
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from mrio_functions import (
    load_exiobase_sut_tables,
)
from functions import (
    reclassify_missing_autoproducers_input,
    evaluate_proxy_step,
    make_product_average_proxy,
    make_total_supply_proxy,
)

# * IEA fuels that doesn't map properly to EXIOBASE products
# TODO: Should find a better solution for this.
renewable_iea_product = [
    "Hydro",
    "Geothermal",
    "Solar photovoltaics",
    "Solar thermal",
    "Tide, wave and ocean",
    "Wind",
    "Nuclear"
]

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
mrio_sut_path = Path(os.getenv("MRIO_SUT_PATH"))
mrio_start_year = int(os.getenv("MRIO_START_YEAR"))
mrio_end_year = int(os.getenv("MRIO_END_YEAR"))
cc_mrio_name = os.getenv("CC_MRIO_NAME")
cc_use_local = bool(os.getenv("CC_USE_LOCAL"))
cc_iea_name = os.getenv("CC_IEA_NAME")

# * Prepare country converter
cc = coco.CountryConverter(include_obsolete=True)
if cc_use_local:
    cc_local_path = os.getenv("CC_LOCAL_PATH")
    cc.data = pd.read_csv(
        cc_local_path,
        sep="\t",
    )

# Script parameters
save_data = bool(os.getenv("SAVE_DATA"))
save_iso3_level = os.getenv("SAVE_ISO3_LEVEL")
if save_iso3_level == "FALSE":
    save_iso3_level = False
else:
    save_iso3_level = True
version = os.getenv("VERSION")
map_autoproducers_to_main_activity_producer = bool(
    os.getenv(
        "MAP_AUTOPRODUCERS_TO_MAIN_ACTIVITY_PRODUCER"
    )
)
create_accounts_in_mrio_product_classification = (
    os.getenv("CREATE_ACCOUNTS_IN_MRIO_PRODUCT_CLASSIFICATION")
)
if create_accounts_in_mrio_product_classification == "FALSE":
    create_accounts_in_mrio_product_classification = False
else:
    create_accounts_in_mrio_product_classification = True

years = np.arange(mrio_start_year, mrio_end_year+1)

# %%
# * Define paths
intermediate_disaggregated_path = (
    data_path
    / "02_intermediate"
    / "extensions"
    / "energy"
    / "01_disaggregated"
)

auxiliary_path = (
    data_path
    / "00_auxiliary"
    / "IEA"
)

# %%
# * Load concordances and masks
flow_industry_concordance = pd.read_excel(
    (
        auxiliary_path
        / "concordances"
        / f"IEA_flows_MRIO_industries.xlsx"
    ),
    index_col=[0]
)
flow_industry_concordance = (
    flow_industry_concordance
    .rename_axis(["Flow"], axis=0)
    .rename_axis(["MRIO_industry"], axis=1)
)

iea_product_mrio_product_concordance = pd.read_excel(
    (
        auxiliary_path
        / "concordances"
        / f"IEA_products_MRIO_products.xlsx"
    ),
    index_col=[0]
)
iea_product_mrio_product_concordance = (
    iea_product_mrio_product_concordance
    .rename_axis(["IEA_product"], axis=0)
    .rename_axis(["MRIO_product"], axis=1)
)

autoproducers_mask = pd.read_excel(
    auxiliary_path / "masks" / "flows" / "autoproducers.xlsx",
    index_col=[0]
)
autoproducers_mask = (
    autoproducers_mask
    .rename_axis(["IEA_product"], axis=0)
    .rename_axis(["MRIO_industry"], axis=1)
    .stack()
    .to_frame("Mask")
    .reset_index()
)

losses_mask = pd.read_excel(
    auxiliary_path / "masks" / "flows" / "losses.xlsx",
    index_col=[0]
)
losses_mask = (
    losses_mask
    .rename_axis(["IEA_product"], axis=0)
    .rename_axis(["MRIO_industry"], axis=1)
)

iea_product_mrio_industry_mask = pd.read_excel(
    (
        auxiliary_path
        / "masks"
        / "filters"
        / f"IEA_products_MRIO_industries.xlsx"
    ),
    index_col=[0]
)
iea_product_mrio_industry_mask = (
    iea_product_mrio_industry_mask
    .rename_axis(["IEA_product"], axis=0)
    .rename_axis(["MRIO_industry"], axis=1)
)

net_use_mask = pd.read_excel(
    auxiliary_path / "masks" / "accounts" / "net_use.xlsx",
    index_col=[0]
)
net_use_mask = (
    net_use_mask
    .stack()
    .rename_axis(["IEA_product", "Flow"], axis=0)
    .to_frame("Mask")
    .reset_index()
)

emission_relevant_mask = pd.read_excel(
    auxiliary_path / "masks" / "accounts" / "emission_relevant.xlsx",
    index_col=[0]
)
emission_relevant_mask = (
    emission_relevant_mask
    .stack()
    .rename_axis(["IEA_product", "Flow"], axis=0)
    .to_frame("Mask")
    .reset_index()
)

# %%
for year in tqdm(years):
    energy_raw = pd.read_csv(
        intermediate_disaggregated_path / str(year) / "energy.tsv",
        sep="\t",
        index_col=[0, 1, 2, 3],
        header=[0]
    )

    # Flows we don't map to any industry are not calculated as input.
    non_input_flows = list(
        flow_industry_concordance.sum(axis=1).loc[
            flow_industry_concordance.sum(axis=1) == 0
        ].index.values
    )
    # Extract input
    energy_input = (
        energy_raw.drop(non_input_flows, axis=0, level="Flow")
    )

    # Set exports to zero to avoid double counting
    energy_input.loc[
        (
            energy_input
            .index
            .get_level_values("Flow")
            .str.contains("(Exports)|(International .* bunker)", regex=True)
        ), :
    ] = 0

    (
        use_tables,
        supply_tables,
        mrio_country_order,
        mrio_industry_order,
        mrio_final_demand_order
    ) = (
        load_exiobase_sut_tables(
            path=mrio_sut_path,
            data_path=data_path,
            year=year
        )
    )

    # * Defining various classifications for easier filtering.
    # TODO: add comments
    proxy_index_names = list(energy_input.index.names)
    index_mapping = (
        energy_input
        .reset_index()
        .loc[:, proxy_index_names]
    )
    index_mapping_excl_iso3 = (
        index_mapping
        .drop("ISO3_country", axis=1)
        .drop_duplicates()
    )
    proxy_index_names_excl_iso3 = list(index_mapping_excl_iso3.columns)

    iea_flows = energy_input.index.get_level_values("Flow").unique()
    autoproducers_flows = iea_flows[iea_flows.str.contains("autoproducer")]
    losses_flows = iea_flows[iea_flows.str.contains("Losses")]
    main_activity_producer_flows = (
        iea_flows[iea_flows.str.contains("main activity producer")]
    )
    proxy_flows = list(
        set(index_mapping_excl_iso3.Flow)
        - set(list(autoproducers_flows) + list(losses_flows))
    )

    # %%
    # * Autoproducers proxy
    # Start from the supply tables and map to IEA product classification
    autoproducers_proxy = (
        iea_product_mrio_product_concordance
        .dot(supply_tables.unstack(level="MRIO_country"))
        .stack(level=["MRIO_country", "MRIO_industry"])
        .reorder_levels(["MRIO_country", "MRIO_industry", "IEA_product"])
        .sort_index()
        .to_frame("Values")
    )

    # %%
    # Add masks and multiply
    autoproducers_proxy = (
        autoproducers_proxy
        .reset_index()
        .merge(
            autoproducers_mask,
            on=["MRIO_industry", "IEA_product"],
            how="outer"
        ).set_index(["MRIO_country", "MRIO_industry", "IEA_product"])
    )

    autoproducers_proxy["Masked values"] = (
        autoproducers_proxy["Values"]
        .mul(autoproducers_proxy["Mask"])
    )

    # Select and re-pivot
    autoproducers_proxy = autoproducers_proxy.loc[:, "Masked values"]
    autoproducers_proxy = autoproducers_proxy.unstack("MRIO_industry")
    autoproducers_proxy = (
        autoproducers_proxy
        .div(autoproducers_proxy.sum(axis=1), axis=0)
        .fillna(0)
    )

    autoproducers_proxy = (
        pd.concat(
            [autoproducers_proxy]*len(autoproducers_flows),
            keys=autoproducers_flows.values,
            names=["Flow"],
            axis=0
        )
        .reorder_levels(proxy_index_names_excl_iso3)
    )

    # %%
    # Reclassify non-mapped autoproducers flows to "main activity producer"
    zero_rows = autoproducers_proxy[autoproducers_proxy.sum(axis=1) == 0].index

    zero_rows_ISO3 = (
        zero_rows
        .to_frame(index=False)
        .merge(
            (
                index_mapping
                .loc[:, ["MRIO_country", "ISO3_country"]]
                .drop_duplicates()
            ),
            on="MRIO_country",
            how="outer"
        )
        .set_index(proxy_index_names)
        .index
    )

    if map_autoproducers_to_main_activity_producer:
        energy_input = (
            reclassify_missing_autoproducers_input(
                energy_input,
                zero_rows_ISO3
            )
        )
    else:
        autoproducers_proxy.loc[zero_rows, "Production of electricity nec"] = 1

    # %%
    # * Losses proxy
    # We map losses directly using the mask instead of using the
    # monetary use tables with the losses.
    # ? Not sure if this should stay a permanent solution.
    losses_extension = (
        index_mapping_excl_iso3[
            index_mapping_excl_iso3.Flow.isin(losses_flows)
        ]
    )

    losses_proxy = (
        losses_mask
        .merge(
            losses_extension,
            on="IEA_product",
            how="outer"
        )
        .set_index(proxy_index_names_excl_iso3)
    )

    # %%
    # * Mapping renewable fuels directly to EXIOBASE sector
    # Reason for this is that IEA renewables are natural energy flows,
    # not products produced by the economy. Otherwise "Geothermal"
    # would be mapped as the output of "Electricity by geothermal"

    renewable_proxy = (
        iea_product_mrio_industry_mask.loc[renewable_iea_product, :]
    )
    renewable_proxy = (
        renewable_proxy
        .reset_index()
        .merge(
            index_mapping_excl_iso3,
            on="IEA_product",
            how="outer"
        )
        .set_index(proxy_index_names_excl_iso3)
        .dropna()
        .loc[pd.IndexSlice[:, main_activity_producer_flows, :], :]
    )

    # %%
    # * Standard proxy
    # Preparing masked flow industry concordance
    masked_flow_industry_concordance = (
        flow_industry_concordance
        .stack()
        .to_frame("concordance")
        .reset_index()
        .merge(
            (
                iea_product_mrio_industry_mask
                .stack()
                .to_frame("mask")
                .reset_index()
            ),
            on="MRIO_industry",
            how="inner"
        )
        .set_index(["Flow", "IEA_product", "MRIO_industry"])
        .unstack(["MRIO_industry"])
    )

    masked_flow_industry_concordance = (
        masked_flow_industry_concordance["concordance"]
        .mul(masked_flow_industry_concordance["mask"])
    )

    masked_flow_industry_concordance = (
        pd.concat(
            [masked_flow_industry_concordance]*len(mrio_country_order),
            keys=mrio_country_order,
            names=["MRIO_country"]
        )
    )

    # %%
    # Creating first proxy based on the use table and the masked concordance
    # We exclude iso3 for now as it is redudant in this step
    proxy_initial = (
        iea_product_mrio_product_concordance
        .dot(use_tables.unstack(level="MRIO_country"))
        .stack(level="MRIO_country")
        .reset_index()
        .merge(
            index_mapping_excl_iso3,
            on=["MRIO_country", "IEA_product"],
            how="outer"
        )
        .set_index(proxy_index_names_excl_iso3)
    )

    standard_proxy = (
        masked_flow_industry_concordance
        .mul(proxy_initial)
    )
    standard_proxy = (
        standard_proxy.drop(non_input_flows, axis=0, level="Flow")
    )

    # Aggregate energy to MRIO region resolution to evaluate
    # the the allocation steps.
    energy_aggregated = (
        energy_input
        .groupby(by=proxy_index_names_excl_iso3)
        .sum()
    )

    # %%
    # Applying the new extensions
    standard_proxy_one = standard_proxy.copy()
    standard_proxy_one.loc[losses_proxy.index, :] = losses_proxy
    standard_proxy_one.loc[autoproducers_proxy.index, :] = autoproducers_proxy
    standard_proxy_one.loc[renewable_proxy.index, :] = renewable_proxy
    evaluate_proxy_step(
        previous_proxy=standard_proxy,
        current_proxy=standard_proxy_one,
        energy=energy_aggregated,
        description="using the monetary proxy (Step 1)."
    )

    # %%
    # Creating total (monetary use on energy products) proxy
    proxy_total = (
        iea_product_mrio_product_concordance
        .div(iea_product_mrio_product_concordance.sum(axis=0), axis=1)
        .replace([np.nan, np.inf, -np.inf], 0)
        .dot(use_tables.unstack(level="MRIO_country"))
        .stack(level="MRIO_country")
        .groupby(level=["MRIO_country"]).sum()
        .reset_index()
        .merge(
            index_mapping_excl_iso3,
            on="MRIO_country",
            how="outer"
        )
        .set_index(proxy_index_names_excl_iso3)
    )

    masked_proxy_total = (
        masked_flow_industry_concordance.mul(
            proxy_total
        ).fillna(0)
    )
    standard_proxy_two = (
        standard_proxy_one[standard_proxy_one.sum(axis=1) != 0]
        .combine_first(
            masked_proxy_total.loc[standard_proxy_one.index, :]
        )
    )
    evaluate_proxy_step(
        previous_proxy=standard_proxy_one,
        current_proxy=standard_proxy_two,
        energy=energy_aggregated,
        description=(
            "using the total monetary proxy (Step 2)."
        )
    )

    # Using the average use of energy product within a region
    # as the proxy.
    product_average_proxy = (
        make_product_average_proxy(standard_proxy_two, energy_aggregated)
    )

    standard_proxy_three = (
        standard_proxy_two[standard_proxy_two.sum(axis=1) != 0]
        .combine_first(
            product_average_proxy
        )
    )
    evaluate_proxy_step(
        previous_proxy=standard_proxy_two,
        current_proxy=standard_proxy_three,
        energy=energy_aggregated,
        description=(
            "using the average regional product use (Step 3)."
        )
    )

    # * Outliers
    # Some renewable energy products are in some country not
    # produced by a "Main activity producer" and therefore they will not be
    # allocated in any of the previous steps.
    # This is the case for "Solar thermal", "Geothermal",
    # and "Solar photovoltaics".
    # For these rows, we map the corresponding flow directly to MRIO industries
    # using the flow industry concordance.
    # In case of 1-to-many mappings from flow to MRIO industry,
    # they are weighted by their total output.
    outliers = (
        standard_proxy_three[
            (standard_proxy_three.sum(axis=1) == 0)
            & (energy_aggregated.Values != 0)
        ]
        .index
        .to_frame(index=False)
    )

    non_renewable_outliers = (
        set(outliers.IEA_product) - set(renewable_iea_product)
    )
    if len(non_renewable_outliers) > 0:
        print(
            "Unexpected outlier energy products:",
            non_renewable_outliers
        )
        print(
            energy_aggregated[
                (standard_proxy_three.sum(axis=1) == 0)
                & (energy_aggregated.Values != 0)
            ].loc[pd.IndexSlice[:, :, list(non_renewable_outliers)]]
        )

    total_supply_proxy = make_total_supply_proxy(
        supply_tables,
        flow_industry_concordance
    )

    outliers_proxy = (
        outliers
        .merge(
            total_supply_proxy.reset_index(),
            on=["MRIO_country", "Flow"],
            how="left"
        )
        .set_index(["MRIO_country", "Flow", "IEA_product"])
    )

    combined_proxy = (
        outliers_proxy
        .combine_first(standard_proxy_three)
    )

    evaluate_proxy_step(
        previous_proxy=standard_proxy_three,
        current_proxy=combined_proxy,
        energy=energy_aggregated,
        description=(
            "after handling outliers."
        )
    )

    # %%
    proxy_disaggregated = (
        combined_proxy
        .reset_index()
        .merge(
            (
                index_mapping
                .loc[:, ["MRIO_country", "ISO3_country"]]
                .drop_duplicates()
            ),
            on="MRIO_country",
            how="outer"
        )
        .set_index(proxy_index_names)
    )
    proxy_normalised = (
        abs(proxy_disaggregated)
        .div(abs(proxy_disaggregated).sum(axis=1), axis=0)
        .replace([np.nan, np.inf, -np.inf], 0)
    )


    energy_input_vector = energy_input.loc[:, "Values"]
    energy_input_vector.name = None

    energy_use = (
        proxy_normalised
        .mul(energy_input_vector, axis=0)
    )

    energy_use = energy_use.loc[energy_use.sum(axis=1) != 0, :]
    energy_use.columns.names = ["MRIO_industry"]

    # %%
    if save_iso3_level:
        base_level = ["ISO3_country"]
    else:
        base_level = []

    net_energy_use = (
        energy_use
        .stack()
        .to_frame("Values")
        .reset_index()
        .merge(
            net_use_mask,
            on=["IEA_product", "Flow"],
            how="left"
        )
        .set_index(proxy_index_names)
        .set_index(["MRIO_industry"], append=True)
    )

    net_energy_use["Masked values"] = (
        net_energy_use["Values"].mul(net_energy_use["Mask"])
    )

    net_energy_use = (
        net_energy_use
        .loc[:, "Masked values"]
        .unstack(["MRIO_country", "MRIO_industry"])
        .fillna(0)
        .groupby(level=(base_level + ["Flow", "IEA_product"]), axis=0)
        .sum()
    )

    net_energy_use = (
        net_energy_use[
            net_energy_use.sum(axis=1) != 0
        ]
    )

    # %%
    emission_relevant_energy = (
        energy_use
        .stack()
        .to_frame("Values")
        .reset_index()
        .merge(
            emission_relevant_mask,
            on=["IEA_product", "Flow"],
            how="left"
        )
        .set_index(proxy_index_names)
        .set_index(["MRIO_industry"], append=True)
    )

    emission_relevant_energy["Masked values"] = (
        emission_relevant_energy["Values"]
        .mul(emission_relevant_energy["Mask"])
    )

    emission_relevant_energy = (
        emission_relevant_energy
        .loc[:, "Masked values"]
        .unstack(["MRIO_industry"])
        .fillna(0)
    )

    emission_relevant_energy = (
        emission_relevant_energy[
            emission_relevant_energy.sum(axis=1) != 0
        ]
    )

    # %%
    energy_use = (
        energy_use
        .unstack("MRIO_country")
        .swaplevel("MRIO_country", "MRIO_industry", axis=1)
        .fillna(0)
        .groupby(level=(base_level+["Flow", "IEA_product"]), axis=0)
        .sum()
    )

    energy_use = energy_use[energy_use.sum(axis=1) != 0]

    emission_relevant_energy = (
        emission_relevant_energy
        .unstack("MRIO_country")
        .swaplevel("MRIO_country", "MRIO_industry", axis=1)
        .fillna(0)
        .groupby(level=(base_level+["Flow", "IEA_product"]), axis=0)
        .sum()
    )

    # %%
    # Save dataframes
    save_path = (
        data_path
        / "03_processed"
        / "extensions"
        / "ixi"
        / "stressors"
        / "energy"
        / f"{version}"
        / f"{year}"
    )
    os.makedirs(save_path, exist_ok=True)

    if save_data:
        print(f"Saving data (Year {year})")
        industry_order_slice = pd.IndexSlice[
            mrio_country_order,
            mrio_industry_order
        ]
        final_demand_slice = pd.IndexSlice[
            mrio_country_order,
            mrio_final_demand_order
        ]

        energy_use.loc[:, industry_order_slice].to_csv(
            save_path / f"gross_energy_use.tsv",
            sep="\t"
        )
        net_energy_use.loc[:, industry_order_slice].to_csv(
            save_path / f"net_energy_use.tsv",
            sep="\t"
        )
        emission_relevant_energy.loc[:, industry_order_slice].to_csv(
            save_path / f"emission_relevant_energy.tsv",
            sep="\t"
        )

        energy_use.loc[:, final_demand_slice].to_csv(
            save_path / "gross_energy_use_Y.tsv",
            sep="\t"
        )
        net_energy_use.loc[:, final_demand_slice].to_csv(
            save_path / "net_energy_use_Y.tsv",
            sep="\t"
        )
        emission_relevant_energy.loc[:, final_demand_slice].to_csv(
            save_path / "emission_relevant_energy_Y.tsv",
            sep="\t"
        )

    else:
        pass
