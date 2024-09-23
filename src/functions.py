# -*- coding: utf-8 -*-
"""
Non-MRIO specific functions that are used across all scripts.


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
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import country_converter as coco

# * Ensuring that file works when run as a VS
# * interactive session or executed from main folder.
try:
    __IPYTHON__  # Checks if run as interactive.
    project_folder = "../"  # Execute from main folder
except NameError:  # If not run as interactive, then
    project_folder = "./"  # Execute from repo folder
project_path = Path(project_folder)

# * Load enviroment variables and parameters
# ! ensure that file is updated to user dependent version.
load_dotenv(project_folder + "env/variables.env", override=True)
cc_use_local = bool(os.getenv("CC_USE_LOCAL"))
cc_iea_name = os.getenv("CC_IEA_NAME")

# Prepare country converter
cc = coco.CountryConverter(include_obsolete=True)

if cc_use_local:
    cc_local_path = os.getenv("CC_LOCAL_PATH")
    cc.data = pd.read_csv(
        cc_local_path,
        sep="\t",
    )


def group_IEA_flows_names(data: pd.DataFrame):
    """IEA uses indents to indicate what groups each flow
    belongs too. We combine the levels, such that all flows
    contain the name of previous groups.

    Args:
        data (pd.DataFrame): IEA data.

    Returns:
        pd.DataFrame: IEA data with flows grouped.
    """
    flows = pd.DataFrame(
        data.columns.get_level_values("Flow").unique().values,
        columns=["original"]
    )
    flows["level"] = (
        flows["original"]
        .apply(lambda x: int((len(x) - len(x.lstrip()))/3))
    )

    tmp = pd.DataFrame()
    parent_flows = []
    sep = "; "

    for index, row in flows.iterrows():
        level = row.level
        n_parents = len(parent_flows)
        if level == n_parents:
            activity = row["original"].lstrip()
            test = sep.join(parent_flows + [activity])

        elif level > n_parents:
            parent_flows.append(activity)
            activity = row["original"].lstrip()
            test = sep.join(parent_flows + [activity])

        elif level < n_parents:
            level_drop = int(n_parents - level)
            parent_flows = parent_flows[:len(parent_flows)-level_drop]
            activity = row["original"].lstrip()
            test = sep.join(parent_flows + [activity])

        tmp = pd.concat([tmp, pd.Series(test)], axis=0, ignore_index=True)

    flows["grouped"] = (
        tmp[0]
        .str.strip(".1")
        .str.replace("Heat output; ", "")
        .str.replace("Electricity output (GWh); ", "", regex=False)
        .str.replace("Total energy supply; ", "")
    )

    flow_mapper = flows.set_index(["original"])["grouped"].to_dict()

    data = data.rename(flow_mapper, axis=1, level=1)
    return data


def new_countries(df: pd.DataFrame):
    """test if new countries have been included in the IEA
    WEEB dataset and compare that with the country converter package.

    Args:
        df (pd.DataFrame): Energy data

    Returns:
        list: countries that are potentially new.
    """
    current_countries = set(cc.data[cc_iea_name])
    new_countries = set(df.index.get_level_values("Country"))

    potential = pd.DataFrame(
        list(new_countries-current_countries),
        columns=["Potential new"]
    )
    potential["Test"] = (
        potential["Potential new"]
        .apply(lambda x: cc.convert(x, to="ISO3"))
    )
    potential["IEA_current"] = (
        potential["Potential new"]
        .apply(lambda x: cc.convert(x, to=cc_iea_name))
    )
    potential_new_countries = (
        potential[potential["Test"] != "not found"]["Potential new"].to_list()
    )

    return potential_new_countries


def isnumber(x):
    """simple function to test if a
    value is a number.

    Args:
        x (any): input

    Returns:
        bool: true if a number.
    """
    try:
        float(x)
        return True
    except Exception:
        return False


def evaluate_proxy_step(
    previous_proxy,
    current_proxy,
    energy,
    description="Insert description",
    print_details=False,
    return_index=False
):
    """evaluates how much energy is allocated
    after a proxy has been applied.

    Args:
        previous_proxy (pd.DataFrame): proxy from previous step.
        current_proxy (_type_): proxy that has been applied.
        energy (pd.DataFrame): energy data.
        description (str, optional): describe the proxy step.
            Defaults to "Insert description".
        print_details (bool, optional): prints out the evaluation results.
            Defaults to False.
        return_index (bool, optional): returns the indexes of non-mapped
            energy data. Defaults to False.

    Returns:
        [None, pd.Index]: returns pd.Index if return_index is true.
    """
    # TODO: Add inline comments

    energy_filtered = energy.loc[previous_proxy.index]
    energy_total = energy_filtered.sum().sum()
    index_previous = previous_proxy[previous_proxy.sum(axis=1) != 0].index
    zero_index_previous = previous_proxy[previous_proxy.sum(axis=1) == 0].index

    index_current = current_proxy[current_proxy.sum(axis=1) != 0].index
    mapped_index = list(set(index_current)-set(index_previous))

    energy_not_mapped = energy_filtered.loc[zero_index_previous]
    energy_not_mapped_postive = (
        energy_not_mapped[energy_not_mapped > 0]
        .sum().sum()
    )
    energy_not_mapped_negative = (
        energy_not_mapped[energy_not_mapped < 0]
        .sum().sum()
    )

    positive_energy = energy_filtered.copy()
    positive_energy[positive_energy < 0] = 0

    negative_energy = energy_filtered.copy()
    negative_energy[negative_energy > 0] = 0

    positive_energy = positive_energy.loc[mapped_index]
    energy_allocated_pos = (
        np.round(positive_energy.sum(), 1)
        .values[0]
    )
    energy_share_pos = np.round((energy_allocated_pos / energy_total)*100, 3)

    negative_energy = negative_energy.loc[mapped_index]
    energy_allocated_neg = (
        np.round(negative_energy.sum(), 1)
        .values[0]
    )
    energy_share_neg = np.round((energy_allocated_neg / energy_total)*100, 3)

    energy_not_mapped_share = (
        np.round(energy_not_mapped_postive/energy_total*100, 3)
    )
    energy_neg_not_mapped_share = (
        np.round(energy_not_mapped_negative/energy_total*100, 3)
    )

    row_share = np.round(len(mapped_index) / len(previous_proxy), 5)
    print(
        f"{energy_not_mapped_postive:.2e} TJ ({energy_not_mapped_share} %)",
        f"and {energy_not_mapped_negative:.2e} TJ",
        f"({energy_neg_not_mapped_share} %) is not mapped.\n",
        "This step allocates:",
        f"{energy_allocated_pos:.2e} TJ ({energy_share_pos} %)",
        f"and {energy_allocated_neg:.2e} TJ ({energy_share_neg} %)",
        f"of positive and negative gross energy use is mapped {description}\n",
        f"This amounts to {row_share} % of the rows. \n\n"
    )

    if print_details:
        print("These are mapped as follows:")
        print(current_proxy.loc[mapped_index, :])

    if return_index:
        return mapped_index

    return None


def reclassify_missing_autoproducers_input(
        energy: pd.DataFrame,
        zero_rows: pd.Index
):
    """ 'Autoproducer' energy that is not mapped using the supply
    table, is reclassified as "Main activity producer" in the energy data.

    Args:
        energy (pd.DataFrame): energy data where energy
            should be reclassified.
        zero_rows (pd.Index): index of where autoproducers_proxies_very_long
            is equal to zero.

    Returns:
        pd.DataFrame: reclassified energy data.
    """
    # Stack energy data to quickly operate on flows.
    energy_tmp = (
        energy
        .unstack("Flow")
        .stack(level=0)
    )
    # Take out autoproducers rows that maps to nothing
    energy_tmp_zero_rows_tmp = (
        energy
        .loc[zero_rows]
        .unstack("Flow")
        .stack(level=0)
    )
    # Map them instead to main activity producer
    # TODO: use df.rename() instead of df.index = ...
    energy_tmp_zero_rows_tmp.columns = (
        energy_tmp_zero_rows_tmp
        .columns
        .str.replace("autoproducer", "main activity producer")
    )

    # Reshape to original format
    energy_tmp_zero_rows_tmp = (
        energy_tmp_zero_rows_tmp
        .unstack(level=-1)
        .stack("Flow")
        .reorder_levels(list(energy.index.names))
    )

    energy_tmp = (
        energy_tmp
        .unstack(level=-1)
        .stack("Flow")
        .reorder_levels(list(energy.index.names))
    )

    zero_rows_renamed = energy_tmp_zero_rows_tmp.index

    # Remove energy data from "Autoproducer" rows
    energy_tmp.loc[zero_rows] -= energy.loc[zero_rows]
    # Add energy data to "Main acitivy producer" rows
    energy_tmp.loc[zero_rows_renamed] += (
        energy_tmp_zero_rows_tmp.loc[zero_rows_renamed].values
    )

    return energy_tmp


def make_total_supply_proxy(
        supply_tables: pd.DataFrame,
        concordance: pd.DataFrame
):
    """creates a proxy based on total supply.

    Args:
        supply_tables (pd.DataFrame): monetary supply tables.
        concordance (pd.DataFrame): IEA flow to MRIO industry concordance.

    Returns:
        pd.DataFrame: total supply proxy.
    """
    total_supply = (
        supply_tables
        .stack()
        .groupby(["MRIO_country", "MRIO_industry"]).sum()
        .to_frame("values")
        .reset_index()
    )
    direct_mapping = (
        concordance
        .stack()
        .to_frame("concordance")
        .reset_index()
    )

    total_supply_proxy = (
        total_supply
        .merge(
            direct_mapping[direct_mapping.concordance != 0],
            on="MRIO_industry",
            how="left"
        )
        .set_index(["MRIO_country", "Flow", "MRIO_industry"])
        .unstack("MRIO_industry")
    )
    total_supply_proxy = (
        total_supply_proxy["values"]
        .mul(total_supply_proxy["concordance"])
        .dropna(axis=0, how="all")
        .fillna(0)
    )

    return total_supply_proxy


def make_product_average_proxy(
        proxy: pd.DataFrame,
        energy: pd.DataFrame
):
    """make a proxy based on how an energy
    product is otherwise being used within
    a region.

    Args:
        proxy (pd.DataFrame): proxy from previous step.
        energy (pd.DataFrame): energy data

    Returns:
        pd.DataFrame: product average proxy
    """
    product_average_proxy = (
        proxy
        .div(proxy.sum(axis=1), axis=0)
        .fillna(0)
        .mul(energy.Values, axis=0)
        .groupby(level=["MRIO_country", "IEA_product"], axis=0)
        .transform(sum)
    )

    return product_average_proxy
