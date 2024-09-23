# -*- coding: utf-8 -*-
"""
MRIO specific functions.


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
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_exiobase_classification(path: Path):
    """loads exiobase classifications file to easily convert
    between codes and names.

    Args:
        path (Path): path to folder with mappings.

    Returns:
        dict: dictionary with the different mappings.
    """
    # * Various mapping between MRIO classifications
    final_demand = pd.read_excel(
        path / "concordances" / "product_industry.xlsx",
        index_col=None,
        header=[0, 1, 2, 3],
        sheet_name="FinalDemand"
    )

    product_industry_concordance = pd.read_excel(
        path / "concordances" / "product_industry.xlsx",
        index_col=[0, 1, 2, 3],
        header=[0, 1, 2, 3],
        sheet_name="ProIndCon"
    )

    country_list = pd.read_excel(
        path / "countries" / "country_list.xlsx",
        sheet_name="Regions"
    )

    final_demand_names = final_demand.columns.get_level_values(1)
    final_demand_codes = final_demand.columns.get_level_values(2)

    final_demand_map = dict(zip(
        final_demand_names,
        final_demand_codes
    ))

    final_demand_map_reverse = dict(zip(
        final_demand_codes,
        final_demand_names
    ))

    product_names = (
        product_industry_concordance
        .index.get_level_values(1)
    )
    product_codes = (
        product_industry_concordance
        .index.get_level_values(2)
    )
    industry_names = (
        product_industry_concordance
        .columns.get_level_values(1)
    )
    industry_codes = (
        product_industry_concordance
        .columns.get_level_values(2)
    )

    product_code_map = dict(zip(
        product_names,
        product_codes,
    ))
    product_code_map_reverse = dict(zip(
        product_codes,
        product_names,
    ))

    industry_code_map = dict(zip(
        industry_names,
        industry_codes,
    ))
    industry_code_map_reverse = dict(zip(
        industry_codes,
        industry_names,
    ))

    # These includes industry + final demand sectors
    industry_sectors_map = {
        **industry_code_map,
        **final_demand_map
    }
    industry_sectors_map_reverse = {
        **industry_code_map_reverse,
        **final_demand_map_reverse
    }

    # These includes products + final demand sectors
    product_sectors_map = {
        **product_code_map,
        **final_demand_map
    }
    product_sectors_map_reverse = {
        **product_code_map_reverse,
        **final_demand_map_reverse
    }

    exiobase_classification = {
        "Final demand sectors": final_demand,
        "Product industry concordance": product_industry_concordance,
        "Country list": country_list,
        "Product name to code": product_code_map,
        "Product code to name": product_code_map_reverse,
        "Industry name to code": industry_code_map,
        "Industry code to name": industry_code_map_reverse,
        "Final demand name to code": final_demand_map,
        "Final demand code to name": final_demand_map_reverse,
        "Industry sectors name to code": industry_sectors_map,
        "Industry sectors code to name": industry_sectors_map_reverse,
        "Product sectors name to code": product_sectors_map,
        "Product sectors code to name": product_sectors_map_reverse
    }

    return exiobase_classification


def load_exiobase_sut_tables(
        path: Path,
        data_path: Path,
        year: int
):
    """loads monetary SUT tables and prepares them to be used in
    03_build_accounts.py.

    Args:
        path (Path): path to SUT tables.
        data_path (Path): path to data.
        year (int): year of the SUT tables.

    Returns:
        (pd.DataFrame, pd.DataFrame, list, list, list): supply
            and use tables, as well as MRIO order lists used to
            save final results.
    """
    auxiliary_path = (
        data_path
        / "00_auxiliary"
        / f"exiobase_core"
    )
    mrio_classifications = load_exiobase_classification(
        path=auxiliary_path
    )

    mrio_final_demand_order = (
        mrio_classifications["Final demand sectors"]
        .columns.get_level_values(1).values.tolist()
    )
    mrio_industry_order = list(
        mrio_classifications["Industry name to code"].keys()
    )
    mrio_country_order = (
        mrio_classifications["Country list"]["Code"]
        .values.tolist()
    )

    use_tables = pd.DataFrame()
    supply_tables = pd.DataFrame()

    for region in tqdm(
            mrio_country_order,
            desc=f"Extensions for {year}. Looping regions:"):
        # Load use tables
        # Domestic
        usebpdom = pd.read_csv(
            path / f"{region}_{year}_usebpdom.csv",
            index_col=[0],
            header=[0]
        )

        # Import
        usebpimp = pd.read_csv(
            path / f"{region}_{year}_usebpimp.csv",
            index_col=[0],
            header=[0]
        )

        # Combine (drop extra domestic rows)
        use = (usebpdom + usebpimp).dropna(axis=0, how="any")

        # Load and rename supply table
        # ! Dropping last column because it includes exports
        supply = pd.read_csv(
            path / f"{region}_{year}_sup.csv",
            index_col=[0],
            header=[0]
        ).iloc[:, :-1]

        # Concat
        use_tables = pd.concat(
            [pd.concat([use], keys=[region], axis=0), use_tables],
            axis=0
        )
        supply_tables = pd.concat(
            [pd.concat([supply], keys=[region], axis=0), supply_tables],
            axis=0
        )

    # Rename indexes and columns
    use_tables = (
        use_tables
        .rename(mrio_classifications["Product code to name"], axis=0, level=1)
        .rename(mrio_classifications["Industry sectors code to name"], axis=1)
        .rename_axis(["MRIO_country", "MRIO_product"], axis=0)
        .rename_axis(["MRIO_industry"], axis=1)
    )

    # Setting exports to zero in use tables
    export_sectors = list(use_tables.filter(regex="xport", axis=1).columns)
    use_tables.loc[:, export_sectors] = 0

    # Rename indexes and columns
    supply_tables = (
        supply_tables
        .rename(mrio_classifications["Product code to name"], axis=0, level=1)
        .rename(mrio_classifications["Industry sectors code to name"], axis=1)
        .rename_axis(["MRIO_country", "MRIO_product"], axis=0)
        .rename_axis(["MRIO_industry"], axis=1)
    )

    # We expand supply tables with final demand sectors
    # that are zeros to get them on the same dimension
    final_demand_sectors = (
        list(mrio_classifications["Final demand name to code"].keys())
    )
    final_demand = (
        pd.DataFrame(
            np.zeros((
                len(supply_tables.index),
                len(final_demand_sectors)
            )),
            index=supply_tables.index,
            columns=final_demand_sectors,
        ).rename_axis(["MRIO_industry"], axis=1)
    )

    supply_tables = (
        pd.concat(
            [supply_tables, final_demand],
            names=["MRIO_industry"],
            axis=1
        )
    )
    return (
        use_tables,
        supply_tables,
        mrio_country_order,
        mrio_industry_order,
        mrio_final_demand_order
    )