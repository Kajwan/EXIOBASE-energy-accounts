Repository to be published along with the manuscript:
>EXIOBASE energy accounts: Improving precision in an open-sourced procedure applicable to any MRIO database.

The procedure consists of four scripts which are described briefly in the table below (from paper).

To run the procedure, you first need to
1. Install the conda environment using the  *env/conda_env.yaml* file. 2
2. Make a copy of the *env/template_variables.env* named *env/variables.env*, and adjust the variable names for your system.

| Script | 01_unpack_weeb.py | 02_balance_to_accounting.py | 03_disaggregate_and_adjust.py| 04_build_accounts.py|
| --- | --- | --- | --- | --- |
| **Purpose of script** | Prepare the WEEB for processing by removing non-numeric inputs and categorizing flow names. | Transform the energy balances to energy accounts. | Disaggregates the energy accounts from 156 countries and 35 regional aggregates to 208 UN regions. Energy accounts are transformed from territorial to residential principle using transport models for international aviation and marine flows, as well as road transport within the EU. | Creates proxies for how the IEA energy accounts should be allocated to MRIO industries by first creating proxies using monetary supply-use tables as well as other auxiliary data. <br> Energy is then allocated using these proxies and masks are applied to create the «net energy use» and «emission relevant energy» accounts. |
| **Input(s)**| Comma separated file created with Beyond 20/20 browser with the dimensions: [Year, country, product] x [Unit, flow] from the WBIG.ivt file. | Yearly energy balances dataset.| • Yearly energy accounts<br>• UN GDP data<br>• Adjustment factors| • Residential principle energy accounts<br>• Masks and concordances<br> • Monetary use and supply tables 
| **Output(s)**| Prepared energy balances data split into yearly files. | Energy accounts for all IEA regions with bunkers allocated to a global «World» region. | Disaggregated and adjusted energy accounts following the residential principle. | Net energy use & emission relevant energy use account at IEA product, IEA flow, and UN region level resolution.|

Besides the four scripts, two Python modules are required. One that is MRIO specific (mrio_functions.py) and one that is generic (functions.py).

The procedure relies on the [country converter package](https://pypi.org/project/country-converter/) for mapping countries. If the official version of the package is not updated to your needs, you ned to modify 

The code tries to adhere to the *pycodestyle* linter, but ignores maximum line length in some cases.
