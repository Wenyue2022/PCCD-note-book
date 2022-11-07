This file explains the steps to install and run the PCCD method described in the article
'The post-transcriptional regulation of TFs in postmitotic motoneurons shapes the axon-muscle connectome'
by Wenyue Guan et al., 2021.

To run the GRN, geometric and the 3D architectural-GRN models it is necessary to install first some python packages. The procedure has been automated using CONDA. Here is how to proceed.

######################
 Installation
######################

STEP 0: (Preliminary) please install the CONDA system on your system:
(https://conda.io/projects/conda/en/latest/user-guide/install)
Open a command line shell on your system, and go in the directory where you downloaded the simulation files.

$ cd path_of_the_directory_where_you_downloaded_and_uncompressed_the_files

STEP 1: Install the programming environment using the conda file pccd.yml and the conda command:

$ conda env create -f pccd.yml -n pccd

This installs the required python packages and creates a conda environment.
Here the conda environment's name is chosen to be 'pccd' via option '-n pccd')

STEP 2: Then activate the new environment:

$ conda activate pccd

You are ready to run the PCCD method.

######################
 Running the code
######################

The method is run using python jupyter notebooks.

To lauch the jupyter notebook, after acitvation of the pccd conda environment, launch the command:

$ jupyter notebook PCCD.ipynb

This should open your web browser with a PCCD notebook page. Then execute de cells in the order of the text (to execute a cell, position the pointer in the cell
and type shift-ENTER). Datafiles used are located in the sub-directory 'DataBasis'
