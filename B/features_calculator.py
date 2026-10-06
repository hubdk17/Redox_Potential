# libraries
import os
import shutil
import logging
from tqdm import tqdm
from collections import OrderedDict
from Bio.PDB import PDBList, PDBParser
#from PyBioMed.PyProtein import CTD
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath("../"))
from utils import (
    get_baricentro,
    get_atoms_coord,
    get_covariance,
    inizializza_dict_amm,
    feature_conteggio,
    feature_conteggio_specific_atom,
    specific_feature,
    d3to1,
    amm_names
)

# Parameters
round_c=2
all_atom_CofAtom=False              
list_Bar = list(np.arange(8, 17))  # set range for sampling r1
list_CofAtom = list(np.arange(3, 6))  # set range for sampling r2
path_dir = ""
path_dataset="../Database Redox Pot Fe2S2 proteins.xlsx"  #initial file di partenza

cofactors_names=["FES", "FE"]             #name of the cofactor
cofactor_index_dict={               #index considered for the cofactor
     "FES": (0,4),
     "FE": (0,1)
 }

atom_specific_cofactors={
     "FES":["FE1","FE2","S1","S2"],  #specific atom considered for the cofactor
     "FE":["FE"]
 }



# initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create a dir to save the pdb filese
#if "pdb-files" not in os.listdir():
#    os.mkdir("pdb-files")
path_pdb = "../pdb-files"

# create a dir to save features files

if "dataset_features" not in os.listdir():
    os.mkdir("dataset_features")



# read file
dataset = pd.read_excel(os.path.join(path_dir, path_dataset)).set_index(
    "Unnamed: 0"
)[["PDB Code(s)", "mV", "pH", ]] #add "Methodology" column if it needs to be considered
dataset=dataset.dropna(subset=["PDB Code(s)","mV","pH"]) #add "Methodology" column if it needs to be considered


proteins_PDB = list(OrderedDict.fromkeys(dataset["PDB Code(s)"]))  # list of PDB ID used
#proteins_PDB=[el for el in proteins_PDB if len(el)==4]
#proteins_PDB=['1frr', '1wri', '2qh7']
# read file with amino acids features
table_amm = pd.read_csv(
    os.path.join(path_dir, "../tableAmm.txt"), sep="\t", index_col=1
)
#table_amm.index = [el.upper() for el in table_amm.index]
table_amm = table_amm.iloc[:, 1:]

# %% start "for cycle" to consider different combination of radii
for bar in list_Bar:
    for CofAtom in list_CofAtom:
        logger.info(
            f"Starting feature computation for dataset of barycenter: {str(bar)} and CofAtom_radius: {str(CofAtom)}"
        )

        # %%
        df_total = pd.DataFrame()  # initialize pandas dataframe to save results
        names = list()  # initialize list to save name for dataframe columns
        # %% access to pdb database
        pdbl = PDBList()

        # %% "for cycle" on each protein

        for idx, name_protein in tqdm(enumerate(proteins_PDB), desc="Fe-S proteins"):
            # download pdb file
            if os.path.isfile(path_pdb+"/"+name_protein+".pdb")==False and len(name_protein)>4:
                #il file pdb non c'è allora salto
                continue
            if os.path.isfile(path_pdb+"/"+name_protein+".pdb")==False:
                #se non c'è il file lo scarico
                pdbl.retrieve_pdb_file(name_protein, pdir=path_pdb, file_format="pdb")
                shutil.copy(path_pdb+"/pdb"+name_protein+".ent", path_pdb+"/"+name_protein+".pdb")
            parser = PDBParser(PERMISSIVE=True, QUIET=True)
            structure = parser.get_structure(
                os.path.join(path_pdb, name_protein.lower()),
                os.path.join(path_pdb, f"{name_protein.lower()}.pdb"),
            )
            
            # generation dict
            dict_residues = dict()  # inizialize dict for residues
            Cof_coord = dict()  # inizialize dict for barycenter coordinate
            Cof_coords = (
                dict()
            )  # inizialize dict for coordinate for each atom of the isoalloxazine CofAtom

            # % start for cycle
            for model in structure:
                # header of the pdb file
                header = structure.header
                chains = model.get_chains()

                # scan on chains
                for chain in chains:
                    atom_el={}
                    
                    residue_names = [
                        residue.resname for residue in chain.get_residues()
                    ]  # check on FES and FE

                    if len(list(set(cofactors_names) & set(residue_names)))==0:
                        logger.info("pdb:"+name_protein+",chain:"+chain.id+":NO cofactor found")
                        continue
                    else:
                        names.append(name_protein + "chain_" + chain.id)

                    dict_residues[chain.id] = dict()

                    # scan on residues
                    for residue in chain.get_residues():
                        if residue.resname in amm_names:
                            # Dictionary with IDs as keys and amino acid names as values
                            dict_residues[chain.id][residue.id[1]] = residue.resname
                        #elif len(residue.resname)==4 and residue.resname[0]=="A" and residue.resname[1:] in amm_names:
                            #dict_residues[chain.id][residue.id[1]] = residue.resname[1:]

                        elif residue.resname in cofactors_names:
                            # The residue under consideration is the cofactor of interest
                            ind1=cofactor_index_dict[residue.resname][0]
                            ind2=cofactor_index_dict[residue.resname][1]
                            # save info about the cofactor
                            for el in residue.get_atoms():
                                #considering cofactor's atoms
                                if el.id in atom_specific_cofactors[residue.resname]:
                                    atom_el[el.id]=el.coord
                            # calculate barycenter coordinate
                            Cof_coord_el = get_baricentro(residue, ind1, ind2)
                            # calculate CofAtom's atoms coordinate
                            Cof_coords_el_dict = get_atoms_coord(residue, ind1, ind2)

                    atom_names=list(Cof_coords_el_dict.keys())

                    # features about amino acids count
                    if all_atom_CofAtom:
                        dict_cont = inizializza_dict_amm(amm_names,all_atom_CofAtom=all_atom_CofAtom)
                    else:
                        dict_cont = inizializza_dict_amm(amm_names,all_atom_CofAtom=all_atom_CofAtom,atom_names=atom_names)

                    dict_cont = feature_conteggio(
                        dict_cont,
                        chain,
                        Cof_coord_el,
                        Cof_coords_el_dict,
                        bar,
                        CofAtom,
                        dict_residues,
                        amm_names,
                        all_atom_CofAtom=all_atom_CofAtom
                    )
                    
                    # features calculation
                    if all_atom_CofAtom:
                        names_type=["Bar","Protein","CofAtom"]
                    else:

                        names_type=["Bar","Protein"]
                        names_type.extend(["CofAtom."+el for el in atom_names])
                    
                    total=dict()
                    for name in names_type:
                    # count number of total amino acids
                        total[name] = sum(
                            [dict_cont[name+f".{nome}"] for nome in amm_names]
                        )  # respect the r1 sphere
                        if total[name]==0:
                            total[name]=1
                        
                        for col in table_amm.columns:  # 28+28
                            values = table_amm[col]
                            val_feature = np.sum(
                                [
                                    values[nome] * dict_cont[name+f".{nome}"]
                                    for nome in amm_names
                                ]
                            )
                            dict_cont[name+f".{col}"] = val_feature

                        # add some specific feature
                        dict_cont = specific_feature(
                            dict_cont, prefix=name+".", mean=True, total=total[name]
                        )

                    # features about specific atoms
                    for atom_id in atom_el.keys():
                        atom_nearest_res, atom_3_nearest_res=feature_conteggio_specific_atom(
                            chain,
                            atom_el[atom_id],
                            dict_residues,
                            amm_names,
                        )
                        # nearest amino acid respect N5
                        for col in table_amm.columns:  # 28
                            value = table_amm.loc[atom_nearest_res, col]
                            dict_cont[atom_id+"_nearest"+f".{col}"] = value

                        # 3 nearest amino acid respect N5
                        for col in table_amm.columns:  # 28
                            value = table_amm.loc[atom_3_nearest_res, col].sum()
                            dict_cont["Around_"+atom_id+f".{col}"] = value

                    dict_cont["PDB Code(s)"] = name_protein

                    # add some information from the pdb file header
                    #dict_cont["organism"] = header["source"]["1"]["organism_scientific"]

                    # features by PyBioMed
                    lista_fasta = ""
                    for residue_name in residue_names:
                        if residue_name in d3to1.keys():
                            lista_fasta = lista_fasta + d3to1[residue_name]

                    #protein_descriptor = CTD.CalculateC(lista_fasta)

                    # dicts merge between "aa count features" and "PyBioMed features"
                    dict_cont = {**dict_cont}#, **protein_descriptor}

                    ###end features calculation !!
                    df = pd.DataFrame.from_dict(dict_cont, orient="index")

                    # df_total update !!
                    df_total = pd.concat([df_total, df], axis=1)

        # %% end of for cycle on proteins list
        df_total = df_total.fillna(0)
        # columns name update
        df_total.columns = names
        df_total = df_total.transpose()

        cols = df_total.columns.tolist()
        df_total = df_total[cols]
        logger.info(
            f"Saving Features for dataset of barycenter: {str(bar)} and CofAtom_radius: {str(CofAtom)}"
        )
        #compute number of chains
        df_total_len = df_total.groupby("PDB Code(s)")[df_total.columns[0]].agg(
            lambda x: len(x)
        )

        df_total = df_total.groupby("PDB Code(s)").agg(
            lambda x: np.round(np.mean(x), round_c)
        )  # groupby for PDB ID if a protein has 2+ chains
        df_total.insert(loc = 0, column = 'n_chains', value = df_total_len.values)

        dataset2 = dataset.join(
            df_total, on="PDB Code(s)"
        )  # join pandas function to add information about Em and pH to the features dataset
        
        dataset2.to_excel(
            os.path.join(
                path_dir,
                f"dataset_features/dataset_protein_{str(bar)}_{str(CofAtom)}.xlsx",
            )
        )  # save final dataset
