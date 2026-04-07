"""
Annotation conversion and processing utilities
"""

import csv
import os
from pathlib import Path

import pandas as pd
import numpy as np
import subprocess

def check_bedtools_installed():
    """Validates that the bedtools binary is accessible in the system PATH."""
    try:
        subprocess.run(['bedtools', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "The 'bedtools' binary was not found. "
            "If you installed zavolab_pyutils via pip, you must install bedtools manually "
            "(e.g., 'conda install -c bioconda bedtools' or 'sudo apt install bedtools')."
        )

def parse_gtf_attributes_into_pd_dataframes(gtf_file,input_skiprows=5) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Get full gtf, exons, and genes from a GTF annotation file.
    
    Parameters
    ----------
    gtf_file : str
        Path to input GTF file.
    input_skiprows : int, optional
        Number of header lines to skip in the GTF file. Default is 5 (compatible with GENCODE GTF files).
    Returns
    -------
    gtf_df : pd.DataFrame
        Original pandas-parsed GTF DataFrame.
    genes_df : pd.DataFrame
        DataFrame containing gene-level information.
    exons_df : pd.DataFrame
        DataFrame containing exon-level information.
    Notes
    -----
    """ 
    gtf_df = pd.read_csv(gtf_file, delimiter="\t", index_col=None, header=None, skiprows=input_skiprows)

    # extract gene-level information
    genes = gtf_df.loc[gtf_df[2]=='gene'].reset_index(drop=True)
    genes['gene_type'] = genes[8].str.split('gene_type "',expand=True)[1].str.split('";',expand=True)[0]
    genes['gene_name'] = genes[8].str.split('gene_name "',expand=True)[1].str.split('";',expand=True)[0]
    genes['gene_id'] = genes[8].str.split('gene_id "',expand=True)[1].str.split('";',expand=True)[0]
    print(f"Extracted gene-level information for {len(genes)} genes.")

    # extract exon-level information
    exons = gtf_df.loc[gtf_df[2]=='exon'].reset_index(drop=True)
    exons['gene_type'] = exons[8].str.split('gene_type "',expand=True)[1].str.split('";',expand=True)[0]
    exons = exons.loc[exons['gene_type'].isin(['protein_coding'])].reset_index(drop=True) # leaving only protein-coding genes
    exons['transcript_id'] = exons[8].str.split('transcript_id "',expand=True)[1].str.split('";',expand=True)[0]
    exons['gene_id'] = exons[8].str.split('gene_id "',expand=True)[1].str.split('";',expand=True)[0]
    exons['exon_number'] = exons[8].str.split('exon_number ',expand=True)[1].str.split(';',expand=True)[0].str.replace('"','').astype('int')
    exons['t']=1
    exons = pd.merge(exons.drop(['t'],axis=1),
                     exons.groupby('transcript_id').agg({'t':sum}).reset_index(),how='inner',on='transcript_id')
    print(f"Extracted exon-level information for {len(exons)} exons.")
    genes_df = genes
    exons_df = exons

    return gtf_df, genes_df, exons_df

def get_terminal_exons(
    exons_df:pd.DataFrame,
    min_exons_per_transcript=3, 
    exclude_overlapping_exons=True, 
    TE_extension=100,temp_dir=None
    ) -> pd.DataFrame:
    """
    Get terminal exons from a GTF annotation file.
    
    Parameters
    ----------
    exons_df : pd.DataFrame
        DataFrame containing exon-level information.
    min_exons_per_transcript : int, optional
        Minimum number of exons per transcript to consider. Default is 3.
        To allow, e.g. for gene expression analysis using at least one internal exon
    exclude_overlapping_exons : bool, optional
        Whether to exclude overlapping exons. Default is True.
    TE_extension : int, optional
        Number of base pairs to extend the terminal exons. Default is 100.
    temp_dir : str, optional
        Path to temporary directory for intermediate files. 
        If None, a 'temp' directory will be created in the current working directory or $TMPDIR environment variable if present.
        Default is None.
    Returns
    -------
    TE_bed_df : pd.DataFrame
        BED-formatted DataFrame containing only terminal exons.
    Notes
    -----
    """
    check_bedtools_installed() # Fail fast with a helpful error message if bedtools is not available

    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = os.getenv('TMPDIR', default='./temp/')  # Use environment variable for temporary directory or fallback to './temp/'
        os.makedirs(temp_dir, exist_ok=True)
    else:
        os.makedirs(temp_dir, exist_ok=True)
    print(f"Checked bedtools installation and created temporary directory at {temp_dir}...")

    # selecting terminal exons of transcripts with at least min_exons_per_transcript exons (to allow for gene expression analysis using at least one internal exon)
    exons_for_tandemPAS = exons_df.loc[(exons_df['t']==exons_df['exon_number'])&(exons_df['t']>=min_exons_per_transcript)].reset_index(drop=True)
    exons_for_tandemPAS = exons_for_tandemPAS[[0,3,4,6,'transcript_id','gene_id']].rename(columns ={0:'chr',3:'ex_start',4:'ex_end',6:'strand'})
    ### there should be no exon from the same gene that starts downstream from the reported start
    most_distal_exons_of_genes = exons_for_tandemPAS.groupby(['gene_id','chr','strand']).agg({'ex_start':max,'ex_end':min}).reset_index().rename(columns={'ex_start':'ex_start_dist','ex_end':'ex_end_dist'})
    exons_selected_plus = pd.merge(exons_for_tandemPAS,most_distal_exons_of_genes.loc[most_distal_exons_of_genes['strand']=='+'].rename(columns={'ex_start_dist':'ex_start'})[['gene_id','ex_start']],how='inner',on=['gene_id','ex_start'])
    exons_selected_plus = exons_selected_plus.groupby(['chr','strand','gene_id','ex_start']).agg({'ex_end':max}).reset_index()[['chr','strand','gene_id','ex_start','ex_end']]

    exons_selected_minus = pd.merge(exons_for_tandemPAS,most_distal_exons_of_genes.loc[most_distal_exons_of_genes['strand']=='-'].rename(columns={'ex_end_dist':'ex_end'})[['gene_id','ex_end']],how='inner',on=['gene_id','ex_end'])
    exons_selected_minus = exons_selected_minus.groupby(['chr','strand','gene_id','ex_end']).agg({'ex_start':min}).reset_index()[['chr','strand','gene_id','ex_start','ex_end']]
    exons_for_tandemPAS = pd.concat([exons_selected_plus,exons_selected_minus]).reset_index(drop=True)
    print(f"Selected {len(exons_for_tandemPAS)} terminal exons from transcripts with at least {min_exons_per_transcript} exons.")

    ###
    # exclude overlapping exons (for unstranded data)
    ###

    if exclude_overlapping_exons:
        # add extension of 100 nt to 3'end to check for overlaps
        exons_for_tandemPAS_plus = exons_for_tandemPAS.loc[exons_for_tandemPAS['strand']=='+']
        exons_for_tandemPAS_plus['ex_end_OL'] = exons_for_tandemPAS_plus['ex_end']+TE_extension
        exons_for_tandemPAS_plus['ex_start_OL'] = exons_for_tandemPAS_plus['ex_start']

        exons_for_tandemPAS_minus = exons_for_tandemPAS.loc[exons_for_tandemPAS['strand']=='-']
        exons_for_tandemPAS_minus['ex_end_OL'] = exons_for_tandemPAS_minus['ex_end']
        exons_for_tandemPAS_minus['ex_start_OL'] = exons_for_tandemPAS_minus['ex_start']-TE_extension

        exons_for_tandemPAS = pd.concat([exons_for_tandemPAS_plus,exons_for_tandemPAS_minus]).reset_index(drop=True)

        exons_for_tandemPAS['score'] = 0
        
        nonsorted_bed = os.path.join(temp_dir, 'exons_for_tandemPAS.bed')
        exons_for_tandemPAS[['chr','ex_start_OL','ex_end_OL','gene_id','score','strand']].to_csv(
            nonsorted_bed, 
            sep=str('\t'),header=False,index=None,quoting=csv.QUOTE_NONE,
        )
        
        # sort bed file
        sorted_bed = os.path.join(temp_dir, 'exons_for_tandemPAS.sorted.bed')
        clustered_bed = os.path.join(temp_dir, 'exons_for_tandemPAS.clustered.bed')
        command = 'bedtools sort -i '+nonsorted_bed+' > '+sorted_bed
        out = subprocess.check_output(command, shell=True)
        
        print(f"Sorted BED file for bedtools clustering.")
        # cluster overlapping exons
        command = 'bedtools cluster -i '+sorted_bed+' > '+clustered_bed
        out = subprocess.check_output(command, shell=True)
        print(f"Clustered BED file")
        
        exons_for_tandemPAS_clustred = pd.read_csv(clustered_bed,delimiter="\t",index_col=None,header=None)
        exons_for_tandemPAS_clustred['t']=1
        gr = exons_for_tandemPAS_clustred.groupby(6).agg({'t':sum}).reset_index()

        non_overlapping_exons = list(gr.loc[gr['t']==1][6].unique())
        exons_for_tandemPAS_selected = exons_for_tandemPAS_clustred.loc[exons_for_tandemPAS_clustred[6].isin(non_overlapping_exons)].reset_index(drop=True)

        exons_for_tandemPAS_selected = pd.merge(exons_for_tandemPAS_selected.rename(columns={3:'gene_id'}),
                                            exons_for_tandemPAS[['gene_id','ex_start','ex_end']],
                                            how='left',on='gene_id')
        print(f"Selected {len(exons_for_tandemPAS_selected)} non-overlapping exons.")
    else:
        exons_for_tandemPAS_selected = exons_for_tandemPAS.copy()
        print(f"Skipped overlapping exon filtering. Retained {len(exons_for_tandemPAS_selected)} terminal exons.")

    exons_for_tandemPAS_selected['ex_start'] = exons_for_tandemPAS_selected['ex_start']-1 # to stick to bed format
    TE_bed_df = exons_for_tandemPAS_selected[[0,'ex_start','ex_end','gene_id',4,5]].copy()

    return TE_bed_df


def get_GTF_for_gene_expression_analysis():

    return None



def convert_gff_to_gtf(gff_file, gtf_file):
    """
    Convert GFF format annotation file to GTF format.
    
    Parameters
    ----------
    gff_file : str
        Path to input GFF file.
    gtf_file : str
        Path to output GTF file.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function handles GFF3 to GTF format conversion, standardizing
    the attribute column format and adjusting coordinate systems as needed.
    """
    # Placeholder implementation
    raise NotImplementedError("convert_gff_to_gtf is currently under development.")


def convert_gtf_to_gff(gtf_file, gff_file):
    """
    Convert GTF format annotation file to GFF3 format.
    
    Parameters
    ----------
    gtf_file : str
        Path to input GTF file.
    gff_file : str
        Path to output GFF3 file.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function handles GTF to GFF3 format conversion, standardizing
    the attribute column format.
    """
    # Placeholder implementation
    raise NotImplementedError("convert_gtf_to_gff is currently under development.")


def parse_gtf_attributes(attribute_string):
    """
    Parse GTF/GFF attribute column into a dictionary.
    
    Parameters
    ----------
    attribute_string : str
        The attribute column from a GTF/GFF file.
    
    Returns
    -------
    dict
        Parsed attributes as key-value pairs.
    """
    # Placeholder implementation
    raise NotImplementedError("parse_gtf_attributes is currently under development.")
