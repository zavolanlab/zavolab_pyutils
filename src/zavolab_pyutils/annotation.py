"""
Annotation conversion and processing utilities
"""

import csv
import os
from pathlib import Path

import pandas as pd
import numpy as np
import subprocess

import argparse
import sys
from Bio import SeqIO

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

def parse_gtf_attributes_into_pd_dataframes(gtf_file,
                                            input_skiprows=5,
                                            gene_type_field='gene_type',
                                            extract_exon_number=True,
                                            extract_gene_name_in_exons=True,
                                            verbose=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Get full gtf, exons, and genes from a GTF annotation file.
    
    Parameters
    ----------
    gtf_file : str
        Path to input GTF file.
    input_skiprows : int, optional
        Number of header lines to skip in the GTF file. Default is 5 (compatible with GENCODE GTF files).
    gene_type_field : str, optional
        The field name for gene type in the GTF file. Default is 'gene_type'.
    extract_exon_number : bool, optional
        Whether to extract exon_number from the GTF attributes. Default is True.
    extract_gene_name_in_exons : bool, optional
        Whether to extract gene_name from the GTF attributes in exon entries. Default is True.
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
    genes['gene_type'] = genes[8].str.split(gene_type_field+' "',expand=True)[1].str.split('"',expand=True)[0]
    genes['gene_name'] = genes[8].str.split('gene_name "',expand=True)[1].str.split('"',expand=True)[0]
    genes['gene_id'] = genes[8].str.split('gene_id "',expand=True)[1].str.split('"',expand=True)[0]
    print(f"Extracted gene-level information for {len(genes)} genes.")

    # extract exon-level information
    exons = gtf_df.loc[gtf_df[2]=='exon'].reset_index(drop=True)
    if verbose:
        print(f"extracted {len(exons)} exon elements in the provided gtf_df dataframe. Will exctract gene_type\n")
    exons['gene_type'] = exons[8].str.split(gene_type_field+' "',expand=True)[1].str.split('"',expand=True)[0]
    if verbose:
        print(f"extracted gene_type of exon elements in the provided gtf_df dataframe. Will exctract transcript_id\n")
    exons['transcript_id'] = exons[8].str.split('transcript_id "',expand=True)[1].str.split('"',expand=True)[0]
    if verbose:
        print(f"extracted transcript_id of exon elements in the provided gtf_df dataframe. Will exctract gene_id\n")
    exons['gene_id'] = exons[8].str.split('gene_id "',expand=True)[1].str.split('"',expand=True)[0]
    if verbose:
        print(f"extracted gene_id of exon elements in the provided gtf_df dataframe.\n")
    if extract_gene_name_in_exons:
        if verbose:
            print(f"Will extract gene_name of exon elements in the provided gtf_df dataframe.\n")
        exons['gene_name'] = exons[8].str.split('gene_name "',expand=True)[1].str.split('"',expand=True)[0]
        if verbose:
            print(f"extracted gene_name of exon elements in the provided gtf_df dataframe.\n")
    if extract_exon_number:
        if verbose:
            print(f"Will extract exon_number of exon elements in the provided gtf_df dataframe.\n")
        exons['exon_number'] = exons[8].str.split('exon_number ',expand=True)[1].str.split(';',expand=True)[0].str.replace('"','').astype('int')
        if verbose:
            print(f"extracted exon_number of exon elements in the provided gtf_df dataframe.\n")
    if verbose:
        print(f"Will extract total number of exons in the transcripts (column 't') in the provided gtf_df dataframe.\n")
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
        print(f"Selecting non-overlapping exons. Using bedtools to overlap exon coordinates.")
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


def get_GTF_for_gene_expression_analysis(
        exons_df:pd.DataFrame,
        out_gtf_path: str,
        gene_types_to_include: list = None,
        exclude_first_exons: bool = True,
        exclude_terminal_exons: bool = True,
        remove_segments_associated_with_multiple_genes_on_same_strand: bool = True,
        f: float = 0.8,
        temp_dir: str = None,
    ) -> pd.DataFrame:
    """
    Get GTF annotation for gene expression analysis, containing (almost) constitutive exons of selected gene types
    
    Parameters
    ----------
    exons_df : pd.DataFrame
        DataFrame containing exon-level information. Expected to have columns 'transcript_id','gene_type', 'gene_id', 'gene_name',
        'exon_number', 't' (total number of exons in the transcript), and other standard GTF columns.
        The output of `parse_gtf_attributes_into_pd_dataframes()` can be used as input for this function.
    out_gtf_path : str
        Path to output GTF file that will be written with the selected genomic regions, GENCODE-like formatted with gene-transcript-exon structure.
    gene_types_to_include : list, optional
        List of gene types to include. Default is None, which would result in inclusion of all present gene types.
    exclude_first_exons : bool, optional
        Whether to exclude first exons. Default is True.
    exclude_terminal_exons : bool, optional
        Whether to exclude terminal (last) exons. Default is True.
    remove_segments_associated_with_multiple_genes_on_same_strand : bool, optional
        Whether to remove segments associated with multiple genes on the same strand. Default is True.
    f : float, optional
        Minimum fraction of transcripts of a gene that must contain an exonic segment for it to be included. Default is 0.8.
    temp_dir : str, optional
        Path to temporary directory for intermediate files. 
        If None, a 'temp' directory will be created in the current working directory or $TMPDIR environment variable if present. Default is None.    
    Returns
    -------
    gene_expression_gtf : pd.DataFrame
        GTF-formatted DataFrame containing selected genomic regions, GENCODE-like formatted with gene-transcript-exon structure.
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
    print(f"Checked bedtools installation and created temporary directory at {temp_dir}...\n")

    exons_sel_df = exons_df.copy()
    print(f"start with {len(exons_sel_df)} exons.")
    if exclude_first_exons:
        exons_sel_df = exons_sel_df.loc[exons_sel_df['exon_number']>1].reset_index(drop=True)
        print(f"After excluding first exons: {len(exons_sel_df)} exons.")
    if exclude_terminal_exons:
        exons_sel_df = exons_sel_df.loc[(exons_sel_df['t']==1)|(
            (exons_sel_df['t']>exons_sel_df['exon_number']))].reset_index(drop=True) # take non-terminal and non-first exons only, unless one-exon gene
        print(f"After excluding terminal exons: {len(exons_sel_df)} exons.")
    if gene_types_to_include is not None:
        exons_sel_df = exons_sel_df.loc[exons_sel_df['gene_type'].isin(gene_types_to_include)].reset_index(drop=True)
        print(f"After filtering by gene types: {len(exons_sel_df)} exons.")
    
    exons_sel_df['score'] = 0
    exons_sel_bed = exons_sel_df[[0,3,4,'transcript_id','score',6]]
    exons_sel_bed[3] = exons_sel_bed[3]-1
    
    nonsorted_EXONS_bed = os.path.join(temp_dir, 'exons_in_transcripts.bed')
    sorted_EXONS_bed = os.path.join(temp_dir, 'exons_in_transcripts.sorted.bed')

    exons_sel_bed.to_csv(nonsorted_EXONS_bed, sep=str('\t'),header=False,index=None,quoting=csv.QUOTE_NONE)
    command = 'bedtools sort -i '+nonsorted_EXONS_bed+' > '+sorted_EXONS_bed
    out = subprocess.check_output(command, shell=True)
    print(f"Sorted BED file with all exons. Exons with same genomic coordinates but different transcript_id are kept as separate entries.\n")

    ####
    # get all unique exonic segments
    ####
    exons_sel_df['start_True'] = True
    exons_sel_df['start_False'] = False

    a1 = []
    a1 = a1 + [exons_sel_df[[0,3,6,'start_True']].drop_duplicates().rename(columns={3:1,6:2,'start_True':3}),
                        exons_sel_df[[0,4,6,'start_False']].drop_duplicates().rename(columns={4:1,6:2,'start_False':3})]

    points = pd.concat(a1).sort_values([2,0,1,3]).drop_duplicates([0,1,2]).reset_index(drop=True) # prioritize element ends over starts if the coordinate matches exactly

    b = []
    prev_chr,prev_coord,prev_str='',-1,''
    for row in points.values:
        if (row[0]==prev_chr and row[2]==prev_str):
            b.append([row[0],prev_coord,(row[1]-1 if row[3] else row[1]),'n',0,row[2]])
        prev_chr,prev_coord,prev_str = row[0],(row[1] if row[3] else row[1]+1),row[2]
    segments = pd.DataFrame(b)
    segments[1] = segments[1]-1
    
    print(f"Identified {len(segments)} unique exonic segments.\n")
    nonsorted_SEGMENTS_bed = os.path.join(temp_dir, 'exonic_segments.bed')
    sorted_SEGMENTS_bed = os.path.join(temp_dir, 'exonic_segments.sorted.bed')
    segments.to_csv(nonsorted_SEGMENTS_bed, sep=str('\t'),header=False,index=None,quoting=csv.QUOTE_NONE)

    command = 'bedtools sort -i '+nonsorted_SEGMENTS_bed+' > '+sorted_SEGMENTS_bed
    out = subprocess.check_output(command, shell=True)
    print(f"Sorted BED file with all unique exonic segments.\n")

    ###
    # intersect segments with exons to calculate in how many transcripts each segment is present
    ###
    intersection_bed = os.path.join(temp_dir, 'exonic_segments.intersection.bed')
    command = 'bedtools intersect -sorted -wao -s -a '+sorted_SEGMENTS_bed+' -b '+sorted_EXONS_bed+' > '+intersection_bed
    out = subprocess.check_output(command, shell=True)
    print(f"Intersected exonic segments with exons to calculate transcript support for each segment.\n")

    exonic_segments_intersection = pd.read_csv(intersection_bed,delimiter="\t",index_col=None,header=None)
    exonic_segments_intersection['segment_len'] = exonic_segments_intersection[2]-exonic_segments_intersection[1]
    # QC: remove cases when overlap is exactly the segment length
    exonic_segments_intersection = exonic_segments_intersection.loc[exonic_segments_intersection[12]==exonic_segments_intersection['segment_len']].reset_index(drop=True)
    
    exonic_segments_intersection = exonic_segments_intersection[[0,1,2,5,'segment_len',9]].drop_duplicates().reset_index(drop=True)

    # calculate how many transcripts there are in every gene, and also add gene_ids
    gr = exons_sel_df[['transcript_id','gene_id']].drop_duplicates().reset_index(drop=True)
    gr['c']=1
    gr = pd.merge(gr.drop(['c'],axis=1),gr.groupby('gene_id').agg({'c':sum}).reset_index(),how='left',on='gene_id')

    exonic_segments_intersection = pd.merge(exonic_segments_intersection.rename(columns={9:'transcript_id'}),gr,how='left',on='transcript_id')

    if remove_segments_associated_with_multiple_genes_on_same_strand:
        # remove segments that are associated with more than ONE gene ON THE SAME STRAND
        gr = exonic_segments_intersection[[0,1,2,5,'segment_len','gene_id']].drop_duplicates().reset_index(drop=True) 
        gr['t']=1
        gr = gr.groupby([0,1,2,5,'segment_len']).agg({'t':sum}).reset_index()
        exonic_segments_intersection = pd.merge(exonic_segments_intersection,gr.loc[gr['t']==1][[0,1,2,5]].reset_index(drop=True),how='inner',on=[0,1,2,5])
        print(f"Removed segments that are associated with more than one gene on the same strand. Retained {len(exonic_segments_intersection[[0,1,2,5]].drop_duplicates())} segments.\n")

    print(f"Identified {len(exonic_segments_intersection[[0,1,2,5]].drop_duplicates())} exonic segments corresponding to \n"+\
        f"{len(exonic_segments_intersection)} exons in {len(exonic_segments_intersection['transcript_id'].unique())} transcripts \n"+\
        f"in {len(exonic_segments_intersection['gene_id'].unique())} genes.\n")    
    
    # calculate in how many transcripts each segment is present
    exonic_segments_intersection['t']=1
    gr = exonic_segments_intersection.groupby([0,1,2,5,'segment_len','gene_id','c']).agg({'t':sum}).reset_index()
    
    gr = pd.merge(gr,exons_df[['gene_id','gene_name','gene_type']].drop_duplicates(),how='inner',on='gene_id')
    
    exonic_elements = gr.loc[gr['t']>gr['c']*f].reset_index(drop=True).copy()
    print(f"\nAfter requiring that each segment is present in at least {f*100}% of transcripts, \n"+\
          f"retained are {len(exonic_elements[[0,1,2,5]].drop_duplicates())} exonic segments \n"+\
        f"in {len(exonic_elements['gene_id'].unique())} genes.")    
    
    print("\nFormatting the final GTF file with gene-transcript-exon structure. \n"+\
          "Each exon corresponds to a selected exonic segment, \n"+
          "and gene coordinates are defined as the min start and max end of the exonic segments associated with the gene. \n"+\
            "Every gene has just one transcript, so gene and transcript elements are identical. \n"+\
                f"The output GTF file will be sorted by gene_id and then by genomic coordinates.\n")
    ###
    # make final gtf file
    ###
    exonic_elements = exonic_elements.rename(columns={1:3,2:4,5:6})
    exonic_elements[1] = 'CUSTOM'
    exonic_elements[2] = 'exon'
    exonic_elements[5] = '.'
    exonic_elements[7] = '.'
    exonic_elements = pd.concat([exonic_elements.loc[exonic_elements[6]=='+'].sort_values([0,3],ascending=[True,True]),
                                exonic_elements.loc[exonic_elements[6]=='-'].sort_values([0,3],ascending=[True,False])]).reset_index(drop=True)
    exonic_elements['d'] = 1
    exonic_elements['exon_number'] = exonic_elements[['gene_id','d']].groupby('gene_id').cumsum()
    exonic_elements['exon_id'] = exonic_elements['gene_id']+':exon_'+exonic_elements['exon_number'].astype('str')
    exonic_elements[8] = 'gene_id "'+exonic_elements['gene_id']+'"; transcript_id "'+exonic_elements['gene_id']+'";'+\
    ' gene_type "'+exonic_elements['gene_type']+'"; gene_name "'+exonic_elements['gene_name']+'"; transcript_type "'+exonic_elements['gene_type']+'"; transcript_name "'+exonic_elements['gene_name']+'"; '+\
    'exon_number "'+exonic_elements['exon_number'].astype('str')+'"; exon_id "'+exonic_elements['exon_id']+'"'
    exonic_elements['order'] = 3
    exonic_elements_gtf = exonic_elements[list(range(0,9))+['gene_id','order']].copy()

    exonic_elements_transcripts = exonic_elements.groupby([0,1,2,5,6,7,'gene_id','gene_name','gene_type']).agg({3:min,4:max}).reset_index()
    exonic_elements_transcripts[2] = 'transcript'
    exonic_elements_transcripts[8] = 'gene_id "'+exonic_elements_transcripts['gene_id']+'"; transcript_id "'+exonic_elements_transcripts['gene_id']+'";'+\
    ' gene_type "'+exonic_elements_transcripts['gene_type']+'"; gene_name "'+exonic_elements_transcripts['gene_name']+'"; transcript_type "'+exonic_elements_transcripts['gene_type']+\
    '"; transcript_name "'+exonic_elements_transcripts['gene_name']+'"'
    exonic_elements_transcripts['order'] = 2
    exonic_elements_transcripts_gtf = exonic_elements_transcripts[list(range(0,9))+['gene_id','order']]

    exonic_elements_genes = exonic_elements.groupby([0,1,2,5,6,7,'gene_id','gene_name','gene_type']).agg({3:min,4:max}).reset_index()
    exonic_elements_genes[2] = 'gene'
    exonic_elements_genes[8] = 'gene_id "'+exonic_elements_genes['gene_id']+'"; gene_type "'+exonic_elements_genes['gene_type']+'"; gene_name "'+exonic_elements_genes['gene_name']+'"'
    exonic_elements_genes['order'] = 1
    exonic_elements_genes_gtf = exonic_elements_genes[list(range(0,9))+['gene_id','order']]

    gene_expression_gtf = pd.concat([exonic_elements_genes_gtf,exonic_elements_transcripts_gtf,exonic_elements_gtf]).sort_values(['gene_id','order']).drop(['gene_id','order'],axis=1)
    gene_expression_gtf = gene_expression_gtf.reset_index(drop=True)
    
    Path(out_gtf_path).parent.mkdir(parents=True, exist_ok=True)
    gene_expression_gtf.to_csv(out_gtf_path, sep=str('\t'),header=False,index=None,quoting=csv.QUOTE_NONE)
    print("Final gtf file written to "+out_gtf_path+"\n")
    return gene_expression_gtf

def extract_exonic_segments_from_gtfDF_and_make_bed(
        gtf_df:pd.DataFrame,
        out_bed_path: str,
        temp_dir=None
) -> pd.DataFrame:
    """
    Get full exons in .bed format and write them to a file
    
    Parameters
    ----------
    gtf_df : pd.DataFrame
        pandas DataFrame from a GTF file.
        The output of `parse_gtf_attributes_into_pd_dataframes()` can be used as input for this function.
    out_bed_path : str
        Path to output BED file that will be written with the selected exons.
    temp_dir : str, optional
        Path to temporary directory for intermediate files. 
        If None, a 'temp' directory will be created in the current working directory or $TMPDIR environment variable if present. Default is None. 
    
    Returns
    -------
    ann_exons_df : pd.DataFrame
        bed-formatted DataFrame containing annotated exons with columns ['chr', 'start', 'end', 'name', 'score', 'strand'].
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
    print(f"Checked bedtools installation and created temporary directory at {temp_dir}...\n")
    
    ann_exons_df = gtf_df.loc[gtf_df[2]=='exon'].reset_index(drop=True)
    print(f"Start with {len(ann_exons_df)} exons in the provided gtf_df dataframe.\n")
    # just get unique exons based on genomic coordinates and strand, 
    # ignoring transcript_id and gene_id (there can be multiple transcripts with the same exon)
    ann_exons_df = ann_exons_df[[0,3,4,6]].drop_duplicates().reset_index(drop=True) 
    print(f"Identified {len(ann_exons_df)} unique exonic segements.\n")
    
    ann_exons_df['name'] = ann_exons_df.index # just add arbitrary integer name for each exon, since bed format requires a name column
    ann_exons_df['start'] = ann_exons_df[3]-1
    ann_exons_df['score'] = 0 # also arbitrary score column, since bed format requires a score column
    
    nonsorted_SEGMENTS_bed = os.path.join(temp_dir, 'ann_exons_df.bed')
    ann_exons_df[[0,'start',4,'name','score',6]].to_csv(nonsorted_SEGMENTS_bed, sep=str('\t'),header=False,index=None,quoting=csv.QUOTE_NONE)

    Path(out_bed_path).parent.mkdir(parents=True, exist_ok=True)
    command = 'bedtools sort -i '+nonsorted_SEGMENTS_bed+' > '+out_bed_path
    out = subprocess.check_output(command, shell=True)
    print(f"successfully wrote BED file with {len(ann_exons_df)} exonic elements to {out_bed_path}\n")
    return ann_exons_df

def genbank_to_fasta_and_gtf(input_gb_file_path:str, 
                             out_fasta_file_path:str, 
                             out_gtf_file_path:str,
                             chromosome_name:str=None):
    """
    Parses a GenBank file and exports its sequence to a FASTA file
    and its features to a GTF file.
    Parameters
    ----------
    input_gb_file_path : str
        Path to input GenBank file.
    out_fasta_file_path : str
        Path to output FASTA file.
    out_gtf_file_path : str
        Path to output GTF file.
    chromosome_name : str, optional
        Optional name for the chromosome/sequence in the output files. If not provided, the GenBank record ID will be used as the chromosome name.
    Returns
    -------
    Notes
    -----
    """
    with open(out_fasta_file_path, "w") as out_fasta, open(out_gtf_file_path, "w") as out_gtf:
        # Parse GenBank records
        for record in SeqIO.parse(input_gb_file_path, "genbank"):
            
            # Use provided chromosome name, otherwise fallback to GenBank record ID
            seq_id = chromosome_name if chromosome_name else record.id
            
            # 1. Write Sequence to FASTA
            out_fasta.write(f">{seq_id} {record.description}\n")
            seq = str(record.seq)
            # Wrap FASTA sequence to 80 characters per line
            for i in range(0, len(seq), 80):
                out_fasta.write(f"{seq[i:i+80]}\n")
            
            # 2. Write Features to GTF
            for feature in record.features:
                # Skip the 'source' feature (it spans the entire sequence length)
                if feature.type == "source":
                    continue
                
                attributes = []
                
                # Standardize GTF mandatory attributes: gene_id and transcript_id
                gene_id = None
                if "gene" in feature.qualifiers:
                    gene_id = feature.qualifiers["gene"][0]
                elif "locus_tag" in feature.qualifiers:
                    gene_id = feature.qualifiers["locus_tag"][0]
                else:
                    gene_id = f"{feature.type}_{feature.location.start+1}_{feature.location.end}"
                attributes.append(f'gene_id "{gene_id}"')
                
                transcript_id = None
                if "transcript_id" in feature.qualifiers:
                    transcript_id = feature.qualifiers["transcript_id"][0]
                else:
                    transcript_id = gene_id
                attributes.append(f'transcript_id "{transcript_id}"')
                
                # Append all other qualifiers as GTF attributes
                for key, values in feature.qualifiers.items():
                    if key not in ["gene", "locus_tag", "transcript_id"]:
                        # GTF attributes must be double-quoted; escape internal quotes
                        val = str(values[0]).replace('"', '\\"')
                        attributes.append(f'{key} "{val}"')
                        
                attr_str = "; ".join(attributes) + ";"
                
                # Handle multi-part locations (e.g. joined exons/CDS)
                parts = feature.location.parts if hasattr(feature.location, "parts") else [feature.location]
                
                for loc in parts:
                    # Biopython uses 0-based start (inclusive) and 0-based end (exclusive).
                    # GTF requires 1-based start (inclusive) and 1-based end (inclusive).
                    start = loc.start + 1
                    end = int(loc.end)
                    
                    strand_map = {1: '+', -1: '-', 0: '.', None: '.'}
                    strand = strand_map.get(loc.strand, '.')
                    
                    # Write GTF line: seqname source feature start end score strand frame attribute
                    out_gtf.write(f"{seq_id}\tGenBank\t{feature.type}\t{start}\t{end}\t.\t{strand}\t.\t{attr_str}\n")
