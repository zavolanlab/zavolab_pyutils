"""
CLI module to transform transcriptomic coordinates (from a 3-column BED file) 
to genomic coordinates (based on a GTF file) and output an IGV-compatible sorted GTF.
"""

import argparse
import logging
import os
import csv
import subprocess
import sys
from pathlib import Path
import pandas as pd

# Import the existing robust parser from zavolab_pyutils
from zavolab_pyutils.general.annotation import parse_gtf_attributes_into_pd_dataframes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_transcript_coords(exons: pd.DataFrame) -> pd.DataFrame:
    """Calculates relative transcriptomic coordinates for each exon."""
    logging.info("Building transcriptomic coordinates for exons...")
    
    exons = exons.copy()
    
    # Rename standard GTF positional columns for readability based on output from annotation.py
    exons = exons.rename(columns={0: 'chrom', 3: 'start_gtf', 4: 'end_gtf', 6: 'strand'})
    
    exons['start_gtf'] = exons['start_gtf'].astype(int)
    exons['end_gtf'] = exons['end_gtf'].astype(int)
    
    # Sort exons structurally: Ascending genomic for +, Descending genomic for -
    exons_plus = exons[exons['strand'] == '+'].sort_values(['transcript_id', 'start_gtf'], ascending=[True, True])
    exons_minus = exons[exons['strand'] == '-'].sort_values(['transcript_id', 'end_gtf'], ascending=[True, False])
    
    exons_sorted = pd.concat([exons_plus, exons_minus]).reset_index(drop=True)
    
    # Cumulative lengths map genomic sequences to transcriptomic positions
    exons_sorted['exon_length'] = exons_sorted['end_gtf'] - exons_sorted['start_gtf'] + 1
    exons_sorted['tr_end'] = exons_sorted.groupby('transcript_id')['exon_length'].cumsum()
    exons_sorted['tr_start'] = exons_sorted['tr_end'] - exons_sorted['exon_length']
    
    # Fallback to calculate exon_number if it failed to extract during parsing
    if 'exon_number' not in exons_sorted.columns or exons_sorted['exon_number'].isnull().any():
        exons_sorted['exon_number'] = exons_sorted.groupby('transcript_id').cumcount() + 1
        
    return exons_sorted

def map_regions_to_genomic(bed_df: pd.DataFrame, exons_sorted: pd.DataFrame) -> pd.DataFrame:
    """Maps requested transcriptomic BED ranges to genomic overlaps using the exons."""
    logging.info("Mapping transcriptomic coordinates to genomic coordinates...")
    
    bed_cols = len(bed_df.columns)
    if bed_cols < 3:
        raise ValueError("Input BED file must have at least 3 columns: transcript_id, start, end")
    
    # Ensure the first three columns are named correctly; 4th column is optional element_name
    col_names = ['transcript_id', 'start_tr', 'end_tr']
    if bed_cols >= 4:
        col_names.append('element_name')
    bed_df.columns = col_names + [f"extra_{i}" for i in range(bed_cols - len(col_names))]
    
    # Validate uniqueness of element_name if provided
    if 'element_name' in bed_df.columns:
        dup_mask = bed_df['element_name'].duplicated(keep=False)
        n_non_unique_names = bed_df.loc[dup_mask, 'element_name'].nunique()
        if n_non_unique_names > 0:
            logging.error(
                f"element_name column is not unique: {n_non_unique_names} name(s) appear more than once "
                f"across {dup_mask.sum()} row(s)."
            )
            raise ValueError(
                f"element_name values must be unique across all BED rows, "
                f"but {n_non_unique_names} name(s) are duplicated."
            )
    
    # Inner merge finds all exons belonging to the requested transcripts
    merged = pd.merge(bed_df, exons_sorted, on='transcript_id', how='inner')
    
    # Overlap condition: transcriptomic segment falls within the exon's bounds
    overlaps = merged[(merged['tr_end'] > merged['start_tr']) & (merged['tr_start'] < merged['end_tr'])].copy()
    
    # Report BED rows that produced no output (absent from GTF or invalid strand/coordinates)
    id_col = 'element_name' if 'element_name' in bed_df.columns else 'transcript_id'
    bed_ids = set(bed_df[id_col])
    overlapping_ids = set(overlaps[id_col]) if not overlaps.empty else set()
    missing_ids = bed_ids - overlapping_ids
    if missing_ids:
        logging.warning(
            f"{len(missing_ids)} element(s) from the input BED produced no genomic coordinates in the output GTF "
            f"(possible causes: transcript ID absent from GTF, unsupported strand value, or coordinates out of transcript bounds). "
            f"Affected element(s): {', '.join(sorted(missing_ids))}"
        )

    if overlaps.empty:
        logging.warning("No overlapping regions found! Please ensure transcript IDs match between BED and GTF.")
        return overlaps

    # Narrow down the boundaries relative to the specific exon overlap
    overlaps['overlap_tr_start'] = overlaps[['start_tr', 'tr_start']].max(axis=1)
    overlaps['overlap_tr_end'] = overlaps[['end_tr', 'tr_end']].min(axis=1)
    
    # Compute exact genomic coordinates based on strand
    plus_mask = overlaps['strand'] == '+'
    overlaps.loc[plus_mask, 'gen_start'] = overlaps.loc[plus_mask, 'start_gtf'] + (overlaps.loc[plus_mask, 'overlap_tr_start'] - overlaps.loc[plus_mask, 'tr_start'])
    overlaps.loc[plus_mask, 'gen_end'] = overlaps.loc[plus_mask, 'start_gtf'] + (overlaps.loc[plus_mask, 'overlap_tr_end'] - overlaps.loc[plus_mask, 'tr_start']) - 1
    
    minus_mask = overlaps['strand'] == '-'
    overlaps.loc[minus_mask, 'gen_end'] = overlaps.loc[minus_mask, 'end_gtf'] - (overlaps.loc[minus_mask, 'overlap_tr_start'] - overlaps.loc[minus_mask, 'tr_start'])
    overlaps.loc[minus_mask, 'gen_start'] = overlaps.loc[minus_mask, 'end_gtf'] - (overlaps.loc[minus_mask, 'overlap_tr_end'] - overlaps.loc[minus_mask, 'tr_start']) + 1
    
    overlaps['gen_start'] = overlaps['gen_start'].astype(int)
    overlaps['gen_end'] = overlaps['gen_end'].astype(int)
    
    return overlaps

def create_output_gtf(overlaps: pd.DataFrame, output_file: str, temp_dir: str):
    """Formats overlapping regions into standard GTF format and sorts for IGV."""
    if overlaps.empty:
        logging.info("Exiting GTF creation step due to empty overlap dataframe.")
        return

    logging.info("Generating output GTF features...")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Use element_name from BED col 4 if provided, otherwise construct a unique ID
    if 'element_name' in overlaps.columns:
        overlaps['new_id'] = overlaps['element_name'].astype(str)
    else:
        overlaps['new_id'] = overlaps['transcript_id'] + "_" + overlaps['start_tr'].astype(str) + "_" + overlaps['end_tr'].astype(str)
    
    overlaps['source_out'] = 'CUSTOM'
    overlaps['score_out'] = '.'
    overlaps['frame_out'] = '.'
    
    # Construct attributes 
    overlaps['gene_attrs'] = (
        'gene_id "' + overlaps['new_id'] + '"; ' +
        'transcript_id "' + overlaps['new_id'] + '"; ' +
        'gene_type "' + overlaps['gene_type'] + '"; ' +
        'gene_name "' + overlaps['gene_name'] + '";'
    )
    overlaps['exon_attrs'] = overlaps['gene_attrs'] + ' exon_number "' + overlaps['exon_number'].astype(str) + '";'
    
    # 1. Exon rows
    gtf_exons = overlaps[['chrom', 'source_out', 'gen_start', 'gen_end', 'score_out', 'strand', 'frame_out', 'exon_attrs']].copy()
    gtf_exons.insert(2, 'feature', 'exon')
    gtf_exons.columns = range(9)
    
    # 2. Gene rows (aggregate min start and max end across all exons)
    gtf_genes = overlaps[['chrom', 'source_out', 'gen_start', 'gen_end', 'score_out', 'strand', 'frame_out', 'gene_attrs']].copy()
    gtf_genes.insert(2, 'feature', 'gene')
    gtf_genes = gtf_genes.groupby(['chrom', 'source_out', 'feature', 'score_out', 'strand', 'frame_out', 'gene_attrs']).agg({'gen_start': 'min', 'gen_end': 'max'}).reset_index()
    gtf_genes = gtf_genes[['chrom', 'source_out', 'feature', 'gen_start', 'gen_end', 'score_out', 'strand', 'frame_out', 'gene_attrs']]
    gtf_genes.columns = range(9)
    
    # 3. Transcript rows (matches gene boundaries but marked 'transcript')
    gtf_transcripts = gtf_genes.copy()
    gtf_transcripts[2] = 'transcript'
    
    final_gtf = pd.concat([gtf_genes, gtf_transcripts, gtf_exons])
    
    unsorted_gtf_path = os.path.join(temp_dir, 'unsorted_mapped.gtf')
    final_gtf.to_csv(unsorted_gtf_path, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
    
    # 4. Sort GTF using standard bash sort
    logging.info("Sorting mapped GTF for IGV compatibility using bash sort...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    command = f"sort -k1,1 -k4,4n {unsorted_gtf_path} > {output_file}"
    subprocess.run(command, shell=True, check=True)
    
    logging.info(f"Sorted mapped GTF successfully written to {output_file}")

def dump_parsed_input_gtf(exons: pd.DataFrame, output_file: str, temp_dir: str):
    """
    Takes the natively parsed exons dataframe, constructs proper gene/transcript hierarchies,
    and outputs a pristine, IGV-compatible sorted GTF of the input annotation.
    """
    logging.info("Generating parsed, IGV-compatible version of the Input GTF...")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    exons_out = exons.copy()
    exons_out[3] = exons_out[3].astype(int)
    exons_out[4] = exons_out[4].astype(int)
    
    # Ensure missing strings are handled cleanly
    exons_out['gene_id'] = exons_out['gene_id'].fillna('unknown_gene')
    exons_out['transcript_id'] = exons_out['transcript_id'].fillna('unknown_transcript')
    
    # Construct attributes
    exons_out['gene_attrs'] = (
        'gene_id "' + exons_out['gene_id'].astype(str) + '"; ' +
        'gene_type "' + exons_out['gene_type'].astype(str) + '"; ' +
        'gene_name "' + exons_out['gene_name'].astype(str) + '";'
    )
    exons_out['transcript_attrs'] = exons_out['gene_attrs'] + ' transcript_id "' + exons_out['transcript_id'].astype(str) + '";'
    exons_out['exon_attrs'] = exons_out['transcript_attrs'] + ' exon_number "' + exons_out['exon_number'].astype(str) + '";'
    
    # 1. Exon rows
    gtf_exons = exons_out[[0, 1, 3, 4, 5, 6, 7, 'exon_attrs']].copy()
    gtf_exons.insert(2, 'feature', 'exon')
    gtf_exons.columns = range(9)
    
    # 2. Transcript rows
    gtf_transcripts = exons_out[[0, 1, 3, 4, 5, 6, 7, 'transcript_id', 'transcript_attrs']].copy()
    gtf_transcripts.insert(2, 'feature', 'transcript')
    gtf_transcripts = gtf_transcripts.groupby([0, 1, 'feature', 5, 6, 7, 'transcript_id', 'transcript_attrs']).agg({3: 'min', 4: 'max'}).reset_index()
    gtf_transcripts = gtf_transcripts[[0, 1, 'feature', 3, 4, 5, 6, 7, 'transcript_attrs']]
    gtf_transcripts.columns = range(9)
    
    # 3. Gene rows
    gtf_genes = exons_out[[0, 1, 3, 4, 5, 6, 7, 'gene_id', 'gene_attrs']].copy()
    gtf_genes.insert(2, 'feature', 'gene')
    gtf_genes = gtf_genes.groupby([0, 1, 'feature', 5, 6, 7, 'gene_id', 'gene_attrs']).agg({3: 'min', 4: 'max'}).reset_index()
    gtf_genes = gtf_genes[[0, 1, 'feature', 3, 4, 5, 6, 7, 'gene_attrs']]
    gtf_genes.columns = range(9)
    
    final_input_gtf = pd.concat([gtf_genes, gtf_transcripts, gtf_exons])
    
    unsorted_input_path = os.path.join(temp_dir, 'unsorted_rebuilt_input.gtf')
    final_input_gtf.to_csv(unsorted_input_path, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE)
    
    # 4. Sort GTF
    logging.info("Sorting input GTF for IGV compatibility using bash sort...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    command = f"sort -k1,1 -k4,4n {unsorted_input_path} > {output_file}"
    subprocess.run(command, shell=True, check=True)
    
    logging.info(f"Sorted input GTF successfully written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Map transcriptomic BED coordinates to genomic coordinates and create an IGV-compatible sorted GTF.")
    parser.add_argument("-b", "--bed", required=True, help="Input BED file with transcriptomic coordinates (transcript_id, start, end).")
    parser.add_argument("-g", "--gtf", required=True, help="Input GTF file with genomic coordinates.")
    parser.add_argument("-o", "--out", required=True, help="Output sorted GTF file containing the mapped regions.")
    parser.add_argument("--out_input_gtf", required=False, help="Optional: Output path to dump a cleanly formatted, IGV-compatible sorted GTF of the input annotation itself.")
    parser.add_argument("-t", "--tempdir", default="./temp", help="Temporary directory for intermediate files.")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Parse GTF using zavolab_pyutils existing tools
        logging.info(f"Parsing GTF file using annotation.py utils: {args.gtf}")
        _, _, exons = parse_gtf_attributes_into_pd_dataframes(
            args.gtf,
            extract_exon_number=True,
            extract_gene_name_in_exons=True,
            verbose=False
        )
        
        # Ensure fallbacks for optional attributes if they weren't natively present in the GTF
        if 'gene_name' not in exons.columns:
            exons['gene_name'] = exons['gene_id']
        if 'gene_type' not in exons.columns:
            exons['gene_type'] = 'unknown'
            
        exons['gene_name'] = exons['gene_name'].fillna(exons['gene_id'])
        exons['gene_type'] = exons['gene_type'].fillna('unknown')
        
        # Step 2: (Optional) Dump the parsed input GTF immediately
        if args.out_input_gtf:
            dump_parsed_input_gtf(exons, args.out_input_gtf, args.tempdir)
            
        # Step 3: Index mapping
        exons_sorted = build_transcript_coords(exons)
        
        # Step 4: Load query regions
        logging.info(f"Parsing BED file: {args.bed}")
        bed_df = pd.read_csv(args.bed, sep='\t', header=None)
        
        # Step 5: Map and build outputs
        overlaps = map_regions_to_genomic(bed_df, exons_sorted)
        create_output_gtf(overlaps, args.out, args.tempdir)
        
    except Exception as e:
        logging.error(f"Error executing script: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()