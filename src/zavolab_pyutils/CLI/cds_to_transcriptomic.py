"""
CLI module to extract transcriptomic coordinates of CDS elements from a GTF file,
outputting a 0-based BED-like file (transcript_id, start, end).
"""

import argparse
import logging
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
    
    # Rename standard GTF positional columns for readability
    exons = exons.rename(columns={0: 'chrom', 3: 'start_gtf', 4: 'end_gtf', 6: 'strand'})
    
    exons['start_gtf'] = exons['start_gtf'].astype(int)
    exons['end_gtf'] = exons['end_gtf'].astype(int)
    
    # Sort exons structurally: Ascending genomic for +, Descending genomic for -
    exons_plus = exons[exons['strand'] == '+'].sort_values(['transcript_id', 'start_gtf'], ascending=[True, True])
    exons_minus = exons[exons['strand'] == '-'].sort_values(['transcript_id', 'end_gtf'], ascending=[True, False])
    
    exons_sorted = pd.concat([exons_plus, exons_minus]).reset_index(drop=True)
    
    # Cumulative lengths map genomic sequences to transcriptomic positions (0-based)
    exons_sorted['exon_length'] = exons_sorted['end_gtf'] - exons_sorted['start_gtf'] + 1
    exons_sorted['tr_end'] = exons_sorted.groupby('transcript_id')['exon_length'].cumsum()
    exons_sorted['tr_start'] = exons_sorted['tr_end'] - exons_sorted['exon_length']
    
    return exons_sorted

def map_cds_to_transcriptomic(cds_df: pd.DataFrame, exons_sorted: pd.DataFrame, aggregate: bool = True, include_stop_codon: bool = False) -> pd.DataFrame:
    """Maps genomic CDS elements to transcriptomic coordinates based on exon structure."""
    logging.info("Mapping genomic CDS coordinates to transcriptomic space...")
    
    # Rename CDS positional columns
    cds = cds_df.rename(columns={0: 'chrom', 3: 'start_gtf', 4: 'end_gtf', 6: 'strand'}).copy()
    cds['start_gtf'] = cds['start_gtf'].astype(int)
    cds['end_gtf'] = cds['end_gtf'].astype(int)
    
    # Extract transcript_id for CDS elements
    cds['transcript_id'] = cds[8].str.extract(r'transcript_id "([^"]+)"', expand=False)
    
    # Drop rows missing transcript_id to prevent merge errors
    missing_tr = cds['transcript_id'].isna().sum()
    if missing_tr > 0:
        logging.warning(f"Dropping {missing_tr} CDS entries that lack a 'transcript_id' attribute.")
        cds = cds.dropna(subset=['transcript_id'])
        
    # Inner merge pairs CDS regions with their corresponding exons
    merged = pd.merge(
        cds[['transcript_id', 'start_gtf', 'end_gtf', 'strand']],
        exons_sorted[['transcript_id', 'start_gtf', 'end_gtf', 'tr_start', 'strand']],
        on=['transcript_id', 'strand'],
        how='inner',
        suffixes=('_cds', '_exon')
    )
    
    # A CDS segment strictly belongs to the exon that fully contains it 
    # (GTF format guarantees CDS features are bounded by exon features)
    overlaps = merged[
        (merged['start_gtf_cds'] >= merged['start_gtf_exon']) &
        (merged['end_gtf_cds'] <= merged['end_gtf_exon'])
    ].copy()
    
    if overlaps.empty:
        logging.warning("No overlapping CDS and Exon regions found. Ensure GTF CDS elements match internal exon boundaries.")
        return pd.DataFrame(columns=['transcript_id', 'start', 'end'])
        
    # Compute exact transcriptomic coordinates (0-based start, half-open interval)
    plus_mask = overlaps['strand'] == '+'
    overlaps.loc[plus_mask, 'cds_tr_start'] = overlaps.loc[plus_mask, 'tr_start'] + (overlaps.loc[plus_mask, 'start_gtf_cds'] - overlaps.loc[plus_mask, 'start_gtf_exon'])
    overlaps.loc[plus_mask, 'cds_tr_end'] = overlaps.loc[plus_mask, 'tr_start'] + (overlaps.loc[plus_mask, 'end_gtf_cds'] - overlaps.loc[plus_mask, 'start_gtf_exon']) + 1
    
    minus_mask = overlaps['strand'] == '-'
    overlaps.loc[minus_mask, 'cds_tr_start'] = overlaps.loc[minus_mask, 'tr_start'] + (overlaps.loc[minus_mask, 'end_gtf_exon'] - overlaps.loc[minus_mask, 'end_gtf_cds'])
    overlaps.loc[minus_mask, 'cds_tr_end'] = overlaps.loc[minus_mask, 'tr_start'] + (overlaps.loc[minus_mask, 'end_gtf_exon'] - overlaps.loc[minus_mask, 'start_gtf_cds']) + 1
    
    overlaps['cds_tr_start'] = overlaps['cds_tr_start'].astype(int)
    overlaps['cds_tr_end'] = overlaps['cds_tr_end'].astype(int)
    
    # Calculate total transcript lengths to ensure we don't extend past the transcript bound
    tr_lens = exons_sorted.groupby('transcript_id')['tr_end'].max().reset_index(name='tr_len')

    if aggregate:
        # Group by transcript_id to define the contiguous ORF in transcriptomic coordinates
        logging.info("Aggregating contiguous CDS elements into full ORF transcriptomic coordinates...")
        output_df = overlaps.groupby('transcript_id').agg(
            start=('cds_tr_start', 'min'),
            end=('cds_tr_end', 'max')
        ).reset_index()

        if include_stop_codon:
            logging.info("Extending 3' end by 3nt to include the STOP codon...")
            output_df = output_df.merge(tr_lens, on='transcript_id', how='left')
            # Add 3 nt, bounded to a maximum of the full transcript length
            output_df['end'] = (output_df['end'] + 3).clip(upper=output_df['tr_len'])
            output_df = output_df.drop(columns=['tr_len'])

    else:
        # Output individual CDS segments
        logging.info("Outputting individual CDS segments in transcriptomic coordinates...")
        output_df = overlaps[['transcript_id', 'cds_tr_start', 'cds_tr_end']].rename(
            columns={'cds_tr_start': 'start', 'cds_tr_end': 'end'}
        ).sort_values(['transcript_id', 'start']).reset_index(drop=True)

        if include_stop_codon:
            logging.info("Extending 3' end of the terminal CDS segment by 3nt to include the STOP codon...")
            # Left merge transcript length first to prevent indices misalignment
            output_df = output_df.merge(tr_lens, on='transcript_id', how='left')
            
            # Find the row containing the 3' terminal segment (max 'end') for each transcript
            terminal_idx = output_df.groupby('transcript_id')['end'].idxmax()
            
            # Add 3 nt to the terminal segments only, capping by total transcript length
            output_df.loc[terminal_idx, 'end'] = (output_df.loc[terminal_idx, 'end'] + 3).clip(upper=output_df.loc[terminal_idx, 'tr_len'])
            output_df = output_df.drop(columns=['tr_len'])
        
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Extract transcriptomic coordinates of CDS elements from a GTF file.")
    parser.add_argument("-g", "--gtf", required=True, help="Input GTF file with genomic coordinates.")
    parser.add_argument("-o", "--out", required=True, help="Output tab-delimited BED-like file (transcript_id, start, end).")
    parser.add_argument("--individual_segments", action='store_true', help="Output each CDS segment individually instead of an aggregated contiguous ORF region per transcript.")
    parser.add_argument("--include_stop_codon", action='store_true', help="Include the STOP codon in the output coordinates by extending the 3' end by 3nt (caps safely at the transcript's boundary). Assumes standard GTF where CDS excludes the STOP codon.")
    
    args = parser.parse_args()
    
    try:
        logging.info(f"Parsing GTF file using annotation.py utils: {args.gtf}")
        
        # Use zavolab_pyutils parser which safely extracts GTF as well as gene/exon DataFrames
        gtf_df, _, exons = parse_gtf_attributes_into_pd_dataframes(
            args.gtf,
            extract_exon_number=False,
            extract_gene_name_in_exons=False,
            verbose=False
        )
        
        logging.info("Identifying CDS elements...")
        cds_df = gtf_df[gtf_df[2] == 'CDS'].copy()
        
        # Guard Check: Stop execution if no CDS entries are present
        if cds_df.empty:
            logging.error("No CDS elements found in the input GTF file. Exiting.")
            sys.exit(1)
            
        logging.info(f"Found {len(cds_df)} CDS entries.")
            
        # Step 1: Build relative index mapping from exons
        exons_sorted = build_transcript_coords(exons)
        
        # Step 2: Map CDS segments to transcriptomic coordinates
        mapped_cds = map_cds_to_transcriptomic(
            cds_df, 
            exons_sorted, 
            aggregate=not args.individual_segments,
            include_stop_codon=args.include_stop_codon
        )
        
        # Guard Check: Final sanity check on intersections
        if mapped_cds.empty:
            logging.error("Failed to map any CDS elements to transcriptomic coordinates. Check GTF consistency.")
            sys.exit(1)
        
        # Step 3: Write tab-delimited output
        logging.info(f"Writing mapped CDS coordinates to: {args.out}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        mapped_cds.to_csv(args.out, sep='\t', header=False, index=False)
        logging.info("Done.")
        
    except Exception as e:
        logging.error(f"Error executing script: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()