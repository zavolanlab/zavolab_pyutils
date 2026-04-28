import argparse
from pathlib import Path
import sys
import pysam
import pyBigWig
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

def get_rc(seq):
    return seq.translate(str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn"))[::-1]

def extract_sites_from_bw(bw_file, strand):
    sites = []
    with pyBigWig.open(bw_file) as bw:
        for chrom in bw.chroms():
            intervals = bw.intervals(chrom)
            if intervals:
                for start, end, val in intervals:
                    # Expand merged intervals into 1bp positions, if present
                    for pos in range(start, end):
                        sites.append({'chrom': chrom, 'pos': pos, 'strand': strand, 'score': val})
    return sites

def plot_cs_motifs():
    parser = argparse.ArgumentParser(description="Generate motif meta-plots around positions.")
    parser.add_argument("--bw_plus", required=True)
    parser.add_argument("--bw_minus", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--motifs", nargs='+', required=True, help="space-separated list of N-mers (e.g. AAUAAA AAGAAA)")
    parser.add_argument("--window_up", type=int, default=40)
    parser.add_argument("--window_down", type=int, default=20)
    parser.add_argument("--anchor", choices=['start', 'end', 'center'], default='start', help="position within the motif occurrence for relative plotting")
    parser.add_argument("--weighting", choices=['count', 'uniform'], default='count', 
                        help="Whether to weight positional frequencies by the number of motif occurrences in the window (uniform) or treat each site equally (count)")
    parser.add_argument("--bins", type=int, default=1, help="Number of quantile bins based on score in the .bigwig files, default=1 (no binning)")
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    # 1. Load Data
    sites = extract_sites_from_bw(args.bw_plus, '+') + extract_sites_from_bw(args.bw_minus, '-')
    df_sites = pd.DataFrame(sites)
    
    if df_sites.empty:
        print("[WARNING] No positions found in BigWigs.")
        sys.exit(0)

    # 2. Quantile Binning based on score
    try:
        df_sites['bin'] = pd.qcut(df_sites['score'], q=args.bins, labels=False, duplicates='drop')
    except ValueError:
        df_sites['bin'] = 0  # Fallback if all scores are identical
        print("[WARNING] all positions have identical scores in BigWigs.")

    unique_bins = sorted(df_sites['bin'].unique())
    fasta = pysam.FastaFile(args.fasta)
    
    sns.set_theme(style="whitegrid")

    for current_bin in unique_bins:
        bin_sites = df_sites[df_sites['bin'] == current_bin]
        total_sites = len(bin_sites)
        
        count_data = []
        pos_data = []
        sites_with_motif = {m: 0 for m in args.motifs} # initialize count of sites with at least one motif occurrence
        WARNING_casted = False
        for _, row in bin_sites.iterrows():
            chrom, pos, strand = row['chrom'], row['pos'], row['strand']
            
            try:
                chrom_len = fasta.get_reference_length(chrom)
            except KeyError:
                if not WARNING_casted:
                    print( f"[WARNING] Chromosome {chrom} not found in FASTA. Skipping position {chrom}:{pos}. Further warngings are suppressed to avoid flooding the output.")
                    WARNING_casted = True
                continue

            # Fetch sequence (strand aware)
            if strand == '+':
                start = pos - args.window_up
                end = pos + args.window_down + 1
            else:
                start = pos - args.window_down
                end = pos + args.window_up + 1
                
            # Handle out-of-bounds at chromosome ends by padding with Ns to preserve coordinates
            pad_left = max(0, -start)
            pad_right = max(0, end - chrom_len)
            
            fetch_start = max(0, start)
            fetch_end = min(chrom_len, end)
            
            try:
                seq = fasta.fetch(chrom, fetch_start, fetch_end).upper()
            except (KeyError, ValueError):
                if not WARNING_casted:
                    print( f"[WARNING] Error fetching sequence for {chrom}:{fetch_start}-{fetch_end}. Skipping. Further warngings are suppressed to avoid flooding the output.")
                    WARNING_casted = True
                continue
                
            seq = ("N" * pad_left) + seq + ("N" * pad_right)
            if strand == '-':
                seq = get_rc(seq) # we search for motifs on the strand-specific sequence, so reverse complement if on minus strand

            # Search Motifs
            for motif in args.motifs:
                motif_dna = motif.upper().replace('U', 'T')
                L = len(motif_dna)
                
                # Overlapping regex search
                matches = [m.start() for m in re.finditer(f'(?=({motif_dna}))', seq)]
                n_matches = len(matches)
                
                count_data.append({'motif': motif, 'count': n_matches})
                
                if n_matches > 0:
                    sites_with_motif[motif] += 1
                    weight = 1.0 if args.weighting == 'count' else (1.0 / n_matches)
                    
                    for match_idx in matches:
                        if args.anchor == 'start':
                            anchor_idx = match_idx
                        elif args.anchor == 'end':
                            anchor_idx = match_idx + L - 1
                        else: # center
                            anchor_idx = match_idx + (L // 2)
                            
                        rel_pos = anchor_idx - args.window_up
                        pos_data.append({'motif': motif, 'rel_pos': rel_pos, 'weight': weight})

        df_counts = pd.DataFrame(count_data)
        df_pos = pd.DataFrame(pos_data)
        
        prefix = f"{args.out_prefix}_bin{current_bin+1}of{len(unique_bins)}"
        dir_path = Path(prefix).parent
        dir_path.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # PLOT 1: Motif Instance Counts (Histogram)
        # ---------------------------------------------------------
        plt.figure(figsize=(8, 5))
        # Calculate proportion of sites having exactly K occurrences
        count_props = df_counts.groupby(['motif', 'count']).size() / total_sites
        count_props = count_props.reset_index(name='proportion')
        sns.barplot(data=count_props, x='count', y='proportion', hue='motif')
        plt.title(f"Motif Occurrences per Window\n(Bin {current_bin+1}, n={total_sites})")
        plt.xlabel("Number of occurrences in window")
        plt.ylabel("Proportion of sites")
        plt.tight_layout()
        plt.savefig(f"{prefix}_counts.pdf")
        plt.close()

        if df_pos.empty:
            continue

        # ---------------------------------------------------------
        # Helper to plot Positional Frequencies
        # ---------------------------------------------------------
        def plot_positional(denominator_dict, suffix, title_cond):
            freq_data = []
            for m in args.motifs:
                m_data = df_pos[df_pos['motif'] == m]
                denom = denominator_dict.get(m, total_sites)
                if denom == 0: continue
                
                # Sum weights per relative position
                pos_sums = m_data.groupby('rel_pos')['weight'].sum()
                for p, w in pos_sums.items():
                    freq_data.append({'motif': m, 'rel_pos': p, 'freq': w / denom})
                    
            if not freq_data: return
            
            df_plot = pd.DataFrame(freq_data)
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_plot, x='rel_pos', y='freq', hue='motif', drawstyle='steps-mid')
            plt.axvline(0, color='black', linestyle='--', alpha=0.5, label='Cleavage Site')
            plt.title(f"Positional Frequency of Motifs ({title_cond})\nBin {current_bin+1}")
            plt.xlabel(f"Position relative to the site ({args.anchor})")
            plt.ylabel("Proportion")
            plt.xlim(-args.window_up, args.window_down)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{prefix}_{suffix}.pdf")
            plt.close()

        # PLOT 2: Positional (Condition 1: Only >=1 occurrences)
        plot_positional(sites_with_motif, "pos_conditional", "Given present in window")

        # PLOT 3: Positional (Condition 2: All sites)
        all_sites_denom = {m: total_sites for m in args.motifs}
        plot_positional(all_sites_denom, "pos_all_sites", "Across all sites")

if __name__ == "__main__":
    plot_cs_motifs()