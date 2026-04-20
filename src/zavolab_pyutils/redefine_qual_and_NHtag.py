#!/usr/bin/env python3
"""
this is a CLI module to redefine the quality string and NH tag value in a bam file, based on the actually present alignments.
It's useful e.g. when the bam file is an output of UMI deduplication or custom filtering which resulted in 
some originally multi-mapping reads being reduced to a single alignment, but the quality string and NH tag value were not updated accordingly.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import warnings
warnings.simplefilter('ignore')

import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import HTSeq

def write_to_new_bam(cur_alignment_list, cur_NH, bam_writer, change_MAPQ, MAPQ_UM, MAPQ_MM):
    if change_MAPQ: # according to the methodology in STAR aligner, the MAPQ value is set to 255 for uniquely mapped reads
        if cur_NH == 1:
            MAPQ = MAPQ_UM
        else:
            MAPQ = MAPQ_MM
    else:
        MAPQ = None
        
    for alignment in cur_alignment_list:
        if MAPQ is not None:
            alignment.aQual = MAPQ
            
        # Safely update existing NH tag, or append it if it doesn't exist
        nh_found = False
        updated_fields = []
        for tag in alignment.optional_fields:
            if tag[0] == 'NH':
                updated_fields.append(('NH', cur_NH))
                nh_found = True
            else:
                updated_fields.append(tag)
        
        # If the NH tag wasn't in the original alignments (e.g., from minimap2), add it now
        if not nh_found:
            updated_fields.append(('NH', cur_NH))
            
        alignment.optional_fields = updated_fields
        bam_writer.write(alignment)

def main():
    """ Parse dirty bam file and redefine correctly the quality string and NH tag value. Assumes that the input bam file is sorted by read name! """
    parser = ArgumentParser(description=main.__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument("--input_bam_file", dest="input_bam_file", help="Path to the READ NAME - SORTED bam file.", required=True, metavar="FILE")
    parser.add_argument("--out_bam_file", dest="out_bam_file", help="Path to the output bam file", required=True, metavar="FILE")
    parser.add_argument("--skip_MAPQ_change", dest="change_MAPQ", action="store_false", help="Flag to prevent changing MAPQ values (By default, MAPQ is changed)")
    parser.add_argument("--MAPQ_UM", dest="MAPQ_UM", help="MAPQ value for uniquely mapped reads", metavar="INT", type=int, default=255)
    parser.add_argument("--MAPQ_MM", dest="MAPQ_MM", help="MAPQ value for multi-mapped reads", metavar="INT", type=int, default=0)    
    
    try:
        options = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(1)

    almnt_file = HTSeq.SAM_Reader(options.input_bam_file)
    bam_writer = HTSeq.BAM_Writer.from_BAM_Reader(options.out_bam_file, almnt_file)
    
    cur_name, cur_NH, cur_alignment_list = '', 0, []
    
    for almnt in almnt_file:
        read_name = almnt.read.name
        if cur_name != read_name:
            if cur_name != '':
                write_to_new_bam(cur_alignment_list, cur_NH, bam_writer, options.change_MAPQ, options.MAPQ_UM, options.MAPQ_MM)
            cur_NH = 0
            cur_alignment_list = []    
            
        cur_NH += 1
        cur_name = read_name
        cur_alignment_list.append(almnt)

    if len(cur_alignment_list) > 0:
        write_to_new_bam(cur_alignment_list, cur_NH, bam_writer, options.change_MAPQ, options.MAPQ_UM, options.MAPQ_MM)
        
    bam_writer.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(1)