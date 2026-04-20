#!/usr/bin/env python3

from __future__ import (absolute_import, division, print_function, unicode_literals)
import warnings
warnings.simplefilter('ignore')

import sys
import pysam
from argparse import ArgumentParser, RawTextHelpFormatter

def main():
    """ Normalize UMI lengths in the RX tag of a BAM file to prevent umi_tools assertion errors. """
    parser = ArgumentParser(description=main.__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--input_bam", dest="input_bam", help="Path to the input bam file", required=True)
    parser.add_argument("--output_bam", dest="output_bam", help="Path to the output bam file", required=True)
    parser.add_argument("--target_len", dest="target_len", type=int, help="Target UMI length", required=True)
    
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(1)

    # pysam natively preserves coordinate sorting when iterating and writing
    bam_in = pysam.AlignmentFile(args.input_bam, "rb")
    bam_out = pysam.AlignmentFile(args.output_bam, "wb", template=bam_in)

    for read in bam_in:
        if read.has_tag('RX'):
            umi = str(read.get_tag('RX'))
            # Truncate if too long, pad with 'N's if too short
            if len(umi) > args.target_len:
                umi = umi[:args.target_len]
            elif len(umi) < args.target_len:
                umi = umi.ljust(args.target_len, 'N')
            read.set_tag('RX', umi, value_type='Z')
        bam_out.write(read)

    bam_in.close()
    bam_out.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupt!\n")
        sys.exit(1)