"""
Tests for annotation module.
"""

import pytest
from zavolab_pyutils.annotation import parse_gtf_attributes, convert_gff_to_gtf, convert_gtf_to_gff


class TestAnnotation:
    """Test suite for annotation functions."""
    
    def test_parse_gtf_attributes_basic(self):
        """Test parsing basic GTF attributes."""
        attr_str = 'gene_id "ENSG00000223972"; transcript_id "ENST00000456328";'
        result = parse_gtf_attributes(attr_str)
        assert isinstance(result, dict)
    
    def test_convert_gff_to_gtf(self, tmp_path):
        """Test GFF to GTF conversion."""
        # Create dummy GFF file
        gff_file = tmp_path / "test.gff"
        gtf_file = tmp_path / "test.gtf"
        
        gff_file.write_text("scaffold1\tensembl\tgene\t1\t100\t.\t+\t.\tID=gene1\n")
        
        convert_gff_to_gtf(str(gff_file), str(gtf_file))
        # Check that function completes without error
    
    def test_convert_gtf_to_gff(self, tmp_path):
        """Test GTF to GFF conversion."""
        # Create dummy GTF file
        gtf_file = tmp_path / "test.gtf"
        gff_file = tmp_path / "test.gff"
        
        gtf_file.write_text('scaffold1\tensembl\tgene\t1\t100\t.\t+\t.\tgene_id "gene1";\n')
        
        convert_gtf_to_gff(str(gtf_file), str(gff_file))
        # Check that function completes without error
