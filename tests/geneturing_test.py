import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import List, Optional, Dict, Any
import os, sys

sys.path.append('/project/GCRB/Hon_lab/s440862/courses/se/MODULE_3_MATERIALS/mod3')
from geneturing import (
    exact_match,
    gene_disease_association,
    human_genome_dna_alignment,
)

class TestMetrics(unittest.TestCase):

    def test_exact_match(self):
        self.assertEqual(exact_match("apple", "apple"), 1)
        self.assertEqual(exact_match("apple", "banana"), 0)
        self.assertEqual(exact_match("Apple", "apple"), 0)

    def test_gene_disease_association(self):
        self.assertEqual(gene_disease_association("GENE1, GENE2", "GENE1, GENE2"), 1.0)
        self.assertEqual(gene_disease_association("GENE1", "GENE1, GENE2"), 0.5)
        self.assertEqual(gene_disease_association("GENE3", "GENE1, GENE2"), 0.0)
        self.assertEqual(gene_disease_association("GENE1, GENE2, GENE3", "GENE1, GENE2"), 1.0)
        self.assertEqual(gene_disease_association("", ""), 1.0) 
        self.assertEqual(gene_disease_association("GENE1", ""), 0.0) 
        self.assertEqual(gene_disease_association("", "GENE1"), 0.0) 

    def test_human_genome_dna_alignment(self):
        self.assertEqual(human_genome_dna_alignment("chr1:100-200", "chr1:100-200"), 1.0)
        self.assertEqual(human_genome_dna_alignment("chr1:300-400", "chr1:100-200"), 0.5)
        self.assertEqual(human_genome_dna_alignment("chr2:100-200", "chr1:100-200"), 0.0)
        self.assertEqual(human_genome_dna_alignment("chrX:1-10", "chrX:20-30"), 0.5)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

### python -m unittest geneturing_test.py
