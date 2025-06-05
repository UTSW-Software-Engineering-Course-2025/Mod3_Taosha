import json

msg = {
    "genehop": {
        "system_msg": [
            {
                "role": "system",
                "content": "Hello. You are an expert in bioinformatics, and your job now is to use the NCBI Web APIs to give accurate and neat step-to-step answers to genomic questions in aspects of ['sequence gene alias', 'Disease gene location', 'SNP gene function']. \
                For example, if I ask 'What are the aliases of the gene that contains this sequnece:ATTGTGAGAGTAACCAACGTGGGGTTACGGGGGAGAATCTGGAGAGAAGAGAAGAGGTTAACAACCCTCCCACTTCCTGGCCACCCCCCTCCACCTTTTCTGGTAAGGAGCCC. Let's decompose the question to sub-questions and solve them step by step.', \
                    your answer should be: 'Answer: ['FCGR3A', 'CD16', 'FCG3', 'CD16A', 'FCGR3', 'IGFR3', 'IMD20', 'FCR-10', 'FCRIII', 'CD16-II', 'FCGRIII', 'FCRIIIA', 'FcGRIIIA']'.  \
                        If it is a question in terms of 'sequence gene alias' or 'Disease gene location', you should answer with a list of the componens;  \
                            if it is a question in terms of 'SNP gene function', you should answer with a short, summary sentence. Thanks.",
            }
        ],
        "example_msg": [
            {
                "role": "user",
                "content": "Question: What are the aliases of the gene that contains this sequnece:ACTTCCAACATGGCGGCGGCCGGGGCGGCGGTGGCGCGCAGCCCGGGAATCGGAGCGGGACCTGCGCTGAGAGCCCGGCGCTCGCCCCCGCCGCGGGCCGCACGGCTGCCGCG. Let's decompose the question to sub-questions and solve them step by step.",
            },
            {
                "role": "assistant",
                "content": json.dumps({"answer": ["QSOX2", "SOXN", "QSCN6L1"]}),
            },
            {
                "role": "user",
                "content": "Question: List chromosome locations of the genes related to Split-foot malformation with mesoaxial polydactyly. Let's decompose the question to sub-questions and solve them step by step.",
            },
            {"role": "assistant", "content": json.dumps({"answer": ["2q31.1"]})},
            {
                "role": "user",
                "content": "Question: What is the function of the gene associated with SNP rs1218214598? Let's decompose the question to sub-questions and solve them step by step.",
            },
            {"role": "assistant", "content": json.dumps({"answer": "ncRNA"})},
            {
                "role": "user",
                "content": "Question: What is the function of the gene associated with SNP rs937944577? Let's decompose the question to sub-questions and solve them step by step.",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "Answer": "This gene is a member of the intermediate filament family. Intermediate filaments are proteins which are primordial components of the cytoskeleton and nuclear envelope. The proteins encoded by the members of this gene family are evolutionarily and structurally related but have limited sequence homology, with the exception of the central rod domain. Multiple alternatively spliced transcript variants encoding different isoforms have been found for this gene."
                    }
                ),
            },
        ],
    },
    "geneturing": {
        "system_msg": [
            {
                "role": "system",
                "content": "Hello. You are an expert in bioinformatics, and your job now is to use the NCBI Web APIs to give accurate and neat answers to genomic questions in aspects of ['Gene alias', 'Gene disease association', 'Gene location', 'Human genome DNA aligment', 'Multi-species DNA aligment', 'Gene name conversion', 'Protein-coding genes', 'Gene SNP association', 'SNP location']. For example, if I ask 'What is the official gene symbol of LMP10?', your answer should be: 'Answer: PSMB10'.",
            }
        ],
        "example_msg": [
            {
                "role": "user",
                "content": "Question: What is the official gene symbol of LMP10?",
            },
            {"role": "assistant", "content": "Answer: PSMB10"},
            {
                "role": "user",
                "content": "Question: Which gene is SNP rs1217074595 associated with?",
            },
            {"role": "assistant", "content": "Answer: LINC01270"},
            {
                "role": "user",
                "content": "Question: What are genes related to Meesmann corneal dystrophy?",
            },
            {"role": "assistant", "content": "Answer: KRT12, KRT3"},
            {
                "role": "user",
                "content": "Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT",
            },
            {"role": "assistant", "content": "Answer: chr15:91950805-91950932"},
        ],
    },
}
