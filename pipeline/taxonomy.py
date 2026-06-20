"""Medical taxonomy for classifying open questions.

Three-level hierarchy adapted from MeSH + clinical research domains.
"""

TAXONOMY = {
    "Clinical Medicine": {
        "Diagnosis & Screening": [
            "biomarker discovery", "imaging diagnostics", "point-of-care testing",
            "differential diagnosis", "screening guidelines", "diagnostic accuracy",
        ],
        "Treatment & Therapeutics": [
            "drug efficacy", "combination therapy", "treatment resistance",
            "optimal dosing", "treatment duration", "comparative effectiveness",
        ],
        "Prognosis & Outcomes": [
            "survival prediction", "risk stratification", "long-term outcomes",
            "quality of life", "patient-reported outcomes",
        ],
        "Patient Safety": [
            "adverse events", "medication errors", "hospital-acquired infections",
            "surgical complications", "polypharmacy",
        ],
        "Pediatrics": [
            "developmental disorders", "pediatric dosing", "neonatal care",
            "childhood diseases", "growth disorders",
        ],
        "Geriatrics": [
            "frailty", "multimorbidity", "cognitive decline",
            "falls prevention", "end-of-life care",
        ],
    },
    "Oncology": {
        "Cancer Biology": [
            "tumor microenvironment", "metastasis mechanisms", "cancer stem cells",
            "tumor heterogeneity", "angiogenesis",
        ],
        "Cancer Diagnosis": [
            "early detection", "liquid biopsy", "tumor markers",
            "staging accuracy", "minimal residual disease",
        ],
        "Cancer Therapeutics": [
            "targeted therapy", "chemoresistance", "radiation optimization",
            "combination immunotherapy", "CAR-T cell therapy",
        ],
        "Cancer Immunotherapy": [
            "checkpoint inhibitors", "neoantigen prediction", "immune evasion",
            "biomarkers for response", "toxicity management",
        ],
        "Cancer Epidemiology": [
            "risk factors", "cancer prevention", "survivorship",
            "disparities in outcomes", "environmental carcinogens",
        ],
    },
    "Neuroscience & Psychiatry": {
        "Neurodegeneration": [
            "Alzheimer's mechanisms", "Parkinson's progression", "ALS pathogenesis",
            "prion diseases", "neuroprotection", "tau pathology",
        ],
        "Mental Health": [
            "depression mechanisms", "schizophrenia etiology", "PTSD treatment",
            "anxiety disorders", "addiction neurobiology", "suicide prevention",
        ],
        "Neurodevelopment": [
            "autism spectrum", "ADHD mechanisms", "brain plasticity",
            "learning disabilities", "fetal brain development",
        ],
        "Pain & Anesthesia": [
            "chronic pain mechanisms", "opioid alternatives", "neuropathic pain",
            "anesthesia awareness", "pain biomarkers",
        ],
        "Brain-Computer Interfaces": [
            "neural decoding", "brain stimulation", "neuroprosthetics",
            "consciousness measurement",
        ],
    },
    "Infectious Disease & Immunology": {
        "Antimicrobial Resistance": [
            "resistance mechanisms", "novel antibiotics", "stewardship strategies",
            "phage therapy", "drug-resistant TB",
        ],
        "Emerging Pathogens": [
            "pandemic preparedness", "zoonotic spillover", "viral evolution",
            "surveillance systems", "outbreak prediction",
        ],
        "Vaccine Development": [
            "universal flu vaccine", "HIV vaccine", "malaria vaccine",
            "adjuvant design", "mucosal immunity", "mRNA vaccines",
        ],
        "Autoimmune Disorders": [
            "disease triggers", "tolerance mechanisms", "biologics efficacy",
            "lupus pathogenesis", "rheumatoid arthritis",
        ],
        "HIV/AIDS": [
            "functional cure", "latent reservoir", "long-acting treatment",
            "prevention strategies", "comorbidities",
        ],
    },
    "Cardiovascular Medicine": {
        "Heart Failure": [
            "HFpEF mechanisms", "cardiac regeneration", "device therapy",
            "right heart failure", "cardiomyopathy genetics",
        ],
        "Atherosclerosis": [
            "plaque instability", "lipid metabolism", "inflammation pathways",
            "residual risk", "regression mechanisms",
        ],
        "Arrhythmia": [
            "atrial fibrillation mechanisms", "sudden cardiac death",
            "ablation optimization", "genetic arrhythmias",
        ],
        "Hypertension": [
            "resistant hypertension", "renal denervation",
            "target organ damage", "blood pressure variability",
        ],
        "Stroke": [
            "neuroprotection", "reperfusion injury", "secondary prevention",
            "hemorrhagic transformation", "post-stroke recovery",
        ],
    },
    "Genomics & Precision Medicine": {
        "Genetic Variants & Disease": [
            "GWAS interpretation", "rare variants", "polygenic risk scores",
            "structural variants", "non-coding mutations",
        ],
        "Pharmacogenomics": [
            "drug response prediction", "dosing algorithms",
            "population-specific variants", "implementation barriers",
        ],
        "Gene Therapy": [
            "delivery vectors", "off-target effects", "in vivo editing",
            "CRISPR optimization", "epigenome editing",
        ],
        "Multi-omics Integration": [
            "data integration methods", "single-cell multi-omics",
            "spatial transcriptomics", "proteogenomics",
        ],
        "Epigenetics": [
            "DNA methylation", "histone modifications", "epigenetic inheritance",
            "chromatin remodeling", "environmental epigenetics",
        ],
    },
    "Pharmacology & Drug Discovery": {
        "Drug Repurposing": [
            "computational screening", "off-label use evidence",
            "mechanism-based repurposing", "clinical validation",
        ],
        "Target Identification": [
            "undruggable targets", "protein-protein interactions",
            "target validation", "allosteric sites",
        ],
        "ADMET & Toxicology": [
            "hepatotoxicity prediction", "cardiotoxicity",
            "drug-drug interactions", "organ-on-chip models",
        ],
        "Clinical Trials Design": [
            "adaptive designs", "basket/umbrella trials",
            "real-world evidence", "endpoint selection",
        ],
        "Biologics & Biosimilars": [
            "antibody engineering", "biosimilar equivalence",
            "immunogenicity", "next-gen biologics",
        ],
    },
    "Public Health & Epidemiology": {
        "Health Disparities": [
            "social determinants", "access to care", "racial disparities",
            "rural health", "health literacy",
        ],
        "Environmental Health": [
            "air pollution effects", "microplastics", "climate and health",
            "occupational hazards", "water contamination",
        ],
        "Nutrition & Metabolism": [
            "obesity mechanisms", "metabolic syndrome", "gut microbiome",
            "dietary interventions", "nutritional genomics",
        ],
        "Global Health": [
            "neglected tropical diseases", "maternal mortality",
            "child malnutrition", "universal health coverage",
        ],
        "Pandemic Preparedness": [
            "early warning systems", "supply chain resilience",
            "countermeasure development", "global coordination",
        ],
    },
    "Rare & Orphan Diseases": {
        "Diagnosis & Classification": [
            "diagnostic odyssey", "phenotype delineation",
            "undiagnosed diseases", "newborn screening",
        ],
        "Therapeutic Development": [
            "orphan drug incentives", "gene therapy for rare diseases",
            "n-of-1 trials", "natural history studies",
        ],
        "Patient Registries": [
            "data sharing", "patient-reported data",
            "cross-border registries", "outcome measures",
        ],
    },
    "Surgical Sciences": {
        "Minimally Invasive Surgery": [
            "robotic surgery", "single-port techniques",
            "natural orifice surgery", "augmented reality guidance",
        ],
        "Transplantation": [
            "organ preservation", "xenotransplantation", "tolerance induction",
            "bioengineered organs", "allocation algorithms",
        ],
        "Surgical Outcomes": [
            "surgical site infections", "enhanced recovery",
            "training and simulation", "long-term follow-up",
        ],
    },
    "Medical AI & Informatics": {
        "Clinical Decision Support": [
            "alert fatigue", "guideline implementation",
            "predictive models", "explainability",
        ],
        "Medical NLP & LLM": [
            "clinical note understanding", "medical reasoning",
            "hallucination in medical AI", "multilingual medical NLP",
        ],
        "Medical Imaging AI": [
            "radiology AI", "pathology AI", "retinal imaging",
            "multi-modal fusion", "generalization across sites",
        ],
        "EHR & Data Integration": [
            "interoperability", "missing data", "temporal modeling",
            "privacy-preserving analytics", "federated learning",
        ],
        "Telemedicine": [
            "diagnostic accuracy remote", "patient engagement",
            "chronic disease management", "regulatory frameworks",
        ],
    },
    "Other / Cross-disciplinary": {
        "Bioethics": [
            "AI ethics in healthcare", "consent in genomics",
            "resource allocation", "end-of-life decisions",
        ],
        "Regenerative Medicine": [
            "stem cell therapy", "tissue engineering",
            "wound healing", "organoids",
        ],
        "Microbiome": [
            "gut-brain axis", "fecal transplant", "microbiome therapeutics",
            "skin microbiome", "oral microbiome",
        ],
    },
}


def get_level1_categories() -> list[str]:
    return list(TAXONOMY.keys())


def get_level2_categories(level1: str) -> list[str]:
    return list(TAXONOMY.get(level1, {}).keys())


def get_level3_tags(level1: str, level2: str) -> list[str]:
    return TAXONOMY.get(level1, {}).get(level2, [])


def get_all_level2_flat() -> list[str]:
    result = []
    for l1, l2_dict in TAXONOMY.items():
        for l2 in l2_dict:
            result.append(f"{l1} > {l2}")
    return result
