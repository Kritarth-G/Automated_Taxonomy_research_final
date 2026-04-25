# Automated Taxonomy of Advanced Persistent Threat (APT) Attacks
### A Hierarchical Machine Learning Pipeline for Systematic Classification of APT Research Literature

---

| | |
|---|---|
| **Student** | Kanik Kumar |
| **ID** | 2023A7PS0575P |
| **Course** | CS F266 — Study Project |
| **Supervisor** | Prof. Rajesh Kumar |
| **Department** | Computer Science, BITS Pilani, Pilani Campus |
| **Year** | 2025–2026 |

---

## Abstract

This project presents a fully automated, reproducible pipeline that constructs a **proper hierarchical taxonomy** of Advanced Persistent Threat (APT) attacks from 120 real peer-reviewed academic papers published between 2021 and 2026. Using NLP preprocessing with a dual-group relevance filter and Ward Agglomerative Hierarchical Clustering on TF-IDF vectors, the pipeline produces a 3-level taxonomy: 4 top-level divisions → 10 categories → 20 sub-categories.

---

## The Taxonomy (Key Output)

```
APT ATTACKS TAXONOMY (120 papers, 2021–2026)
│
├── A. Graph-Based & Provenance APT Detection (16 papers)
│   └── A.1  Provenance Graphs & Graph Representation Learning
│        └── A.1.1  Provenance Graph Frameworks & Graph-Level Detection (16p)
│
├── B. Machine Learning-Based APT Detection (38 papers)
│   ├── B.1  Advanced DL, GNN, Federated & Reinforcement Learning
│   │    ├── B.1.1  AI-Driven Cyber Defense & Adaptive Threat Intelligence (4p)
│   │    └── B.1.2  Deep Learning, GNN & Federated Multi-Stage Detection (11p)
│   └── B.2  Classical ML, Optimization & Ensemble Methods
│        ├── B.2.1  Kill Chain, IoT ML & Anomaly-Based Detection (7p)
│        ├── B.2.2  Belief-Rule, Ensemble & Multi-Source Feature Fusion (7p)
│        └── B.2.3  Bio-Inspired Optimization & Feature-Selected Deep Learning (9p)
│
├── C. TTP Analysis, Malware Attribution & Simulation (8 papers)
│   └── C.1  TTP-Based Attribution, Malware Analysis & Dataset Construction
│        └── C.1.1  TTP Attribution, RAT Analysis & APT Dataset Construction (8p)
│
└── D. Broad APT Defense, Intelligence & Reviews (58 papers)
    ├── D.1  Deception, Game Theory & Explainable IIoT Defense
    │    ├── D.1.1  Hypergame Theory & Defensive Deception Architectures (2p)
    │    ├── D.1.2  Few-Shot & Privacy-Preserving Traffic Detection (2p)
    │    └── D.1.3  Game Theory, Explainability & IIoT Lateral Movement Defense (8p)
    ├── D.2  Cybersecurity Knowledge Graphs & Ontology-Based Attribution
    │    └── D.2.1  Knowledge Graphs & Ontology-Based APT Attribution (4p)
    ├── D.3  IDS Framework Evaluation, SIEM & Open-Source Benchmarking
    │    └── D.3.1  IDS Frameworks, Elastic Security & Open-Source Tools (3p)
    ├── D.4  Spatio-Temporal Correlation & Kill-Chain LSTM Detection
    │    └── D.4.1  Spatio-Temporal Correlation & Kill-Chain LSTM Models (2p)
    ├── D.5  ICS, OT & Power Grid APT Attacks and Testbeds
    │    ├── D.5.1  APT Cyberattack Testbeds for Power/DER Systems (2p)
    │    └── D.5.2  Power Grid, Cyber-Physical & 5G APT Detection (6p)
    └── D.6  APT Surveys, Systematic Reviews & Broad Detection Studies
         ├── D.6.1  Mobile Device Security & Systematic Literature Reviews (4p)
         ├── D.6.2  APT Campaign Profiles, Phishing & Ransomware Analysis (4p)
         ├── D.6.3  Multi-Stage Detection, SIEM Rules & Attribution Frameworks (11p)
         ├── D.6.4  Deep Packet Inspection & Signature-Based Detection (4p)
         └── D.6.5  Survey-Based Defense Frameworks & Trend Analysis (6p)
```

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│   scraper.py    │────▶│  preprocess.py   │────▶│  taxonomy_builder.py  │
│                 │     │                  │     │                        │
│ Semantic Scholar│     │ HTML decode      │     │ TF-IDF vectorisation  │
│ API queries     │     │ URL removal      │     │ Ward agglomerative    │
│ APT-only filter │     │ Stop-word removal│     │ hierarchical clustering│
│ 120 papers      │     │ Lemmatization    │     │ 3-level taxonomy      │
│ 2021–2026       │     │ Dual-group filter│     │ Dendrogram + Tree     │
└─────────────────┘     └──────────────────┘     └────────────────────────┘
         │                       │                          │
         ▼                       ▼                          ▼
apt_papers_raw.csv    apt_papers_clean.csv    apt_taxonomy_hierarchical.csv
                                              taxonomy_dendrogram.png
                                              taxonomy_tree.png
                                              taxonomy_summary.png
```

---

## Methodology

### Stage 1 — Data Collection (`scraper.py`)

**API**: Semantic Scholar Academic Graph API  
**Queries**: 5 APT-specific Boolean query strings rotated across years  
**Years**: 2021–2026 (~20 papers per year)  
**Post-fetch filter**: Dual-group relevance check on each abstract:

```
Group A (APT-level signals):
  "advanced persistent threat", "apt", "nation-state", "cyber espionage",
  "threat actor", "state-sponsored", "apt28", "apt29", "lazarus", etc.

Group B (TTP/technical signals):
  "malware", "lateral movement", "exfiltration", "command and control",
  "backdoor", "kill chain", "persistence", "privilege escalation", etc.

A paper is KEPT only if it matches ≥1 pattern from BOTH groups.
This ensures only genuine APT papers with specific technique discussions pass.
```

---

### Stage 2 — NLP Preprocessing (`preprocess.py`)

**Step-by-step pipeline applied to each abstract:**

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | HTML entity decode | Handle encoded characters |
| 2 | URL & DOI removal | Remove non-content tokens |
| 3 | Non-ASCII normalisation | Remove encoding artefacts |
| 4 | Lowercase + punctuation removal | Normalise text |
| 5 | Standalone number removal | Remove irrelevant numeric tokens |
| 6 | Stop-word removal | Remove 200+ high-frequency low-value words |
| 7 | Min token length (>2 chars) | Remove abbreviations |
| 8 | NLTK lemmatization | Reduce morphological variants |

**Custom domain stop-words** (50 terms removed beyond standard English):
`paper, propose, method, approach, technique, result, evaluate, security, cyber, network, system, data, detection, defense, model, framework, algorithm, tool...`

---

### Stage 3 — Hierarchical Taxonomy Construction (`taxonomy_builder.py`)

#### Vectorisation: TF-IDF
```python
TfidfVectorizer(
    max_features=2000,    # top 2000 terms by corpus TF-IDF
    ngram_range=(1,2),    # unigrams + bigrams (captures "lateral movement" etc.)
    min_df=2,             # term must appear in ≥2 papers
    max_df=0.80,          # term must appear in ≤80% of papers
    sublinear_tf=True,    # log(1+tf) to reduce impact of very frequent terms
)
# Followed by L2 normalisation → cosine distance
```

#### Clustering: Ward Agglomerative Hierarchical Clustering
```
WHY WARD LINKAGE?
─────────────────
K-Means produces flat buckets — NOT a taxonomy.
Ward Agglomerative Clustering builds a dendrogram (tree):
  1. Start: each paper is its own cluster
  2. At each step: merge the two clusters whose merger minimises
     the total within-cluster sum of squares (Ward criterion)
  3. Result: a full tree from individual papers up to one root
  4. Cut at 3 heights to get 3-level hierarchy simultaneously:
       k=4  → 4 top-level divisions (Level 1)
       k=10 → 10 categories (Level 2)
       k=20 → 20 sub-categories (Level 3)
```

---

## Output Files

| File | Description |
|------|-------------|
| `apt_papers_raw.csv` | 120 real papers fetched from Semantic Scholar API |
| `apt_papers_clean.csv` | After NLP cleaning and dual-group relevance filter |
| `apt_taxonomy_hierarchical.csv` | **Main output**: every paper with L1/L2/L3 taxonomy labels |
| `taxonomy_dendrogram.png` | Ward clustering dendrogram showing 3 cut levels |
| `taxonomy_tree.png` | Visual hierarchy tree diagram |
| `taxonomy_summary.png` | Clean summary table of all 20 sub-categories |

---

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/apt-taxonomy-pipeline.git
cd apt-taxonomy-pipeline

# 2. Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. Set Semantic Scholar API key (optional but faster)
export SEMANTIC_SCHOLAR_API_KEY="your_key_here"

# 6. Run pipeline in order
python scraper.py          # ~15 min  → apt_papers_raw.csv
python preprocess.py       # ~1 min   → apt_papers_clean.csv
python taxonomy_builder.py # ~2 min   → taxonomy outputs
```

---

## Repository Structure

```
apt-taxonomy-pipeline/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── scraper.py                       ← Stage 1: Data collection
├── preprocess.py                    ← Stage 2: NLP preprocessing
├── taxonomy_builder.py              ← Stage 3: Hierarchical clustering
├── apt_papers_raw.csv               ← 120 real APT papers (output of scraper)
├── apt_papers_clean.csv             ← Cleaned papers (output of preprocess)
├── apt_taxonomy_hierarchical.csv    ← TAXONOMY OUTPUT (main result)
├── taxonomy_dendrogram.png          ← Ward dendrogram visualisation
├── taxonomy_tree.png                ← Hierarchy tree diagram
└── taxonomy_summary.png             ← Summary table of all categories
```

---

## Citation

```bibtex
@misc{apt_taxonomy_2025,
  title   = {Automated Taxonomy of Advanced Persistent Threat (APT) Attacks:
             A Hierarchical Machine Learning Pipeline},
  author  = {Kanik Kumar},
  year    = {2025},
  school  = {BITS Pilani, Pilani Campus},
  note    = {CS F266 Study Project, Supervisor: Prof. Rajesh Kumar}
}
```

---

*Built with: Semantic Scholar API · NLTK · scikit-learn · scipy · matplotlib*
