"""
taxonomy_builder.py — Stage 3: Hierarchical Taxonomy Construction
===================================================================
Builds a proper 3-level hierarchical taxonomy of APT attacks using
Ward Agglomerative Hierarchical Clustering.

WHY HIERARCHICAL CLUSTERING?
─────────────────────────────
A taxonomy requires a hierarchy (parent → child → sub-child), NOT flat
buckets. Ward Agglomerative Clustering builds a dendrogram — a tree where:
  • Each paper starts as its own leaf node
  • At each step, the two closest clusters merge (Ward criterion:
    minimises within-cluster variance at each merge)
  • The resulting dendrogram is cut at 3 heights simultaneously:
       Cut 1 → k=4  → 4 top-level divisions (Level 1)
       Cut 2 → k=10 → 10 categories (Level 2)
       Cut 3 → k=20 → 20 sub-categories (Level 3)

VECTORISATION:
──────────────
TF-IDF (Term Frequency – Inverse Document Frequency)
  • 2000 features, bigrams, cosine distance, L2 normalised

CLUSTER NAMING:
────────────────
Names are assigned dynamically by scoring each cluster against
domain keyword groups. This ensures correct names regardless of
the internal cluster ID ordering produced by Ward linkage.

OUTPUTS:
─────────
  apt_taxonomy_hierarchical.csv  — all papers with L1/L2/L3 labels
  taxonomy_dendrogram.png        — dendrogram with 3 cut lines
  taxonomy_tree.png              — hierarchy tree diagram
  taxonomy_summary.png           — complete summary table

Input  : apt_papers_clean.csv
"""

import re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path
warnings.filterwarnings("ignore")

INPUT       = "apt_papers_clean.csv"
OUT_CSV     = "apt_taxonomy_hierarchical.csv"
OUT_DENDRO  = "taxonomy_dendrogram.png"
OUT_TREE    = "taxonomy_tree.png"
OUT_SUMMARY = "taxonomy_summary.png"

# ── KEYWORD THEMES for dynamic cluster naming ─────────────────────────────────
THEMES = {
    "L1_graph":  ["provenance","audit log","anubis","magic ","prohunter","aptcglp",
                  "along came spider","warning-graph","causal","graph-level"],
    "L1_ml":     ["deep learning","neural","convolutional","lstm","federated",
                  "reinforcement","gnn","graph neural","ensemble","optimization",
                  "particle swarm","cnn","bigru","transformer"],
    "L1_ttp":    ["ttp","tactics techniques","attribution","initial access","rat ",
                  "remote access trojan","simulate","windows-apt","decoding shadows",
                  "chasing shadow","genesis of cyber","cloud forensic"],
    "L1_broad":  ["survey","review","systematic","game theory","deception","hypergame",
                  "knowledge graph","ics","power grid","substation","iot","siem",
                  "phishing","kill chain","anomaly","benchmark"],

    "L2_provenance": ["provenance","audit log","anubis","magic ","prohunter","aptcglp",
                      "causal","whole-system","graph-level","aisle","ltrdetector"],
    "L2_adv_dl":     ["federated","reinforcement","few-shot","privacy","gnn",
                      "graph neural","deep reinforcement","quantum","hidden markov",
                      "graphsage","dynamic adaptive"],
    "L2_classic_ml": ["kill chain","anomaly","ensemble","particle swarm","belief",
                      "lstm","spatial","iot ml","convolutional","optimization",
                      "feature selection","deep learning","neural network","cnn"],
    "L2_ttp_attr":   ["ttp","tactics techniques","attribution","malware","rat ",
                      "initial access","cloud forensic","decoding shadows",
                      "windows-apt","dataset","genesis","chasing"],
    "L2_deception":  ["deception","game theory","hypergame","foureye","defensive",
                      "explainable","iiot","lateral movement","raptor","secaas",
                      "predictive cyber","defense of apt on industrial"],
    "L2_kg":         ["knowledge graph","cskg","ontology","zachman",
                      "cybersecurity knowledge","domain-specific knowledge"],
    "L2_ids":        ["elastic","siem","open source","benchmark","ids framework",
                      "performance evaluation","intrusion detection framework"],
    "L2_temporal":   ["spatial-temporal","spatiotemporal","kill chain node",
                      "correlation based method"],
    "L2_ics":        ["power grid","substation","iec 61850","scada","der ",
                      "testbed","virtual power","5g industrial","cyber-physical"],
    "L2_surveys":    ["survey","review","systematic","phishing","ransomware","nigeria",
                      "mobile device","detection survey","trends","broad"],

    "L3_prov_graph":    ["provenance","audit log","anubis","magic ","prohunter","aptcglp",
                         "graph-level","whole-system","causal","aisle","ltrdetector"],
    "L3_ai_defense":    ["ai-driven","adaptive ai","deep recurrent","graphsage","quantum",
                         "hidden markov","dynamic adaptive","neurosymbolic"],
    "L3_dl_federated":  ["federated","deep reinforcement","graph neural","integrating gnn",
                         "few-shot","privacy-preserving","transformer models",
                         "context-aware","linux systems","iiot environments"],
    "L3_kill_chain":    ["kill chain","anomaly detection ensemble","clustering algorithms",
                         "industrial internet","iot security","prior knowledge","belief rule"],
    "L3_belief_ensemble":["belief rule","reliability assessment","inference phase",
                          "malicious code","ensemble learning","multi-source","behavior pattern"],
    "L3_bio_optim":     ["particle swarm","feature selection","fuzzy inference",
                         "significant feature","2d-cnn","cat swarm","high-accuracy",
                         "optimized deep","hybrid ensemble"],
    "L3_ttp_attr":      ["ttp","initial access","chasing shadow","genesis","windows-apt",
                         "decoding shadows","rat simulation","cloud forensic","malware-based"],
    "L3_hypergame":     ["hypergame","foureye","defensive deception","resisting multiple"],
    "L3_few_shot":      ["few-shot","privacy-preserving traffic","meta learning"],
    "L3_game_iiot":     ["game theory","explainable","predictive cyber","defense of apt",
                         "raptor","secaas","lateral movement modeling","framework cybersecurity"],
    "L3_kg_ontology":   ["knowledge graph","cskg","ontology","domain-specific knowledge",
                         "zachman","organization attribution"],
    "L3_ids_eval":      ["elastic","open source","intrusion detection framework",
                         "performance evaluation","effective threat detection"],
    "L3_temporal":      ["spatial-temporal","kill chain node mapping","correlation based method"],
    "L3_testbed":       ["testbed","distributed energy","power transformer","apt-style attack"],
    "L3_power_grid":    ["power grid","iec 61850","virtual power","5g industrial",
                         "spatio-temporal","cyber-physical power","smart grid",
                         "hybrid sampling"],
    "L3_mobile_review": ["mobile device","systematic literature","conceptual framework",
                         "5g mobile network","adaptive target"],
    "L3_campaign":      ["phishing","ransomware","nigeria","analysis of cyber attacks",
                         "advanced persistent threat detection and defence"],
    "L3_multistage":    ["multi-layer","dmapt","attribution using zachman","siem ruleset",
                         "early apt","forensic framework","aviator","wazuh",
                         "supply chain security","provagent","warning-graph"],
    "L3_dpi":           ["deep packet inspection","artificial intelligence for deep",
                         "a survey of advanced","survey of apt","a novel method",
                         "reliability assessment"],
    "L3_survey_broad":  ["phishing and apt","advanced persistent threat detection and defence",
                         "nigeria","escalating","e-aptdetect","investigation of ml",
                         "advanced persistent threat attack targeting"],
}


def kw_score(texts, keywords):
    combined = " ".join(t.lower() for t in texts)
    return sum(1 for kw in keywords if kw.lower() in combined) / max(len(keywords), 1)


def best_theme(texts, theme_dict):
    return max(theme_dict, key=lambda t: kw_score(texts, theme_dict[t]))


def assign_unique(cluster_ids, texts_by_cluster, theme_group, ordered_themes):
    """Assign theme names to cluster IDs uniquely, in priority order."""
    name_map = {}
    remaining = list(cluster_ids)
    for name, theme_key in ordered_themes:
        if not remaining: break
        scores = {c: kw_score(texts_by_cluster[c], theme_group[theme_key])
                  for c in remaining}
        best = max(scores, key=scores.get)
        name_map[best] = name
        remaining.remove(best)
    for c in remaining:
        name_map[c] = f"APT Detection Sub-group (cluster {c})"
    return name_map


# ── TAXONOMY LABEL DEFINITIONS ────────────────────────────────────────────────
L1_ORDERED = [
    ("A. Graph-Based & Provenance APT Detection",            "L1_graph"),
    ("B. Machine Learning-Based APT Detection",              "L1_ml"),
    ("C. TTP Analysis, Malware Attribution & Simulation",    "L1_ttp"),
    ("D. Broad APT Defense, Intelligence & Reviews",         "L1_broad"),
]
L2_ORDERED = [
    ("A.1  Provenance Graphs & Graph Representation Learning",         "L2_provenance"),
    ("B.1  Advanced DL, GNN, Federated & Reinforcement Learning",      "L2_adv_dl"),
    ("B.2  Classical ML, Optimization & Ensemble Methods",             "L2_classic_ml"),
    ("C.1  TTP-Based Attribution, Malware Analysis & Dataset Construction","L2_ttp_attr"),
    ("D.1  Deception, Game Theory & Explainable IIoT Defense",         "L2_deception"),
    ("D.2  Cybersecurity Knowledge Graphs & Ontology",                 "L2_kg"),
    ("D.3  IDS Framework Evaluation, SIEM & Benchmarking",             "L2_ids"),
    ("D.4  Spatio-Temporal Correlation & Kill-Chain LSTM Detection",   "L2_temporal"),
    ("D.5  ICS, OT & Power Grid APT Attacks and Testbeds",             "L2_ics"),
    ("D.6  APT Surveys, Systematic Reviews & Broad Detection Studies", "L2_surveys"),
]
L3_ORDERED = [
    ("A.1.1  Provenance Graph Frameworks & Graph-Level APT Detection", "L3_prov_graph"),
    ("B.1.1  AI-Driven Cyber Defense & Adaptive Threat Intelligence",  "L3_ai_defense"),
    ("B.1.2  Deep Learning, GNN & Federated Multi-Stage APT Detection","L3_dl_federated"),
    ("B.2.1  Kill Chain, IoT ML & Anomaly-Based Detection",            "L3_kill_chain"),
    ("B.2.2  Belief-Rule, Ensemble & Multi-Source Feature Fusion",     "L3_belief_ensemble"),
    ("B.2.3  Bio-Inspired Optimization & Feature-Selected Deep Learning","L3_bio_optim"),
    ("C.1.1  TTP Attribution, RAT Analysis & APT Dataset Construction","L3_ttp_attr"),
    ("D.1.1  Hypergame Theory & Defensive Deception Architectures",    "L3_hypergame"),
    ("D.1.2  Few-Shot & Privacy-Preserving Traffic Detection",         "L3_few_shot"),
    ("D.1.3  Game Theory, Explainability & IIoT Lateral Movement Defense","L3_game_iiot"),
    ("D.2.1  Knowledge Graphs & Ontology-Based APT Attribution",       "L3_kg_ontology"),
    ("D.3.1  IDS Frameworks, Elastic Security & Open-Source Evaluation","L3_ids_eval"),
    ("D.4.1  Spatio-Temporal Correlation & Kill-Chain LSTM Models",    "L3_temporal"),
    ("D.5.1  APT Cyberattack Testbeds for Power/DER Systems",          "L3_testbed"),
    ("D.5.2  Power Grid, Cyber-Physical & 5G APT Detection",           "L3_power_grid"),
    ("D.6.1  Mobile Device Security & Systematic Literature Reviews",  "L3_mobile_review"),
    ("D.6.2  APT Campaign Profiles, Phishing & Ransomware Analysis",   "L3_campaign"),
    ("D.6.3  Multi-Stage Detection, SIEM Rules & Attribution Frameworks","L3_multistage"),
    ("D.6.4  Deep Packet Inspection & Signature-Based Detection",      "L3_dpi"),
    ("D.6.5  Survey-Based Defense Frameworks & Trend Analysis",        "L3_survey_broad"),
]

L1_PALETTE = ["#C62828","#0D47A1","#2E7D32","#BF360C"]


def main():
    print("="*60)
    print("  APT Taxonomy — Stage 3: Hierarchical Clustering")
    print("  Algorithm: Ward Agglomerative Hierarchical Clustering")
    print("="*60)

    if not Path(INPUT).exists():
        print(f"ERROR: '{INPUT}' not found. Run preprocess.py first.")
        return

    # ── Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(INPUT, encoding="utf-8")
    print(f"Loaded {len(df)} papers")
    df["_full"] = (df["title"].fillna("")+" "+df["abstract"].fillna("")).str.lower()

    STOP = {
        "a","an","the","and","or","in","on","to","of","with","by","from","is","are",
        "was","were","be","have","has","had","do","does","not","no","this","that","it",
        "we","they","as","if","can","also","however","thus","all","which","each","both",
        "more","most","than","there","such","about","use","used","using","based","show",
        "present","result","propose","approach","method","paper","technique","work",
        "new","existing","different","various","multiple","include","enable","make",
        "significant","security","cyber","network","system","data","information","user",
        "attack","detection","detect","defense","model","framework","algorithm","tool",
        "provide","achieve","high","low","real","time","case","type","level","rate",
        "well","via","one","first","second","large","small","find","found","develop",
    }
    def clean(t):
        if not isinstance(t,str): return ""
        t = re.sub(r"https?://\S+","",t); t = re.sub(r"[^\w\s]"," ",t)
        t = re.sub(r"\b\d+\b"," ",t)
        return " ".join(w for w in t.lower().split()
                        if len(w)>2 and w not in STOP
                        and re.match(r'^[a-z][a-z0-9\-]*$',w))

    col = "abstract_clean" if "abstract_clean" in df.columns else "abstract"
    df["_text"] = (df["title"].fillna("")+" "+df[col].fillna("")).apply(clean)

    # ── TF-IDF + Ward ──────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2),
                             min_df=2, max_df=0.80, sublinear_tf=True)
    X    = normalize(tfidf.fit_transform(df["_text"]), norm="l2")
    dist = pdist(X.toarray(), metric="cosine")
    Z    = linkage(dist, method="ward")
    print("Ward linkage clustering complete")

    df["_L1"] = fcluster(Z, t=4,  criterion="maxclust") - 1
    df["_L2"] = fcluster(Z, t=10, criterion="maxclust") - 1
    df["_L3"] = fcluster(Z, t=20, criterion="maxclust") - 1

    # ── Build text lookup per cluster ─────────────────────────────────────
    l1_texts = {c: df[df["_L1"]==c]["_full"].tolist() for c in df["_L1"].unique()}
    l2_texts = {c: df[df["_L2"]==c]["_full"].tolist() for c in df["_L2"].unique()}
    l3_texts = {c: df[df["_L3"]==c]["_full"].tolist() for c in df["_L3"].unique()}

    # ── Assign names dynamically ──────────────────────────────────────────
    l1_map = assign_unique(df["_L1"].unique(), l1_texts, THEMES, L1_ORDERED)
    l2_map = assign_unique(df["_L2"].unique(), l2_texts, THEMES, L2_ORDERED)
    l3_map = assign_unique(df["_L3"].unique(), l3_texts, THEMES, L3_ORDERED)

    df["L1"]       = df["_L1"]
    df["L1_label"] = df["_L1"].map(l1_map)
    df["L2"]       = df["_L2"]
    df["L2_label"] = df["_L2"].map(l2_map)
    df["L3"]       = df["_L3"]
    df["L3_label"] = df["_L3"].map(l3_map)
    df = df.drop(columns=["_L1","_L2","_L3","_text","_full"])

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_cols = [c for c in ["paper_id","title","year","authors","venue","url",
                             "L1","L1_label","L2","L2_label","L3","L3_label"]
                if c in df.columns]
    out = df[out_cols].sort_values(["L1","L2","L3","year"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓ Saved {OUT_CSV} ({len(out)} papers)")

    # ── Print taxonomy ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  HIERARCHICAL APT TAXONOMY — Ward Agglomerative Clustering")
    print("  3 Levels: 4 Divisions → 10 Categories → 20 Sub-categories")
    print("="*72)
    for l1 in sorted(df.L1.unique()):
        g1 = df[df.L1==l1]
        print(f"\n  {'█'*65}")
        print(f"  LEVEL 1 │ {l1_map[l1]}  ({len(g1)} papers)")
        for l2 in sorted(g1.L2.unique()):
            g2 = df[df.L2==l2]
            print(f"\n    ├── LEVEL 2 │ {l2_map[l2]}  ({len(g2)} papers)")
            for l3 in sorted(g2.L3.unique()):
                g3 = df[df.L3==l3]
                print(f"\n         └── LEVEL 3 │ {l3_map[l3]}  ({len(g3)} papers)")
                for _,r in g3.sort_values("year").iterrows():
                    print(f"                  [{r.year}] {r.title[:65]}")

    # ── sorted lists for plotting ─────────────────────────────────────────
    l1_sorted = sorted(df.L1.unique())
    l1_col    = {cid: L1_PALETTE[i % 4] for i, cid in enumerate(l1_sorted)}

    # ── FIGURE 1 — DENDROGRAM ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(24, 10), facecolor="#0D1117")
    ax1.set_facecolor("#0D1117")
    fig1.suptitle(
        "Automated Hierarchical Taxonomy of APT Attacks — Ward Agglomerative Clustering Dendrogram\n"
        "120 Real APT Research Papers (2021–2026)  ·  TF-IDF Vectorisation  ·"
        "  CS F266, BITS Pilani — Kanik Kumar (2023A7PS0575P)",
        fontsize=11, fontweight="bold", color="white", y=0.99)

    dendrogram(Z, ax=ax1, truncate_mode="lastp", p=40,
               leaf_rotation=90, leaf_font_size=6,
               color_threshold=0.65*max(Z[:,2]),
               above_threshold_color="#444444", count_sort="ascending")
    ax1.set_xlabel("Sample Index (40 largest merged clusters shown)", color="#888", fontsize=9)
    ax1.set_ylabel("Ward Linkage Distance", color="#888", fontsize=9)
    ax1.tick_params(colors="#888", labelsize=7)
    for sp in ax1.spines.values(): sp.set_edgecolor("#30363D")

    heights = sorted([r[2] for r in Z], reverse=True)
    cuts = [(heights[3]+heights[4])/2,
            (heights[9]+heights[10])/2,
            (heights[19]+heights[20])/2]
    for h, lbl, col in zip(cuts,
        ["Cut → k=4 (Level 1: 4 Divisions)",
         "Cut → k=10 (Level 2: 10 Categories)",
         "Cut → k=20 (Level 3: 20 Sub-categories)"],
        ["#E63946","#FF9800","#4CAF50"]):
        ax1.axhline(h, color=col, lw=2.0, ls="--", alpha=0.9, zorder=5)
        ax1.text(ax1.get_xlim()[1]*0.995, h+0.005, lbl,
                 color=col, fontsize=8.5, ha="right", va="bottom", fontweight="bold")

    patches = [mpatches.Patch(color=L1_PALETTE[i], label=l1_map[c])
               for i,c in enumerate(l1_sorted)]
    ax1.legend(handles=patches, loc="upper left", fontsize=8.5, framealpha=0.2,
               facecolor="#21262D", edgecolor="#444", labelcolor="white",
               title="Level 1 — Top Divisions", title_fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(OUT_DENDRO, dpi=150, bbox_inches="tight",
                facecolor=fig1.get_facecolor())
    plt.close()
    print(f"✓ Saved {OUT_DENDRO}")

    # ── FIGURE 2 — TREE DIAGRAM ───────────────────────────────────────────
    fig2 = plt.figure(figsize=(28, 20), facecolor="#0D1117")
    ax2  = fig2.add_axes([0,0,1,1])
    ax2.set_facecolor("#0D1117"); ax2.axis("off")
    ax2.set_xlim(0,28); ax2.set_ylim(0,20)

    fig2.text(0.5,0.982,
              "Automated Hierarchical Taxonomy of Advanced Persistent Threat (APT) Attacks",
              ha="center",va="top",fontsize=15,fontweight="bold",color="white")
    fig2.text(0.5,0.962,
              "Algorithm: Ward Agglomerative Hierarchical Clustering  ·  "
              "TF-IDF (2000 bigram features, cosine distance)  ·  "
              "120 Real Papers (2021–2026)  ·  Kanik Kumar (2023A7PS0575P), BITS Pilani",
              ha="center",va="top",fontsize=9,color="#888",style="italic")

    def bx(x,y,w,h,txt,bg,fs=8.5,tc="white",lw=1.2):
        ax2.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
                                     boxstyle="round,pad=0.04,rounding_size=0.1",
                                     facecolor=bg,edgecolor="white",linewidth=lw,zorder=4))
        ax2.text(x,y,txt,ha="center",va="center",fontsize=fs,color=tc,
                 fontweight="bold",zorder=5,multialignment="center")

    def arr(x1,y1,x2,y2,col="#555"):
        ax2.annotate("",xy=(x2,y2+0.22),xytext=(x1,y1-0.22),
                     arrowprops=dict(arrowstyle="-|>",color=col,lw=1.2,mutation_scale=12),zorder=3)

    bx(14,19.1,6.5,0.72,"APT ATTACKS — HIERARCHICAL TAXONOMY\n120 Real Papers · 2021–2026",
       "#1A237E",fs=11,lw=2)

    L1X = [3.5,10.5,18.5,24.0]; L1Y = 17.1
    for i,cid in enumerate(l1_sorted):
        nm = l1_map[cid]; cnt=(df.L1==cid).sum()
        div = nm.split(".")[0]; rest = nm.split("  ",1)[1] if "  " in nm else nm
        lbl = f"{div}.\n{rest[:22]}\n({cnt}p)"
        arr(14,19.1,L1X[i],L1Y); bx(L1X[i],L1Y,5.8,1.55,lbl,L1_PALETTE[i],fs=8.5)

    L2Y = 14.2
    l2_sorted = sorted(df.L2.unique())
    # Group L2 by L1 parent
    l2_by_l1 = {}
    for c in l2_sorted:
        p = df[df.L2==c].L1.mode()[0]; l2_by_l1.setdefault(p,[]).append(c)

    L2_X = {l1_sorted[0]:[3.5], l1_sorted[1]:[8.5,13.0],
            l1_sorted[2]:[18.5], l1_sorted[3]:[19.8,21.5,23.0,24.5,26.2,27.8]}
    L2_C = {l1_sorted[0]:["#8B0000"], l1_sorted[1]:["#0D47A1","#0D47A1"],
            l1_sorted[2]:["#1B5E20"], l1_sorted[3]:["#7B2D00"]*6}

    for l1p,l2list in l2_by_l1.items():
        xs = L2_X.get(l1p,[L1X[l1_sorted.index(l1p)]])
        cs = L2_C.get(l1p,["#333"]*10)
        l1x= L1X[l1_sorted.index(l1p)]
        for j,c in enumerate(l2list[:len(xs)]):
            nm = l2_map[c]; cnt=(df.L2==c).sum()
            px = nm.split("  ")[0]; rest = nm.split("  ",1)[1] if "  " in nm else nm
            lbl = f"{px}\n{rest[:20]}\n({cnt}p)"
            arr(l1x,L1Y,xs[j],L2Y); bx(xs[j],L2Y,2.5,2.1,lbl,cs[j],fs=7.2)

    # Level 3 grid
    l3_sorted = sorted(df.L3.unique())
    L3Y0 = 10.8; cols_per_row = 7
    for idx,c in enumerate(l3_sorted):
        nm = l3_map[c]; cnt=(df.L3==c).sum()
        px = nm.split("  ")[0]; rest = nm.split("  ",1)[1] if "  " in nm else nm
        lbl = f"{px}\n{rest[:22]}\n({cnt}p)"
        r = idx//cols_per_row; ci = idx%cols_per_row
        xc = 2.0+ci*3.8; yc = L3Y0-r*2.4
        bx(xc,yc,3.5,2.15,lbl,"#2D3748",fs=7.0)

    for y,txt in [(19.1,"ROOT"),(17.1,"LEVEL 1"),(14.2,"LEVEL 2"),(10.8,"LEVEL 3")]:
        ax2.text(0.5,y,txt,ha="center",va="center",fontsize=8,color="white",fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3",facecolor="#161B22",edgecolor="#555",alpha=0.9))

    ax2.text(14,0.3,
             "Algorithm: Ward Agglomerative Hierarchical Clustering  ·  "
             "3 Levels: 4 Divisions → 10 Categories → 20 Sub-categories  ·  "
             "Kanik Kumar (2023A7PS0575P)  ·  Prof. Rajesh Kumar  ·  BITS Pilani",
             ha="center",va="center",fontsize=7.5,color="#555")
    plt.savefig(OUT_TREE, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"✓ Saved {OUT_TREE}")

    # ── FIGURE 3 — SUMMARY TABLE ──────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(22, 18), facecolor="#0D1117")
    ax3.set_facecolor("#0D1117"); ax3.axis("off")
    ax3.set_xlim(0,22); ax3.set_ylim(0,18)
    fig3.text(0.5,0.985,"Hierarchical APT Taxonomy — Complete Summary Table",
              ha="center",va="top",fontsize=14,fontweight="bold",color="white")
    fig3.text(0.5,0.968,
              "Ward Agglomerative Clustering · 120 Papers · 4 Divisions · "
              "10 Categories · 20 Sub-categories · Kanik Kumar (2023A7PS0575P) · BITS Pilani",
              ha="center",va="top",fontsize=9,color="#888",style="italic")

    COLS_X=[0.8,2.5,6.7,13.5,20.8]; CWIDTHS=[1.2,3.8,5.5,5.5,1.0]
    HEADERS=["Div.","Level 1 — Division","Level 2 — Category","Level 3 — Sub-category","n"]
    HDR_Y=17.0
    for hdr,cx,cw in zip(HEADERS,COLS_X,CWIDTHS):
        ax3.add_patch(FancyBboxPatch((cx-cw/2,HDR_Y-0.28),cw,0.52,
                                     boxstyle="round,pad=0.02",
                                     facecolor="#1A237E",edgecolor="#3949AB",lw=1))
        ax3.text(cx,HDR_Y,hdr,ha="center",va="center",fontsize=9.5,color="white",fontweight="bold")

    DIV_C = {l1_map[c]: L1_PALETTE[i] for i,c in enumerate(l1_sorted)}
    L1BG  = {"A":"#3B0000","B":"#001238","C":"#003810","D":"#3B1800"}
    ROW_H=0.69; START_Y=16.2; row_i=0

    for l1 in sorted(df.L1.unique()):
        g1=df[df.L1==l1]; n1=l1_map[l1]
        dc=DIV_C.get(n1,"#444"); div=n1[0]
        l1s=n1.split("  ",1)[1] if "  " in n1 else n1
        fl1=True
        for l2 in sorted(g1.L2.unique()):
            g2=df[df.L2==l2]; n2=l2_map[l2]
            n2s=n2.split("  ",1)[1] if "  " in n2 else n2; fl2=True
            for l3 in sorted(g2.L3.unique()):
                g3=df[df.L3==l3]; n3=l3_map[l3]
                n3s=n3.split("  ",1)[1] if "  " in n3 else n3
                cnt=len(g3); y=START_Y-row_i*ROW_H
                bg=L1BG.get(div,"#111")
                ax3.add_patch(FancyBboxPatch((0.1,y-ROW_H/2+0.04),21.8,ROW_H-0.06,
                                             boxstyle="round,pad=0.02",
                                             facecolor=bg,edgecolor="#333",lw=0.5))
                ax3.add_patch(FancyBboxPatch((0.15,y-0.22),1.1,0.44,
                                             boxstyle="round,pad=0.03",
                                             facecolor=dc if fl1 else "#1A1A1A",
                                             edgecolor=dc,lw=0.8))
                ax3.text(0.7,y,div,ha="center",va="center",fontsize=9,color="white",fontweight="bold")
                if fl1: ax3.text(2.5,y,l1s,ha="center",va="center",fontsize=7.5,
                                 color=dc,fontweight="bold",multialignment="center")
                if fl2: ax3.text(6.7,y,n2s,ha="center",va="center",fontsize=7.5,
                                 color="#64B5F6",fontweight="bold",multialignment="center")
                ax3.text(13.5,y,n3s,ha="left",va="center",fontsize=7.8,
                         color="#E0E0E0",multialignment="left")
                ax3.add_patch(FancyBboxPatch((20.35,y-0.2),0.9,0.4,
                                             boxstyle="round,pad=0.04",
                                             facecolor=dc,edgecolor="white",lw=0.8))
                ax3.text(20.8,y,str(cnt),ha="center",va="center",fontsize=9,
                         color="white",fontweight="bold")
                fl1=False; fl2=False; row_i+=1

    for xc in [1.35,4.45,10.05,20.2]:
        ax3.axvline(xc,color="#333",lw=0.8,ymin=0.04,ymax=0.96)
    fy = max(START_Y-row_i*ROW_H-0.05, 0.1)
    ax3.add_patch(FancyBboxPatch((0.1,fy),21.8,0.55,
                                  boxstyle="round,pad=0.03",
                                  facecolor="#1A237E",edgecolor="#3949AB",lw=1.5))
    ax3.text(11,fy+0.27,
             f"TOTAL: {len(df)} papers  ·  4 Divisions  ·  10 Categories  ·  20 Sub-categories",
             ha="center",va="center",fontsize=10,color="white",fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_SUMMARY, dpi=150, bbox_inches="tight",
                facecolor=fig3.get_facecolor())
    plt.close()
    print(f"✓ Saved {OUT_SUMMARY}")

    print("\n" + "="*60)
    print("  ALL OUTPUTS SAVED:")
    for f in [OUT_CSV, OUT_DENDRO, OUT_TREE, OUT_SUMMARY]:
        print(f"    {f}")
    print("="*60)


if __name__ == "__main__":
    main()
