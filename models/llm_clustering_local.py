import ollama
import pandas as pd
import json
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re 


MODEL_CLUSTER = "llama3"
MODEL_JUDGE = "mistral"

# ============================================================
# GPU CONFIGURATION
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
else:
    print(" No GPU detected, using CPU")

# ============================================================
# 1 DÉFINITION DES THÈMES (LLAMA3)
# ============================================================

def extract_json(text):
    """Extrait le premier JSON valide d'un texte LLM"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def define_themes(df, n_themes=20, sample_size=120):
    sample = df.sample(min(sample_size, len(df)))['Proposition'].tolist()
    props_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(sample)])

    prompt = f"""Voici des propositions citoyennes:

{props_text}

Définis exactement {n_themes} thèmes principaux.
Réponds uniquement en JSON: {{"themes": ["..."]}}"""

    r = ollama.chat(model=MODEL_CLUSTER, messages=[{"role": "user", "content": prompt}])
    txt = r["message"]["content"]
    
    # Chercher le JSON dans la réponse (commence par { et finit par })
    start_idx = txt.find('{')
    end_idx = txt.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        txt = txt[start_idx:end_idx+1]
    
    txt = txt.strip()
    
    try:
        return json.loads(txt)["themes"]
    except json.JSONDecodeError as e:
        print(f"Erreur JSON: {e}")
        print(f"Texte reçu: {txt}")
        raise

# ============================================================
# 2 ASSIGNATION DES THÈMES
# ============================================================

def assign_theme(prop, themes):
    themes_text = "\n".join([f"{i}. {t}" for i, t in enumerate(themes)])
    prompt = f"Thèmes:\n{themes_text}\n\nProposition: {prop}\nNuméro du thème ?"
    r = ollama.chat(model=MODEL_CLUSTER, messages=[{"role": "user", "content": prompt}])
    ans = r["message"]["content"]
    
    # Extraire uniquement les chiffres ASCII (0-9), ignorer les caractères spéciaux
    digits = ''.join(c for c in ans if c in '0123456789')
    
    try:
        idx = int(digits) if digits else 0
    except ValueError:
        idx = 0
    
    return themes[idx] if 0 <= idx < len(themes) else themes[0]

# ============================================================
# 3 JUGE LOCAL (MISTRAL)
# ============================================================

def judge(prop, theme):
    prompt = f"Thème: {theme}\nProposition: {prop}\n1=ok,0=non Format: score|suggestion"
    r = ollama.chat(model=MODEL_JUDGE, messages=[{"role": "user", "content": prompt}])
    ans = r["message"]["content"]
    parts = ans.split("|")
    score = 1 if "1" in parts[0] else 0
    suggestion = parts[1].strip() if score == 0 and len(parts) > 1 else None
    return score, suggestion

# ============================================================
# 4 EXTRAIRE 3 PROPOSITIONS LES PLUS PROCHES DU CENTROÏDE
# ============================================================

# def get_representative_examples(df_cluster):
#     model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
#     texts = df_cluster["Proposition"].tolist()
#     emb = model.encode(texts)
#     centroid = emb.mean(axis=0)
#     sims = cosine_similarity(centroid.reshape(1, -1), emb)[0]
#     top_idx = sims.argsort()[-5:][::-1]
#     return [texts[i] for i in top_idx]

def get_representative_examples(df_cluster):
    return df_cluster["Proposition"].sample(min(5, len(df_cluster))).tolist()

# ============================================================
# PIPELINE COMPLET
# ============================================================

def run_pipeline(df, n_themes=20):

    print("1. Définition des thèmes")
    themes = define_themes(df, n_themes)

    print("2. Assignation")
    df_out = df.copy()
    df_out["theme"] = [assign_theme(p, themes) for p in tqdm(df["Proposition"])]

    print("3. Création ID clusters")
    theme_to_id = {t: i for i, t in enumerate(sorted(set(df_out["theme"])))}
    df_out["cluster_id"] = df_out["theme"].map(theme_to_id)

    print("3.5 Sauvegarde des assignations")
    df_out.to_csv("llm_clustering_assignments.csv", index=False)
    print("✓ Assignations sauvegardées dans: llm_clustering_assignments.csv")

    print("4. Évaluation LLM-as-a-judge (sur 100 propositions aléatoires)")
    df_judge = df_out.sample(min(100, len(df_out)))
    
    scores, suggestions = [], []
    judge_indices = []
    
    for idx, (p, t) in enumerate(tqdm(zip(df_judge["Proposition"], df_judge["theme"]), total=len(df_judge))):
        s, sug = judge(p, t)
        scores.append(s)
        suggestions.append(sug)
        judge_indices.append(df_judge.index[idx])
        time.sleep(0.1)

    # Ajouter les scores au dataframe original
    df_out["judge_score"] = None
    df_out["judge_suggestion"] = None
    df_out.loc[judge_indices, "judge_score"] = scores
    df_out.loc[judge_indices, "judge_suggestion"] = suggestions

    print("5. Exemples représentatifs par thème")
    summaries = []
    for theme in df_out["theme"].unique():
        sub = df_out[df_out["theme"] == theme]
        examples = get_representative_examples(sub)
        summaries.append({"theme": theme, "examples": examples})

    summary_df = pd.DataFrame(summaries)

    print("6. Sauvegarde finale")
    df_out.to_csv("llm_local_clustering.csv", index=False)
    summary_df.to_json("llm_themes_summary.json", orient="records", force_ascii=False, indent=2)

    print(" Terminé")

    return df_out, summary_df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Charger les données
    df = pd.read_excel("/home/ensai/Cours/PFE/Topic-modeling-with-NLP/data/makeorg_biodiversite.xlsx")
    
    # # Pour tester : prendre un petit sample (50 propositions)
    # df_test = df.sample(min(10, len(df)), random_state=42)
    # print(f"Test avec {len(df_test)} propositions (total: {len(df)})\n")
    
    # Exécuter le pipeline
    df_out, summary_df = run_pipeline(df, n_themes=20)
    
    print("\nRésultats sauvegardés:")
    print("- llm_local_clustering.csv")
    print("- llm_themes_summary.json")
