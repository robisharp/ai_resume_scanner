"""
Resume Screening System - ML Backend
Models: Logistic Regression + Support Vector Machine (SVM)
Task: Binary classification — Eligible (1) vs Not Eligible (0)
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# 1.  KEYWORD CONFIGURATION
#     Define must-have, preferred, and red-flag keywords per role.
#     Customize this section for your hiring pipeline.
# ─────────────────────────────────────────────────────────────

ROLE_KEYWORD_CONFIG = {
    "data_scientist": {
        "must_have": [
            "python", "machine learning", "deep learning", "statistics",
            "data analysis", "sql", "tensorflow", "pytorch", "scikit-learn",
            "numpy", "pandas", "model training", "neural network"
        ],
        "preferred": [
            "nlp", "computer vision", "xgboost", "spark", "hadoop",
            "azure", "aws", "gcp", "docker", "kubernetes",
            "a/b testing", "experiment design", "tableau", "power bi",
            "feature engineering", "hyperparameter tuning", "mlops"
        ],
        "red_flags": [
            "no experience", "beginner", "basic knowledge only",
            "unfamiliar with", "no programming"
        ],
        "min_experience_years": 2
    },
    "software_engineer": {
        "must_have": [
            "java", "python", "c++", "javascript", "algorithms",
            "data structures", "git", "rest api", "unit testing",
            "object oriented", "agile", "code review"
        ],
        "preferred": [
            "react", "node.js", "spring boot", "microservices", "ci/cd",
            "docker", "kubernetes", "aws", "gcp", "azure",
            "graphql", "redis", "postgresql", "mongodb", "kafka"
        ],
        "red_flags": [
            "no coding experience", "non-technical", "no version control"
        ],
        "min_experience_years": 1
    },
    "data_analyst": {
        "must_have": [
            "sql", "excel", "data analysis", "reporting", "dashboard",
            "python", "r", "visualization", "kpi", "business intelligence"
        ],
        "preferred": [
            "tableau", "power bi", "looker", "dax", "etl",
            "google analytics", "a/b testing", "forecasting",
            "pivot tables", "vlookup", "statistical analysis"
        ],
        "red_flags": [
            "no data experience", "no sql", "non-analytical"
        ],
        "min_experience_years": 1
    }
}

# ─────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def extract_years_of_experience(text: str) -> float:
    """Parse years of experience from resume text."""
    text_lower = text.lower()
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+of\s+experience',
    ]
    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        years_found.extend([int(m) for m in matches])
    return max(years_found) if years_found else 0.0


def extract_education_score(text: str) -> float:
    """Score education level found in resume."""
    text_lower = text.lower()
    education_hierarchy = {
        "phd": 4.0, "ph.d": 4.0, "doctorate": 4.0,
        "master": 3.0, "m.s": 3.0, "msc": 3.0, "m.tech": 3.0, "mba": 3.0,
        "bachelor": 2.0, "b.s": 2.0, "b.e": 2.0, "b.tech": 2.0, "undergraduate": 2.0,
        "diploma": 1.0, "associate": 1.0,
        "high school": 0.5, "secondary": 0.5
    }
    score = 0.0
    for keyword, value in education_hierarchy.items():
        if keyword in text_lower:
            score = max(score, value)
    return score


def keyword_feature_engineering(text: str, role: str) -> dict:
    """
    Build a rich feature dict from keyword matching.
    Returns counts and ratios for ML model input.
    """
    text_lower = text.lower()
    config = ROLE_KEYWORD_CONFIG.get(role, ROLE_KEYWORD_CONFIG["data_scientist"])

    must_hits = sum(1 for kw in config["must_have"] if kw in text_lower)
    pref_hits = sum(1 for kw in config["preferred"] if kw in text_lower)
    red_flag_hits = sum(1 for kw in config["red_flags"] if kw in text_lower)

    must_ratio = must_hits / max(len(config["must_have"]), 1)
    pref_ratio = pref_hits / max(len(config["preferred"]), 1)

    years_exp = extract_years_of_experience(text)
    edu_score = extract_education_score(text)

    # Has certifications?
    cert_keywords = ["certified", "certification", "aws certified", "google certified",
                     "pmp", "cfa", "cpa", "ccna", "comptia"]
    cert_count = sum(1 for kw in cert_keywords if kw in text_lower)

    # Resume length as quality signal
    word_count = len(text.split())

    return {
        "must_have_hits": must_hits,
        "must_have_ratio": must_ratio,
        "preferred_hits": pref_hits,
        "preferred_ratio": pref_ratio,
        "red_flag_hits": red_flag_hits,
        "years_experience": years_exp,
        "meets_min_experience": int(years_exp >= config["min_experience_years"]),
        "education_score": edu_score,
        "cert_count": cert_count,
        "word_count": word_count,
        "keyword_density_score": (must_ratio * 0.5 + pref_ratio * 0.3
                                  + edu_score * 0.1 + min(years_exp / 10, 0.1))
    }


# ─────────────────────────────────────────────────────────────
# 3.  SYNTHETIC TRAINING DATA GENERATOR
#     Replace/extend with your own labeled resume dataset.
# ─────────────────────────────────────────────────────────────

def generate_synthetic_training_data(role: str = "data_scientist", n_samples: int = 500):
    """
    Generate labeled training data for demonstration.
    In production: replace this with real annotated resumes.
    """
    config = ROLE_KEYWORD_CONFIG[role]
    resumes, labels = [], []
    rng = np.random.default_rng(42)

    # Noise words to pad ineligible resumes (won't match keywords)
    filler_words = [
        "communication", "leadership", "team player", "hardworking", "motivated",
        "time management", "multitasking", "customer service", "retail", "sales",
        "administrative", "filing", "scheduling", "microsoft word", "ms office",
        "cashier", "inventory", "customer relations", "marketing", "advertising"
    ]

    for _ in range(n_samples):
        label = int(rng.choice([0, 1], p=[0.45, 0.55]))

        if label == 1:  # Eligible candidate
            n_must = rng.integers(int(len(config["must_have"]) * 0.6),
                                  len(config["must_have"]) + 1)
            must_kws = rng.choice(config["must_have"], n_must, replace=False)

            n_pref = rng.integers(int(len(config["preferred"]) * 0.3),
                                  int(len(config["preferred"]) * 0.7))
            pref_kws = rng.choice(config["preferred"], n_pref, replace=False)

            years = rng.integers(2, 12)
            edu = rng.choice(["Bachelor", "Master", "PhD"])
            cert = rng.choice(["", "AWS Certified", "Google Certified"], p=[0.5, 0.3, 0.2])

            text = (
                f"{edu} in Computer Science. {years} years of experience. "
                f"Skilled in {', '.join(must_kws)}. "
                f"Also experienced with {', '.join(pref_kws)}. "
                f"{cert}. Delivered multiple production ML models."
            )

        else:  # Not eligible
            # Ineligible: few or zero must-have keywords, lots of filler
            n_must = rng.integers(0, int(len(config["must_have"]) * 0.3) + 1)
            must_kws = rng.choice(config["must_have"], n_must, replace=False)

            years = rng.integers(0, 2)
            edu = rng.choice(["High School", "Diploma", "Associate"], p=[0.4, 0.35, 0.25])
            red_flag = rng.choice(config["red_flags"])

            n_filler = rng.integers(4, 10)
            filler = rng.choice(filler_words, n_filler, replace=False)

            tech_part = (f"Some exposure to {', '.join(must_kws)}."
                         if n_must > 0 else "No technical background.")

            text = (
                f"{edu} graduate. {years} years experience. "
                f"{tech_part} "
                f"Background in {', '.join(filler)}. "
                f"{red_flag}. Seeking entry-level opportunity."
            )

        resumes.append(text)
        labels.append(label)

    return resumes, labels


# ─────────────────────────────────────────────────────────────
# 4.  MODEL CLASSES
# ─────────────────────────────────────────────────────────────

class ResumeScreener:
    """
    Trains and evaluates both Logistic Regression and SVM models
    on resume text using TF-IDF + keyword features.
    """

    def __init__(self, role: str = "data_scientist"):
        self.role = role
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),    # unigrams + bigrams
            stop_words="english",
            min_df=2,
            sublinear_tf=True      # log(tf) scaling
        )
        self.models = {
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",  # handles class imbalance
                random_state=42,
                solver="lbfgs"
            ),
            "SVM": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,         # needed for predict_proba
                class_weight="balanced",
                random_state=42
            )
        }
        self.trained_models = {}
        self.is_fitted = False

    def _build_feature_matrix(self, texts: list, fit: bool = False) -> np.ndarray:
        """Combine TF-IDF features with hand-crafted keyword features."""
        # TF-IDF
        if fit:
            tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.vectorizer.transform(texts).toarray()

        # Keyword features
        kw_features = np.array([
            list(keyword_feature_engineering(t, self.role).values())
            for t in texts
        ])

        return np.hstack([tfidf_features, kw_features])

    def train(self, resumes: list, labels: list):
        """Train both LR and SVM; print evaluation report."""
        print(f"\n{'='*60}")
        print(f"  Resume Screening System — Role: {self.role.upper()}")
        print(f"{'='*60}")
        print(f"  Training samples : {len(resumes)}")
        print(f"  Eligible (1)     : {sum(labels)}")
        print(f"  Not Eligible (0) : {len(labels) - sum(labels)}")

        X = self._build_feature_matrix(resumes, fit=True)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"\n{'─'*60}")
            print(f"  MODEL: {name}")
            print(f"{'─'*60}")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
            print(f"  Cross-val F1 : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Fit on full train set
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Test set evaluation
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            print(f"\n  Test Set Results:")
            print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
            print(f"  F1 Score  : {f1_score(y_test, y_pred):.3f}")
            print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.3f}")

            print(f"\n  Classification Report:")
            report = classification_report(
                y_test, y_pred,
                target_names=["Not Eligible", "Eligible"],
                digits=3
            )
            for line in report.split("\n"):
                print(f"    {line}")

            print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
            cm = confusion_matrix(y_test, y_pred)
            print(f"    {'':20s}  Pred: Not Elig  Pred: Eligible")
            print(f"    {'Actual: Not Elig':22s}  {cm[0][0]:^14}  {cm[0][1]:^14}")
            print(f"    {'Actual: Eligible':22s}  {cm[1][0]:^14}  {cm[1][1]:^14}")

        self.is_fitted = True
        print(f"\n{'='*60}")
        print("  Training complete.")
        print(f"{'='*60}\n")

    def predict(self, resume_text: str, model_name: str = "Logistic Regression") -> dict:
        """
        Screen a single resume.

        Returns:
            dict with eligibility verdict, confidence score, and feature breakdown.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .train() before .predict().")

        model = self.trained_models[model_name]
        features = self._build_feature_matrix([resume_text], fit=False)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        kw_breakdown = keyword_feature_engineering(resume_text, self.role)
        config = ROLE_KEYWORD_CONFIG[self.role]
        text_lower = resume_text.lower()

        matched_must = [kw for kw in config["must_have"] if kw in text_lower]
        missing_must = [kw for kw in config["must_have"] if kw not in text_lower]
        matched_pref = [kw for kw in config["preferred"] if kw in text_lower]
        red_flags_found = [kw for kw in config["red_flags"] if kw in text_lower]

        return {
            "model_used": model_name,
            "eligible": bool(prediction),
            "verdict": "✅ ELIGIBLE" if prediction == 1 else "❌ NOT ELIGIBLE",
            "confidence": {
                "eligible_probability": round(float(probability[1]), 4),
                "not_eligible_probability": round(float(probability[0]), 4)
            },
            "feature_breakdown": kw_breakdown,
            "keyword_analysis": {
                "matched_must_have": matched_must,
                "missing_must_have": missing_must,
                "matched_preferred": matched_pref,
                "red_flags_found": red_flags_found
            },
            "recommendation": _build_recommendation(prediction, probability[1],
                                                      kw_breakdown, matched_must,
                                                      missing_must, red_flags_found)
        }

    def batch_predict(self, resumes: list, model_name: str = "Logistic Regression") -> pd.DataFrame:
        """Screen a list of resumes and return a ranked DataFrame."""
        results = [self.predict(r, model_name) for r in resumes]
        rows = []
        for i, res in enumerate(results):
            rows.append({
                "applicant_id": i + 1,
                "verdict": res["verdict"],
                "eligible": res["eligible"],
                "eligible_probability": res["confidence"]["eligible_probability"],
                "must_have_matched": len(res["keyword_analysis"]["matched_must_have"]),
                "preferred_matched": len(res["keyword_analysis"]["matched_preferred"]),
                "years_experience": res["feature_breakdown"]["years_experience"],
                "education_score": res["feature_breakdown"]["education_score"],
                "red_flags": len(res["keyword_analysis"]["red_flags_found"])
            })

        df = pd.DataFrame(rows).sort_values("eligible_probability", ascending=False)
        return df

    def save(self, path: str = "resume_screener.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path: str = "resume_screener.pkl") -> "ResumeScreener":
        with open(path, "rb") as f:
            return pickle.load(f)


def _build_recommendation(prediction, prob, features, matched_must,
                           missing_must, red_flags) -> str:
    """Generate a human-readable hiring recommendation."""
    lines = []

    if prediction == 1:
        if prob >= 0.80:
            lines.append("Strong candidate — recommend moving to interview stage immediately.")
        elif prob >= 0.60:
            lines.append("Suitable candidate — proceed with a technical screening call.")
        else:
            lines.append("Borderline eligible — consider a short assessment before deciding.")
    else:
        if prob <= 0.25:
            lines.append("Clearly under-qualified — recommend rejection.")
        else:
            lines.append("Candidate does not meet the minimum bar at this time.")

    if missing_must:
        lines.append(f"Missing critical keywords: {', '.join(missing_must[:5])}.")
    if red_flags:
        lines.append(f"Red flags detected: {', '.join(red_flags)}.")
    if features["years_experience"] == 0:
        lines.append("No verifiable work experience found in resume.")

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────
# 5.  COMPARISON UTILITY
# ─────────────────────────────────────────────────────────────

def compare_models(screener: ResumeScreener, resume_text: str) -> None:
    """Print side-by-side predictions from both models."""
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON FOR SUBMITTED RESUME")
    print(f"{'='*60}")

    for model_name in screener.trained_models:
        result = screener.predict(resume_text, model_name)
        print(f"\n  [{model_name}]")
        print(f"  Verdict   : {result['verdict']}")
        print(f"  Eligible% : {result['confidence']['eligible_probability']*100:.1f}%")
        print(f"  Must-have : {len(result['keyword_analysis']['matched_must_have'])} matched")
        print(f"  Preferred : {len(result['keyword_analysis']['matched_preferred'])} matched")
        print(f"  Red Flags : {result['keyword_analysis']['red_flags_found'] or 'None'}")
        print(f"  Advice    : {result['recommendation']}")


# ─────────────────────────────────────────────────────────────
# 6.  MAIN DEMO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ROLE = "data_scientist"

    # ── Generate training data ──────────────────────────────
    print("\nGenerating synthetic training data...")
    resumes, labels = generate_synthetic_training_data(role=ROLE, n_samples=600)

    # ── Train both models ───────────────────────────────────
    screener = ResumeScreener(role=ROLE)
    screener.train(resumes, labels)

    # ── Test resumes ────────────────────────────────────────
    TEST_RESUMES = [
        # Strong candidate
        """
        PhD in Computer Science from IIT Bombay. 7 years of experience in machine learning
        and data analysis. Proficient in Python, TensorFlow, PyTorch, scikit-learn, SQL,
        pandas, and numpy. Extensive experience with deep learning, NLP, and computer vision.
        Led a team of 5 data scientists at a fintech startup. AWS Certified ML Specialist.
        Published 4 research papers. Skilled in feature engineering, hyperparameter tuning,
        model training, and MLOps using Docker and Kubernetes. Strong background in statistics
        and A/B testing.
        """,

        # Weak / ineligible candidate
        """
        High school graduate with basic knowledge only. No programming experience.
        Worked as a retail store manager for 3 years. Familiar with Microsoft Word and Excel.
        Interested in switching to a tech career. No formal education in computer science.
        Looking for entry-level opportunity. Unfamiliar with coding.
        """,

        # Borderline candidate
        """
        Bachelor in Statistics. 2 years of experience. Knows Python, SQL, and basic
        data analysis. Has worked with pandas and numpy. Some exposure to machine learning
        concepts. No deep learning experience. Basic knowledge of scikit-learn.
        Created dashboards using Power BI.
        """
    ]

    LABELS = ["Strong Candidate", "Ineligible Candidate", "Borderline Candidate"]

    for i, (resume, label) in enumerate(zip(TEST_RESUMES, LABELS)):
        print(f"\n{'━'*60}")
        print(f"  TEST CASE {i+1}: {label}")
        print(f"{'━'*60}")
        compare_models(screener, resume)

    # ── Batch ranking ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BATCH SCREENING — RANKED LEADERBOARD (Logistic Regression)")
    print(f"{'='*60}")
    ranking = screener.batch_predict(TEST_RESUMES, model_name="Logistic Regression")
    print(ranking.to_string(index=False))

    print(f"\n{'='*60}")
    print("  BATCH SCREENING — RANKED LEADERBOARD (SVM)")
    print(f"{'='*60}")
    ranking_svm = screener.batch_predict(TEST_RESUMES, model_name="SVM")
    print(ranking_svm.to_string(index=False))

    # ── Save model ──────────────────────────────────────────
    screener.save("resume_screener.pkl")
    print("\nDone. Model serialized and ready for API integration.\n")
