
# 🏦 Application de Prédiction d’Approbation de Prêt

Cette application web permet de visualiser, explorer et analyser un dataset de demandes de prêts, ainsi que de prédire l’approbation d’un prêt grâce à des modèles de Machine Learning. Elle est construite avec **Streamlit**, **Plotly**, et **Scikit-Learn**, et s’appuie sur un mécanisme automatique de gestion des modèles via un fichier `metadata.json`.

---

## 🚀 Fonctionnalités

### 📊 Exploration des données
- Indicateurs clés (KPIs)
- Histogrammes, boxplots, bar charts, pie charts
- Matrice de corrélation interactive
- Filtres dynamiques sur les revenus et l’éducation
- Aperçu du dataset filtré en temps réel
- Téléchargement des données filtrées (CSV)

### 🔮 Prédiction d’approbation de prêt
- Formulaire complet de saisie :
  - Revenus
  - Co-applicant
  - Montant du prêt
  - Durée
  - Historique de crédit
  - Variables catégorielles (genre, statut marital, zone…)
- Feature engineering automatique :
  - Total income, LoanAmountToIncome
  - EMI, EMIToIncome
  - Transformations logarithmiques
  - Has_Coapplicant
- Alignement automatique des features selon `metadata.json`
- Encodage cohérent avec le dataset d'entraînement
- Chargement dynamique du modèle et du scaler
- Calcul de la probabilité d’approbation
- Barre de progression
- Graphique des 5 features les plus influentes

### 📈 Analyse modèle
- Affichage des caractéristiques des modèles
- Chargement dynamique de :
  - `logistic_regression.pkl`
  - `random_forest.pkl`
  - `scaler.pkl`
- Lecture automatique des performances depuis `metadata.json`

---

## 📂 Structure du projet
projet/
│── app.py
│── requirements.txt
│── README.md
│
├── data/
│ └── loan_data_clean.csv
│
├── models/
│ ├── logistic_regression.pkl
│ ├── random_forest.pkl
│ ├── scaler.pkl
│ └── metadata.json


---

## ⚙️ Installation

### 1) Créer un environnement virtuel

```bash
python -m venv venv

Activer l’environnement :
Windows PowerShell :

venv\Scripts\activate

Linux / macOS :

source venv/bin/activate
2) Installer les dépendances
pip install -r requirements.txt
3) Lancer l’application
streamlit run app.py

L’application s’ouvre automatiquement dans votre navigateur :
http://localhost:8501

📦 Fichier requirements.txt
streamlit==1.31.1
pandas==2.2.0
numpy==1.26.4
plotly==5.18.0
joblib==1.3.2
scikit-learn==1.4.0
🛠️ Technologies utilisées

Python 3.11+

Streamlit (interface web)

Pandas (manipulation des données)

Plotly (visualisations interactives)

Scikit-Learn (modèles ML)

Joblib (chargement des modèles)

Metadata JSON pour la configuration automatique

🔧 Personnalisation

Vous pouvez remplacer :

le dataset dans data/

les modèles dans models/

les informations dans metadata.json

L’application s’adapte automatiquement aux modèles déclarés dans le metadata.

📜 Licence

Projet éducatif et libre d’utilisation.

🤝 Contact

Pour toute question ou amélioration, n’hésitez pas à me contacter.