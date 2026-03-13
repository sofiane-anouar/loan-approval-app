"""
Application Streamlit complète

- Structure Streamlit (TP5)
- Dashboard d'exploration interactif (TP6)
"""

import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

# ============================================================================
# CONFIGURATION PAGE
# ============================================================================

st.set_page_config(
    page_title="Prédiction d'Approbation de Prêt",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# HEADER
# ============================================================================

st.title("🏦 Prédiction d'Approbation de Prêt")

st.markdown(
    "Cette application permet d'explorer un jeu de données de prêts, "
    "de sélectionner un modèle et d'afficher des informations sur ses performances."
)

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Choix du modèle",
    ["Régression Logistique", "Random Forest"],
    index=0,
)

if model_choice == "Régression Logistique":
    st.sidebar.info("📊 Modèle linéaire, interprétable")
else:
    st.sidebar.info("🌳 Modèle ensemble, plus puissant")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📖 À propos")
st.sidebar.markdown(
    "Application pédagogique Streamlit (TP5 + TP6). "
    "Exploration des données, visualisations & ML."
)

# ============================================================================
# FONCTIONS DE CHARGEMENT
# ============================================================================

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/loan_data_clean.csv")
    except FileNotFoundError:
        st.error("❌ Fichier `data/loan_data_clean.csv` introuvable.")
        return None


@st.cache_resource
def load_metadata():
    try:
        with open("models/metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        st.error("❌ metadata.json introuvable dans /models.")
        return None


@st.cache_resource
def load_model(model_name):
    """Charge modèle + scaler via metadata.json"""

    metadata = load_metadata()

    if metadata is None:
        return None, None, None

    try:

        if model_name == "Régression Logistique":
            info = metadata["models"]["logistic_regression"]
        else:
            info = metadata["models"]["random_forest"]

        model_path = info["file"]
        scaler_path = metadata["scaler"]["file"]

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        feature_names = metadata["feature_names"]

        return model, scaler, feature_names

    except Exception as e:

        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None, None


# ============================================================================
# CHARGEMENT DATA
# ============================================================================

df = load_data()

if df is not None:

    tab1, tab2, tab3 = st.tabs(
        ["📊 Exploration", "🔮 Prédiction", "📈 Performance"]
    )

    # ============================================================================
    # EXPLORATION
    # ============================================================================

    with tab1:

        st.subheader("📊 Indicateurs Clés")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Demandes", len(df))

        with col2:
            approval_rate = (df["Loan_Status"].eq("Y").mean() * 100).round(2)
            st.metric("Taux d'Approbation", f"{approval_rate}%")

        with col3:
            st.metric("Montant Moyen", f"{df['LoanAmount'].mean().round(2)} €")

        with col4:
            st.metric("Revenu Moyen", f"{df['ApplicantIncome'].mean().round(2)} €")

        # HISTOGRAMME

        st.subheader("📈 Distribution des Revenus")

        fig_inc = px.histogram(
            df,
            x="ApplicantIncome",
            nbins=30,
            title="Distribution des Revenus des Demandeurs",
            labels={"ApplicantIncome": "Revenu (€)", "count": "Nombre"},
            color_discrete_sequence=["#3498db"],
        )

        fig_inc.add_vline(
            x=df["ApplicantIncome"].mean(),
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text="Moyenne",
        )

        st.plotly_chart(fig_inc, use_container_width=True)

        # BOX PLOT

        st.subheader("📦 Montants de Prêt Demandés")

        fig_box = px.box(
            df,
            y="LoanAmount",
            title="Distribution des Montants de Prêt",
            labels={"LoanAmount": "Montant du Prêt (€)"},
            color_discrete_sequence=["#8e44ad"],
        )

        median = df["LoanAmount"].median()
        q1 = df["LoanAmount"].quantile(0.25)
        q3 = df["LoanAmount"].quantile(0.75)

        fig_box.add_annotation(x=0, y=median, text=f"Médiane : {median}", showarrow=False)
        fig_box.add_annotation(x=0, y=q1, text=f"Q1 : {q1}", showarrow=False)
        fig_box.add_annotation(x=0, y=q3, text=f"Q3 : {q3}", showarrow=False)

        st.plotly_chart(fig_box, use_container_width=True)

        # EDUCATION

        st.subheader("🎓 Approbation selon l'Éducation")

        edu_df = df.groupby(["Education", "Loan_Status"]).size().reset_index(name="Count")
        edu_df["Percentage"] = edu_df.groupby("Education")["Count"].transform(lambda x: x/x.sum()*100)

        edu_yes = edu_df[edu_df["Loan_Status"] == "Y"]

        fig_bar = px.bar(
            edu_yes,
            x="Education",
            y="Percentage",
            color="Education",
            title="Taux d'Approbation par Niveau d'Éducation",
            labels={"Percentage": "Taux (%)"},
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # PIE

        st.subheader("🥧 Répartition Approuvé / Rejeté")

        counts = df["Loan_Status"].value_counts()

        fig_pie = px.pie(
            names=counts.index,
            values=counts.values,
            title="Répartition des Décisions",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.4,
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # CORRELATION

        st.subheader("🔥 Matrice de Corrélation")

        num_df = df.select_dtypes(include=["float64", "int64"])
        corr = num_df.corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
            )
        )

        fig_corr.update_layout(title="Matrice de Corrélation")

        st.plotly_chart(fig_corr, use_container_width=True)

        # FILTRES

        st.markdown("---")
        st.subheader("🔍 Filtres Interactifs")

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            min_val = int(df["ApplicantIncome"].min())
            max_val = int(df["ApplicantIncome"].max())

            income_range = st.slider(
                "Filtrer par revenu",
                min_val,
                max_val,
                (min_val, max_val),
            )

        with col_f2:
            selected_edu = st.multiselect(
                "Niveau d'éducation",
                df["Education"].unique(),
            )

        filtered_df = df[
            (df["ApplicantIncome"] >= income_range[0]) &
            (df["ApplicantIncome"] <= income_range[1])
        ]

        if selected_edu:
            filtered_df = filtered_df[
                filtered_df["Education"].isin(selected_edu)
            ]

        st.write("📄 Données filtrées :")

        st.dataframe(filtered_df)

        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="loan_data_filtered.csv",
            mime="text/csv",
        )

    # ============================================================================
    # PREDICTION
    # ============================================================================

    with tab2:

        st.header("🤖 Prédiction d'Approbation de Prêt")

        st.markdown(
            "Remplissez les informations ci-dessous pour prédire l'approbation du prêt."
        )

        st.markdown("### 📝 Formulaire")

        col1, col2 = st.columns(2)

        with col1:

            applicant_income = st.number_input(
                "Revenu du demandeur (€)",
                min_value=0,
                value=5000,
                step=100,
            )

            coapp_income = st.number_input(
                "Revenu du co-demandeur (€)",
                min_value=0,
                value=0,
                step=100,
            )

            loan_amount = st.number_input(
                "Montant du prêt demandé (€)",
                min_value=0,
                value=120,
                step=5,
            )

            loan_term = st.selectbox(
                "Durée du prêt",
                [360, 180, 120, 84, 60],
                index=0,
            )

            credit_history = st.selectbox(
                "Historique de crédit",
                [1.0, 0.0],
                format_func=lambda x: "Bon" if x == 1.0 else "Mauvais",
            )

        with col2:

            gender_male = st.selectbox("Genre", ["Homme", "Femme"])

            married_yes = st.selectbox(
                "Statut marital",
                ["Marié", "Non marié"],
            )

            self_emp = st.selectbox(
                "Travail indépendant",
                ["Oui", "Non"],
            )

            area = st.selectbox(
                "Zone géographique",
                ["Rural", "Semiurban", "Urban"],
            )

        st.markdown("---")

        if st.button("🔮 Prédire", use_container_width=True):

            st.info("⏳ Préparation des données...")

            input_df = pd.DataFrame([{

                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapp_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,

                "TotalIncome": applicant_income + coapp_income,

                "LoanAmountToIncome": loan_amount / (applicant_income + coapp_income + 1),

                "EMI": loan_amount / (loan_term + 1),

                "EMIToIncome":
                    (loan_amount / (loan_term + 1))
                    / (applicant_income + coapp_income + 1),

                "Log_LoanAmount": np.log(loan_amount + 1),

                "Log_TotalIncome": np.log(applicant_income + coapp_income + 1),

                "Has_Coapplicant": 1 if coapp_income > 0 else 0,

                "Gender_Male": 1 if gender_male == "Homme" else 0,

                "Married_Yes": 1 if married_yes == "Marié" else 0,

                "SelfEmployed_Yes": 1 if self_emp == "Oui" else 0,

                "Area_Semiurban": 1 if area == "Semiurban" else 0,

                "Area_Urban": 1 if area == "Urban" else 0,
            }])

            model, scaler, expected_features = load_model(model_choice)

            if model is None:
                st.error("❌ Impossible de charger le modèle.")
                st.stop()

            input_df = input_df.reindex(columns=expected_features, fill_value=0)

            if model_choice == "Régression Logistique":
                X_scaled = scaler.transform(input_df)
            else:
                X_scaled = input_df

            progress = st.progress(0)
            progress.progress(100)

            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0][1]

            if pred == 1:
                st.success(f"🎉 Prêt APPROUVÉ — Probabilité : {proba:.2%}")
            else:
                st.error(f"❌ Prêt REJETÉ — Probabilité : {proba:.2%}")

            st.markdown("---")

            st.subheader("🧠 Importance des caractéristiques")

            if model_choice == "Random Forest":
                importances = model.feature_importances_
            else:
                importances = abs(model.coef_[0])

            feat_imp = pd.DataFrame({

                "Feature": input_df.columns,
                "Importance": importances,

            }).sort_values("Importance", ascending=False).head(5)

            fig = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 5 des variables influentes",
            )

            st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # PERFORMANCE
    # ============================================================================

    with tab3:

        st.header("📈 Performance du Modèle")
        st.write("Contenu déjà implémenté au TP5.")

else:
    st.error("❌ Impossible de charger les données.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.markdown(
    "<div style='text-align:center'>© 2026 • Application pédagogique Streamlit</div>",
    unsafe_allow_html=True,
)
sofiane ="ak fahem"