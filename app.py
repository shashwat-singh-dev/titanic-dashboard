import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")


st.set_page_config(page_title="Titanic Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['FamilySize'] = df['SibSp'] + df['Parch']
    return df

df = load_data()
selected_class = st.sidebar.selectbox("Filter by Class", ["All", 1, 2, 3])

if selected_class != "All":
    df = df[df['Pclass'] == selected_class]

st.title("🚢 Titanic Survival Analytics Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Passengers", len(df))
col2.metric("Survival Rate", f"{df['Survived'].mean()*100:.2f}%")
col3.metric("Avg Age", f"{df['Age'].mean():.1f}")

st.info("""
Key Insights:
- Females had significantly higher survival rate than males
- First class passengers survived the most
- Higher fare increased survival probability
- Small families had better survival chances
""")

option = st.sidebar.radio(
    "Choose Analysis",
    ["Overview", "Gender", "Class", "Age", "Fare", "Family"]
)

if option == "Overview":
    st.subheader("Dataset Preview")

    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=csv,
        file_name='cleaned_titanic.csv',
        mime='text/csv'
    )

elif option == "Gender":
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', hue='Sex', data=df, ax=ax)
    ax.set_title("Survival by Gender")
    st.pyplot(fig)

elif option == "Class":
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)
    ax.set_title("Survival by Class")
    st.pyplot(fig)

elif option == "Age":
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Age', hue='Survived', bins=20, ax=ax)
    ax.set_title("Age distribution")
    st.pyplot(fig)

elif option == "Fare":
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Fare', hue='Survived', bins=20, ax=ax)
    ax.set_title("Survival by Fare")
    st.pyplot(fig)

elif option == "Family":
    fig, ax = plt.subplots()
    sns.countplot(x='FamilySize', hue='Survived', data=df, ax=ax)
    ax.set_title("Survival by Family Size")
    st.pyplot(fig)
    