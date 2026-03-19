# 1.------IMOPRT LIBRARIES------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#2. -------LOAD DATASET----------

def load_data():
    try:
        return pd.read_csv("titanic.csv")
    except FileNotFoundError:
        print("❌ Error: titanic.csv file not found!")
        exit()

#3.------ DATA CLEANING-----------

def clean_data(df):
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Fill missing values
  if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    #Log Transform
    df['Fare_log'] = np.log1p(df['Fare'])
    
    # Drop useless columns
    df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)
    
    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

# 4. ----DATA VISUALIZATION-----

def visualize(df):

    df_temp = df.copy()
    df_temp['Sex_label'] = df_temp['Sex'].map({0: 'Male', 1: 'Female'})

    plt.figure(figsize=(8,5))
    sns.countplot(x='Survived', hue='Sex_label', data=df_temp)
    plt.title("Survival by Gender")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.countplot(x='Pclass', hue='Survived', data=df_temp)
    plt.title("Survival by Class")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.countplot(x='Pclass', hue='Sex_label', data=df_temp[df_temp['Survived'] == 1])
    plt.title("Survived People: Class vs Gender")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(data=df_temp, x='Age', hue='Survived', bins=20)
    plt.title("Survival by Age")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(data=df_temp, x='Fare_log', hue='Survived', bins=20)
    plt.title("Survival by Log(Fare)")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.countplot(x='FamilySize', hue='Survived', data=df_temp,
    order=sorted(df_temp['FamilySize'].unique()))
    plt.title("Survival by Family Size")
    plt.show()

#5.------- MAIN FUNCTION--------
def main():
    df = load_data()
    df = clean_data(df)
    visualize(df)

    print("\n----- KEY INSIGHTS -----")
    print("1. Females survived more than males")
    print("2. First-class passengers had highest survival rate")
    print("3. Higher fare passengers had better survival chances")
    print("4. Children had slightly better survival chances")
    print("5. Small families had better survival compared to large families")

    print("\nSurvival Rate by Gender:")
    print(df.groupby('Sex')['Survived'].mean().round(2))

    df.to_csv("cleaned_titanic.csv", index=False)

if __name__ == "__main__":
    main()
