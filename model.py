import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('Salary_Data.csv')

# Drop rows with missing Salary
df.dropna(subset=['Salary'], inplace=True)

# Label Encode categorical columns
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education Level'] = le_edu.fit_transform(df['Education Level'])
df['Job Title'] = le_job.fit_transform(df['Job Title'])

# Features and Target
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'salary_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_edu, 'le_edu.pkl')
joblib.dump(le_job, 'le_job.pkl')
