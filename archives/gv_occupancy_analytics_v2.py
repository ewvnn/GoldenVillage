import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_visualize_model(csv_path="data/gv_bookings.csv"):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # 2. Define Features
    categorical_features = ['day_of_week', 'time_slot', 'genre', 'location']
    numeric_features = ['ticket_price'] 
    target = 'occupancy_rate'
    
    df = df.dropna(subset=categorical_features + numeric_features + [target])
    X = df[categorical_features + numeric_features]
    y = df[target]
    
    # 3. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # 4. Pipeline setup
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0)) 
    ])
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Train & Predict
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    # Print Metrics
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE:      {mean_absolute_error(y_test, y_pred):.4f}")
    
    # 7. --- VISUALIZATION ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=y_test, alpha=0.5, color='blue')
    
    # Plot the ideal 'perfect prediction' line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    
    # Formatting
    plt.title('Multiple Linear Regression: Predicted vs. Actual Occupancy')
    plt.xlabel('Predicted Occupancy Rate')
    plt.ylabel('Actual Occupancy Rate')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return model_pipeline

if __name__ == "__main__":
    model = train_and_visualize_model()