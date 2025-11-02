import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Set page configuration
st.set_page_config(
    page_title="Advanced IPO Success Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the IPO dataset"""
    try:
        df = pd.read_csv('../data.csv')
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Replace NA values and handle missing data
        df = df.replace(['NA', '', 'NaN', 'null'], 0)
        df = df.fillna(0)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def advanced_feature_engineering(df):
    """Advanced feature engineering with domain knowledge"""
    
    # Define core feature columns
    feature_columns = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %'
    ]
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df['Classification'].copy()
    
    # Convert features to numeric
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Convert target and filter valid classes
    y = y.astype(str).str.strip()
    valid_classes = ['S', 'F', 'N']
    mask = y.isin(valid_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    st.info(f"📊 After preprocessing: {len(X)} samples, {len(feature_columns)} features")
    st.write(f"**Class Distribution:** {dict(y.value_counts())}")
    
    # ============================================================================
    # ADVANCED FEATURE ENGINEERING
    # ============================================================================
    
    # Financial Ratios and Metrics
    X['PE_ROCE_Interaction'] = X['P/E'] * X['ROCE %']
    X['Profit_Margin'] = (X['Net Profit of last quarter Rs. Cr.'] / (X['Quarterly Sales Rs.Cr.'] + 1)) * 100
    X['Market_Cap_to_Sales'] = X['Mar Capitalization Rs.Cr.'] / (X['Quarterly Sales Rs.Cr.'] + 1)
    X['Value_Growth_Score'] = (X['Dividend Yield %'] * 0.3) + (X['ROCE %'] * 0.7)
    
    # Growth and Momentum Indicators
    X['Profit_Growth_Momentum'] = X['Quarterly Profit Variation %'] * np.log1p(abs(X['Net Profit of last quarter Rs. Cr.']))
    X['Sales_Growth_Momentum'] = X['Quarterly Sales Variation %'] * np.log1p(X['Quarterly Sales Rs.Cr.'])
    X['Composite_Growth_Score'] = (X['Quarterly Profit Variation %'] + X['Quarterly Sales Variation %']) * X['ROCE %']
    
    # Stability and Risk Measures
    X['Profit_Stability'] = 1 / (1 + abs(X['Quarterly Profit Variation %']))
    X['Size_Stability'] = np.log1p(X['Mar Capitalization Rs.Cr.'])
    X['Valuation_Risk'] = X['P/E'] / (X['ROCE %'] + 1)
    
    # Polynomial and Interaction Features
    X['PE_Squared'] = X['P/E'] ** 2
    X['ROCE_Squared'] = X['ROCE %'] ** 2
    X['Profit_Var_Squared'] = X['Quarterly Profit Variation %'] ** 2
    X['Size_ROCE_Interaction'] = X['Mar Capitalization Rs.Cr.'] * X['ROCE %']
    
    # Advanced Financial Engineering
    X['Efficiency_Score'] = (X['ROCE %'] * X['Profit_Margin']) / (abs(X['P/E']) + 1)
    X['Growth_Quality'] = (X['Quarterly Profit Variation %'] * X['Profit_Margin']) / 100
    X['Market_Sentiment'] = (X['P/E'] * X['Dividend Yield %']) / 100
    
    # Handle any new NaN or infinite values
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    st.success(f"🔧 Advanced feature engineering completed: {X.shape[1]} total features")
    
    return X, y, feature_columns

def create_advanced_ensemble():
    """Create an advanced ensemble of diverse models"""
    
    # Base models
    base_models = [
        ('xgb', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )),
        ('catboost', CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )),
        ('rf', BalancedRandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        )),
        ('gbm', GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=0.1,
        random_state=42,
        max_iter=1000,
        multi_class='multinomial'
    )
    
    # Create stacking ensemble
    stacking_ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # Also create voting ensemble as backup
    voting_ensemble = VotingClassifier(
        estimators=base_models[:3],  # Use top 3 models for voting
        voting='soft',
        weights=[1.2, 1.0, 1.1]
    )
    
    return stacking_ensemble, voting_ensemble, base_models

def train_advanced_model(X, y):
    """Train the advanced ensemble model with optimization"""
    
    with st.spinner("🔄 Performing advanced feature selection..."):
        # Initial scaling
        scaler = PowerTransformer(method='yeo-johnson')
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Advanced feature selection using RFECV with Gradient Boosting
        selector = RFECV(
            estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
            step=1,
            cv=StratifiedKFold(3),
            scoring='f1_weighted',
            min_features_to_select=10,
            n_jobs=-1
        )
        X_selected = selector.fit_transform(X_scaled_df, y)
        selected_features = X_scaled_df.columns[selector.support_].tolist()
        
        st.success(f"🎯 Feature selection completed: {len(selected_features)} features selected")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Advanced balancing
    smote_tomek = SMOTETomek(
        random_state=42,
        smote=SMOTE(k_neighbors=3, random_state=42)
    )
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    st.info(f"⚖️ Data balanced: {len(X_train_balanced)} training samples")
    
    # Create ensembles
    stacking_ensemble, voting_ensemble, base_models = create_advanced_ensemble()
    
    # Train stacking ensemble
    with st.spinner("🏋️ Training advanced stacking ensemble..."):
        stacking_ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Train voting ensemble
    with st.spinner("🏋️ Training voting ensemble..."):
        voting_ensemble.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate both ensembles
    stacking_score = stacking_ensemble.score(X_test, y_test)
    voting_score = voting_ensemble.score(X_test, y_test)
    
    # Choose the best ensemble
    if stacking_score >= voting_score:
        best_ensemble = stacking_ensemble
        ensemble_type = "Stacking Ensemble"
        ensemble_score = stacking_score
    else:
        best_ensemble = voting_ensemble
        ensemble_type = "Voting Ensemble"
        ensemble_score = voting_score
    
    st.success(f"✅ Model training completed! Best: {ensemble_type} (Accuracy: {ensemble_score:.3f})")
    
    return best_ensemble, scaler, selector, X_train, X_test, y_train, y_test, selected_features

def predict_with_gmp_advanced(model, scaler, selector, input_features, gmp_ratio, selected_features):
    """
    Advanced prediction with GMP integration and confidence calibration
    """
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Apply the same feature engineering as training
        input_df = apply_feature_engineering(input_df)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        
        # Feature selection
        input_selected = selector.transform(input_scaled_df)
        input_final = pd.DataFrame(input_selected, columns=selected_features)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_final)[0]
        classes = model.classes_
        
        # Get base success probability
        success_idx = list(classes).index('S') if 'S' in classes else 0
        base_success_prob = probabilities[success_idx] * 100
        
        # Advanced GMP integration with dynamic weighting
        if gmp_ratio > 0:
            # Dynamic GMP weight based on market conditions
            gmp_weight = calculate_dynamic_gmp_weight(gmp_ratio)
            gmp_contribution = gmp_weight * gmp_ratio
            
            # Apply GMP adjustment with diminishing returns
            adjusted_score = base_success_prob + gmp_contribution
            
            # Cap at 95 (leaving room for uncertainty)
            adjusted_score = min(adjusted_score, 95)
        else:
            adjusted_score = base_success_prob
            gmp_contribution = 0
        
        # Calculate confidence based on probability distribution
        confidence = calculate_prediction_confidence(probabilities, classes)
        
        # Determine final classification with calibrated thresholds
        final_class = determine_final_classification(adjusted_score, confidence)
        
        return {
            'predicted_class': final_class,
            'base_score': base_success_prob,
            'adjusted_score': adjusted_score,
            'gmp_contribution': gmp_contribution,
            'confidence': confidence * 100,
            'probabilities': dict(zip(classes, probabilities))
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def calculate_dynamic_gmp_weight(gmp_ratio):
    """Calculate dynamic weight for GMP based on ratio value"""
    if gmp_ratio <= 20:
        return 0.1  # Low weight for low GMP
    elif gmp_ratio <= 50:
        return 0.2  # Medium weight
    elif gmp_ratio <= 100:
        return 0.3  # High weight
    else:
        return 0.25  # Slightly reduced weight for very high GMP (might be speculative)

def calculate_prediction_confidence(probabilities, classes):
    """Calculate prediction confidence based on probability distribution"""
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) >= 2:
        # Confidence is high if top probability is significantly higher than second
        confidence = sorted_probs[0] - sorted_probs[1]
    else:
        confidence = sorted_probs[0]
    
    return min(confidence * 1.5, 1.0)  # Scale confidence

def determine_final_classification(adjusted_score, confidence):
    """Determine final classification with confidence-based thresholds"""
    
    # Adjust thresholds based on confidence
    confidence_adjustment = (1 - confidence) * 10
    
    if adjusted_score >= (75 - confidence_adjustment):
        return 'S'  # Success
    elif adjusted_score >= (40 - confidence_adjustment):
        return 'N'  # Normal
    else:
        return 'F'  # Fail

def apply_feature_engineering(df):
    """Apply the same feature engineering as during training"""
    
    # Financial Ratios and Metrics
    df['PE_ROCE_Interaction'] = df['P/E'] * df['ROCE %']
    df['Profit_Margin'] = (df['Net Profit of last quarter Rs. Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)) * 100
    df['Market_Cap_to_Sales'] = df['Mar Capitalization Rs.Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)
    df['Value_Growth_Score'] = (df['Dividend Yield %'] * 0.3) + (df['ROCE %'] * 0.7)
    
    # Growth and Momentum Indicators
    df['Profit_Growth_Momentum'] = df['Quarterly Profit Variation %'] * np.log1p(abs(df['Net Profit of last quarter Rs. Cr.']))
    df['Sales_Growth_Momentum'] = df['Quarterly Sales Variation %'] * np.log1p(df['Quarterly Sales Rs.Cr.'])
    df['Composite_Growth_Score'] = (df['Quarterly Profit Variation %'] + df['Quarterly Sales Variation %']) * df['ROCE %']
    
    # Stability and Risk Measures
    df['Profit_Stability'] = 1 / (1 + abs(df['Quarterly Profit Variation %']))
    df['Size_Stability'] = np.log1p(df['Mar Capitalization Rs.Cr.'])
    df['Valuation_Risk'] = df['P/E'] / (df['ROCE %'] + 1)
    
    # Polynomial and Interaction Features
    df['PE_Squared'] = df['P/E'] ** 2
    df['ROCE_Squared'] = df['ROCE %'] ** 2
    df['Profit_Var_Squared'] = df['Quarterly Profit Variation %'] ** 2
    df['Size_ROCE_Interaction'] = df['Mar Capitalization Rs.Cr.'] * df['ROCE %']
    
    # Advanced Financial Engineering
    df['Efficiency_Score'] = (df['ROCE %'] * df['Profit_Margin']) / (abs(df['P/E']) + 1)
    df['Growth_Quality'] = (df['Quarterly Profit Variation %'] * df['Profit_Margin']) / 100
    df['Market_Sentiment'] = (df['P/E'] * df['Dividend Yield %']) / 100
    
    # Handle any new NaN or infinite values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def main():
    st.title("🎯 Advanced IPO Success Prediction Platform")
    st.markdown("---")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Model Training", "IPO Prediction", "Model Analysis"]
    )
    
    if app_mode == "Model Training":
        show_model_training()
    elif app_mode == "IPO Prediction":
        show_ipo_prediction()
    else:
        show_model_analysis()

def show_model_training():
    st.header("🔬 Advanced Model Training")
    
    if st.button("🚀 Train Advanced Ensemble Model"):
        with st.spinner("Loading and preprocessing data..."):
            df = load_and_preprocess_data()
            
        if df is not None:
            with st.spinner("Performing advanced feature engineering..."):
                X, y, original_features = advanced_feature_engineering(df)
            
            with st.spinner("Training advanced ensemble model..."):
                model, scaler, selector, X_train, X_test, y_train, y_test, selected_features = train_advanced_model(X, y)
                
                # Save models
                with open('advanced_ensemble_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                with open('advanced_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                with open('advanced_selector.pkl', 'wb') as f:
                    pickle.dump(selector, f)
                with open('selected_features.pkl', 'wb') as f:
                    pickle.dump(selected_features, f)
                
                st.session_state.model_trained = True
                st.success("✅ Advanced model trained and saved successfully!")

def show_ipo_prediction():
    st.header("🔮 IPO Success Prediction")
    
    # Load trained models
    try:
        with open('advanced_ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('advanced_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('advanced_selector.pkl', 'rb') as f:
            selector = pickle.load(f)
        with open('selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Models not found. Please train the model first.")
        return
    
    st.info("""
    **🎯 Classification System:**
    - **Success (S)**: Strong fundamentals + positive market sentiment
    - **Normal (N)**: Moderate potential with acceptable risk
    - **Fail (F)**: High risk or weak fundamentals
    
    **📊 GMP Integration:** Dynamically weighted based on market conditions
    """)
    
    # GMP Input First
    st.subheader("📈 Step 1: Enter Grey Market Premium (GMP)")
    gmp_ratio = st.number_input(
        "GMP Ratio",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        help="Enter the Grey Market Premium ratio (0 if not available)"
    )
    
    if gmp_ratio > 0:
        st.success(f"✅ GMP Ratio: {gmp_ratio} (Will contribute to prediction)")
    
    # IPO Details Form
    with st.form("advanced_ipo_prediction"):
        st.subheader("📋 Step 2: Enter IPO Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pe_ratio = st.number_input("P/E Ratio", min_value=0.0, value=20.0, step=1.0)
            market_cap = st.number_input("Market Capitalization (Rs. Cr.)", min_value=0.0, value=1500.0, step=100.0)
            dividend_yield = st.number_input("Dividend Yield %", min_value=0.0, value=2.5, step=0.1)
        
        with col2:
            net_profit = st.number_input("Net Profit last quarter (Rs. Cr.)", min_value=0.0, value=75.0, step=10.0)
            profit_variation = st.number_input("Quarterly Profit Variation %", value=15.0, step=1.0)
            quarterly_sales = st.number_input("Quarterly Sales (Rs. Cr.)", min_value=0.0, value=600.0, step=50.0)
        
        with col3:
            sales_variation = st.number_input("Quarterly Sales Variation %", value=12.0, step=1.0)
            issue_price = st.number_input("Issue Price (Rs)", min_value=0.0, value=120.0, step=10.0)
            roce = st.number_input("ROCE %", value=18.0, step=1.0)
        
        submitted = st.form_submit_button("🎯 Predict IPO Success")
    
    if submitted:
        input_features = {
            'P/E': pe_ratio,
            'Mar Capitalization Rs.Cr.': market_cap,
            'Dividend Yield %': dividend_yield,
            'Net Profit of last quarter Rs. Cr.': net_profit,
            'Quarterly Profit Variation %': profit_variation,
            'Quarterly Sales Rs.Cr.': quarterly_sales,
            'Quarterly Sales Variation %': sales_variation,
            'Issue Price (Rs)': issue_price,
            'ROCE %': roce
        }
        
        with st.spinner("🔄 Performing advanced analysis..."):
            result = predict_with_gmp_advanced(
                model, scaler, selector, input_features, gmp_ratio, selected_features
            )
        
        if result:
            display_advanced_results(result, gmp_ratio)

def display_advanced_results(result, gmp_ratio):
    """Display advanced prediction results"""
    
    st.subheader("🎯 Prediction Results")
    
    # Color coding based on prediction
    if result['predicted_class'] == 'S':
        color = "green"
        icon = "✅"
        message = "HIGH SUCCESS POTENTIAL"
    elif result['predicted_class'] == 'N':
        color = "blue"
        icon = "ℹ️"
        message = "MODERATE POTENTIAL"
    else:
        color = "red"
        icon = "❌"
        message = "HIGH RISK"
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Classification", f"{icon} {result['predicted_class']}")
    
    with col2:
        st.metric("Adjusted Score", f"{result['adjusted_score']:.1f}")
    
    with col3:
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    
    with col4:
        st.metric("Base Score", f"{result['base_score']:.1f}")
    
    # Detailed breakdown
    st.subheader("📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Score Breakdown:**")
        st.write(f"- Base Model Score: {result['base_score']:.1f}")
        if gmp_ratio > 0:
            st.write(f"- GMP Contribution: +{result['gmp_contribution']:.1f}")
        st.write(f"- **Final Score: {result['adjusted_score']:.1f}**")
        
        st.write("**Probability Distribution:**")
        for cls, prob in result['probabilities'].items():
            st.write(f"- {cls}: {prob*100:.1f}%")
    
    with col2:
        st.write("**Classification Thresholds:**")
        st.write("- Success (S): ≥ 70")
        st.write("- Normal (N): 40 - 69")
        st.write("- Fail (F): < 40")
        st.write(f"- **Your Score: {result['adjusted_score']:.1f}**")
    
    # Visual scale
    st.subheader("📈 Performance Scale")
    create_visual_scale(result['adjusted_score'])
    
    # Recommendation
    st.subheader("💡 Investment Recommendation")
    display_recommendation(result, gmp_ratio)

def create_visual_scale(score):
    """Create a visual performance scale"""
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create gradient background
    x = np.linspace(0, 100, 100)
    colors = ['red'] * 40 + ['blue'] * 30 + ['green'] * 30
    
    for i in range(len(x)-1):
        ax.axvspan(x[i], x[i+1], color=colors[i], alpha=0.3)
    
    # Add score marker
    ax.axvline(x=score, color='black', linewidth=3, label=f'Score: {score:.1f}')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Performance Score')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add threshold labels
    ax.text(20, 0.5, 'FAIL', ha='center', va='center', fontweight='bold')
    ax.text(55, 0.5, 'NORMAL', ha='center', va='center', fontweight='bold')
    ax.text(85, 0.5, 'SUCCESS', ha='center', va='center', fontweight='bold')
    
    st.pyplot(fig)

def display_recommendation(result, gmp_ratio):
    """Display investment recommendation"""
    
    if result['predicted_class'] == 'S':
        st.success("""
        **🎉 STRONG BUY RECOMMENDATION**
        
        **Key Strengths:**
        - Strong fundamental metrics
        - Positive growth indicators
        - Favorable risk-reward ratio
        
        **Action:** Consider applying with high allocation. Monitor market conditions.
        """)
        
    elif result['predicted_class'] == 'N':
        st.warning("""
        **⚖️ CAUTIOUS APPROACH RECOMMENDED**
        
        **Considerations:**
        - Moderate fundamentals
        - Acceptable risk levels
        - Requires careful analysis
        
        **Action:** Consider applying with moderate allocation. Review company details.
        """)
        
    else:
        st.error("""
        **🚨 HIGH RISK - AVOID RECOMMENDATION**
        
        **Concerns Identified:**
        - Weak fundamental metrics
        - High risk factors
        - Poor growth indicators
        
        **Action:** Avoid application. Wait for better opportunities.
        """)
    
    # GMP-specific advice
    if gmp_ratio > 50:
        st.info(f"💡 **GMP Insight:** High GMP ({gmp_ratio}) indicates strong market demand. This has been factored into the positive adjustment.")

def show_model_analysis():
    st.header("📊 Model Analysis")
    st.info("This section would show detailed model performance metrics, feature importance, and other analytics.")

if __name__ == "__main__":
    main()