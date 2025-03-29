import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.mosaicplot import mosaic
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_data(file_path):
    """
    Load the thyroid cancer dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def basic_statistics(df):
    """
    Compute and display basic statistical measures for the dataset
    """
    print("\n=== BASIC STATISTICAL ANALYSIS ===")
    
    # 1. Numerical features - centrality and dispersion
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    
    stats_df = pd.DataFrame({
        'Mean': df[numerical_features].mean(),
        'Median': df[numerical_features].median(),
        'Std Dev': df[numerical_features].std(),
        'Min': df[numerical_features].min(),
        'Max': df[numerical_features].max(),
        'IQR': df[numerical_features].quantile(0.75) - df[numerical_features].quantile(0.25),
        'Skewness': df[numerical_features].skew(),
        'Kurtosis': df[numerical_features].kurtosis()
    })
    
    print("\nNumerical Features - Statistical Measures:")
    print(stats_df)
    
    # 2. Categorical features - frequency distribution
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_features:
        print(f"\nFrequency Distribution for {col}:")
        freq_df = df[col].value_counts(normalize=True).reset_index()
        freq_df.columns = [col, 'Proportion']
        freq_df['Count'] = df[col].value_counts().values
        freq_df['Proportion'] = freq_df['Proportion'] * 100
        print(freq_df)
    
    # 3. Save statistics to CSV
    stats_df.to_csv('thyroid_numerical_statistics.csv')
    print("\nNumerical statistics saved to 'thyroid_numerical_statistics.csv'")
    
    return stats_df

def normality_tests(df):
    """
    Perform normality tests on numerical features
    """
    print("\n=== NORMALITY TESTS ===")
    
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove Patient_ID if present as it's not meaningful for normality tests
    if 'Patient_ID' in numerical_features:
        numerical_features.remove('Patient_ID')
        
    normality_results = []
    
    for col in numerical_features:
        # Take a sample if dataset is large to avoid extreme sensitivity
        if len(df) > 5000:
            sample = df[col].dropna().sample(5000, random_state=42)
        else:
            sample = df[col].dropna()
            
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        
        # D'Agostino-Pearson test
        k2_stat, k2_p = stats.normaltest(sample)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(sample, 'norm', args=(np.mean(sample), np.std(sample)))
        
        normality_results.append({
            'Feature': col,
            'Shapiro_Stat': shapiro_stat,
            'Shapiro_p': shapiro_p,
            'Normal_by_Shapiro': shapiro_p > 0.05,
            'K2_Stat': k2_stat, 
            'K2_p': k2_p,
            'Normal_by_K2': k2_p > 0.05,
            'KS_Stat': ks_stat,
            'KS_p': ks_p,
            'Normal_by_KS': ks_p > 0.05,
            'Skewness': sample.skew(),
            'Kurtosis': sample.kurtosis()
        })
    
    normality_df = pd.DataFrame(normality_results)
    print("\nNormality Test Results:")
    print(normality_df[['Feature', 'Normal_by_Shapiro', 'Normal_by_K2', 'Normal_by_KS', 'Skewness', 'Kurtosis']])
    
    # Save results
    normality_df.to_csv('thyroid_normality_tests.csv', index=False)
    print("\nNormality test results saved to 'thyroid_normality_tests.csv'")
    
    # Plot histograms with normal curves for visual inspection
    for col in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, stat='density')
        
        # Add normal curve
        x = np.linspace(df[col].min(), df[col].max(), 100)
        y = stats.norm.pdf(x, df[col].mean(), df[col].std())
        plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')
        
        plt.title(f'Distribution of {col} with Normal Curve')
        plt.legend()
        plt.savefig(f'distribution_{col}.png')
        plt.close()
    
    # Create interactive QQ plots
    for col in numerical_features:
        # Calculate theoretical quantiles
        data = df[col].dropna()
        qq = stats.probplot(data, dist='norm')
        
        # Create QQ plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode='markers',
            name='Data',
            marker=dict(color='blue')
        ))
        
        # Add reference line
        fig.add_trace(go.Scatter(
            x=qq[0][0],
            y=qq[0][0] * qq[1][0] + qq[1][1],
            mode='lines',
            name='Reference Line',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Q-Q Plot for {col}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            template='plotly_white'
        )
        
        fig.write_html(f'qq_plot_{col}.html')
    
    print("\nDistribution plots and Q-Q plots created for all numerical features")
    return normality_df

def group_comparisons(df):
    """
    Analyze differences between benign and malignant groups
    """
    print("\n=== GROUP COMPARISON ANALYSIS ===")
    
    # Group by diagnosis
    groups = df.groupby('Diagnosis')
    
    # Compare numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Patient_ID' in numerical_features:
        numerical_features.remove('Patient_ID')
    
    # 1. Calculate summary statistics by group
    group_stats = groups[numerical_features].agg(['count', 'mean', 'median', 'std'])
    print("\nSummary Statistics by Diagnosis Group:")
    print(group_stats)
    
    # 2. Perform statistical tests
    test_results = []
    
    for col in numerical_features:
        # Get data for each group
        benign_data = df[df['Diagnosis'] == 'Benign'][col].dropna()
        malignant_data = df[df['Diagnosis'] == 'Malignant'][col].dropna()
        
        # Check if enough data points
        if len(benign_data) > 1 and len(malignant_data) > 1:
            # Check normality to decide on test
            _, shapiro_p_benign = stats.shapiro(benign_data.sample(min(5000, len(benign_data)), random_state=42))
            _, shapiro_p_malignant = stats.shapiro(malignant_data.sample(min(5000, len(malignant_data)), random_state=42))
            
            is_normal = (shapiro_p_benign > 0.05) and (shapiro_p_malignant > 0.05)
            
            # Check homogeneity of variance
            _, levene_p = stats.levene(benign_data, malignant_data)
            equal_var = levene_p > 0.05
            
            # Effect size calculation (Cohen's d)
            effect_size = abs(benign_data.mean() - malignant_data.mean()) / np.sqrt((benign_data.var() + malignant_data.var()) / 2)
            
            # Choose appropriate test
            if is_normal and equal_var:
                # Use t-test for normal distributions with equal variance
                stat, p_value = stats.ttest_ind(benign_data, malignant_data, equal_var=True)
                test_name = "Student's t-test"
            elif is_normal and not equal_var:
                # Use Welch's t-test for normal distributions with unequal variance
                stat, p_value = stats.ttest_ind(benign_data, malignant_data, equal_var=False)
                test_name = "Welch's t-test"
            else:
                # Use Mann-Whitney U test for non-normal distributions
                stat, p_value = stats.mannwhitneyu(benign_data, malignant_data)
                test_name = "Mann-Whitney U test"
            
            test_results.append({
                'Feature': col,
                'Test': test_name,
                'Statistic': stat,
                'p-value': p_value,
                'Significant (p<0.05)': p_value < 0.05,
                'Effect Size (Cohen\'s d)': effect_size,
                'Effect Magnitude': 'Small' if effect_size < 0.5 else ('Medium' if effect_size < 0.8 else 'Large')
            })
    
    test_results_df = pd.DataFrame(test_results)
    print("\nStatistical Test Results:")
    print(test_results_df)
    
    # Save results
    test_results_df.to_csv('thyroid_group_comparison_tests.csv', index=False)
    print("\nGroup comparison results saved to 'thyroid_group_comparison_tests.csv'")
    
    # 3. Visualize comparisons
    for col in numerical_features:
        # Box plot comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Diagnosis', y=col, data=df)
        plt.title(f'Comparison of {col} by Diagnosis')
        plt.savefig(f'boxplot_{col}_by_diagnosis.png')
        plt.close()
        
        # Violin plot for distribution comparison
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Diagnosis', y=col, data=df, inner='quartile')
        plt.title(f'Distribution of {col} by Diagnosis')
        plt.savefig(f'violinplot_{col}_by_diagnosis.png')
        plt.close()
    
    # 4. Create interactive visualization for all numerical features
    features_by_diagnosis = {}
    
    for col in numerical_features:
        fig = px.box(df, x='Diagnosis', y=col, color='Diagnosis',
                    title=f'Comparison of {col} by Diagnosis',
                    points='all', template='plotly_white')
        
        fig.update_layout(boxmode='group')
        fig.write_html(f'interactive_boxplot_{col}_by_diagnosis.html')
        
        # Calculate means for table
        features_by_diagnosis[col] = {
            'Benign': df[df['Diagnosis'] == 'Benign'][col].mean(),
            'Malignant': df[df['Diagnosis'] == 'Malignant'][col].mean(),
            'Difference': df[df['Diagnosis'] == 'Malignant'][col].mean() - df[df['Diagnosis'] == 'Benign'][col].mean(),
            'p-value': test_results_df[test_results_df['Feature'] == col]['p-value'].values[0] if not test_results_df[test_results_df['Feature'] == col].empty else np.nan
        }
    
    # Create summary table
    features_summary = pd.DataFrame(features_by_diagnosis).T
    features_summary['Significant'] = features_summary['p-value'] < 0.05
    features_summary = features_summary.sort_values('p-value')
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Feature', 'Benign Mean', 'Malignant Mean', 'Difference', 'p-value', 'Significant'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                features_summary.index,
                features_summary['Benign'].round(3),
                features_summary['Malignant'].round(3),
                features_summary['Difference'].round(3),
                features_summary['p-value'].round(6),
                features_summary['Significant']
            ],
            fill_color=[['white', 'lightgrey'] * len(features_summary)],
            align='left'
        )
    )])
    
    fig.update_layout(title='Summary of Numerical Features by Diagnosis')
    fig.write_html('diagnosis_comparison_summary.html')
    
    print("\nInteractive comparison visualizations created for all numerical features")
    
    # 5. Categorical features - Chi-square tests and association measures
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove Diagnosis from categorical features
    if 'Diagnosis' in categorical_features:
        categorical_features.remove('Diagnosis')
    
    categorical_test_results = []
    
    for col in categorical_features:
        # Create contingency table
        contingency_table = pd.crosstab(df[col], df['Diagnosis'])
        
        # Perform Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramer's V (association measure)
        n = contingency_table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        # Store results
        categorical_test_results.append({
            'Feature': col,
            'Chi-square': chi2,
            'p-value': p,
            'Significant (p<0.05)': p < 0.05,
            'Cramer\'s V': cramer_v,
            'Association Strength': 'Weak' if cramer_v < 0.1 else ('Moderate' if cramer_v < 0.3 else 'Strong')
        })
    
    categorical_test_df = pd.DataFrame(categorical_test_results)
    print("\nCategorical Feature Association with Diagnosis:")
    print(categorical_test_df)
    
    # Save results
    categorical_test_df.to_csv('thyroid_categorical_associations.csv', index=False)
    print("\nCategorical association results saved to 'thyroid_categorical_associations.csv'")
    
    # 6. Visualize top categorical associations
    top_categorical = categorical_test_df.sort_values('Cramer\'s V', ascending=False).head(5)['Feature'].tolist()
    
    for col in top_categorical:
        # Create mosaic plot
        plt.figure(figsize=(10, 8))
        mosaic(df, [col, 'Diagnosis'])
        plt.title(f'Association between {col} and Diagnosis')
        plt.savefig(f'mosaic_{col}_diagnosis.png')
        plt.close()
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        pd.crosstab(df[col], df['Diagnosis'], normalize='index').plot(kind='bar', stacked=True)
        plt.title(f'Proportion of Diagnosis by {col}')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(f'stacked_bar_{col}_diagnosis.png')
        plt.close()
        
        # Create interactive stacked bar
        df_crosstab = pd.crosstab(df[col], df['Diagnosis'], normalize='index')
        df_crosstab = df_crosstab.reset_index()
        df_crosstab_melted = pd.melt(df_crosstab, id_vars=[col], var_name='Diagnosis', value_name='Proportion')
        
        fig = px.bar(df_crosstab_melted, x=col, y='Proportion', color='Diagnosis',
                    title=f'Proportion of Diagnosis by {col}',
                    barmode='stack', template='plotly_white')
        
        fig.update_layout(yaxis_tickformat='.0%')
        fig.write_html(f'interactive_stacked_bar_{col}_diagnosis.html')
    
    return {
        'numerical_tests': test_results_df,
        'categorical_tests': categorical_test_df
    }

def correlation_analysis(df):
    """
    Analyze correlations between variables
    """
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Numeric features for correlation
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Patient_ID' in numerical_features:
        numerical_features.remove('Patient_ID')
    
    # 1. Calculate correlation matrices
    # Pearson correlation
    pearson_corr = df[numerical_features].corr(method='pearson')
    print("\nPearson Correlation Matrix:")
    print(pearson_corr)
    
    # Spearman rank correlation (robust to non-normality)
    spearman_corr = df[numerical_features].corr(method='spearman')
    print("\nSpearman Correlation Matrix:")
    print(spearman_corr)
    
    # Plot correlation heatmaps
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Pearson Correlation Matrix')
    plt.tight_layout()
    plt.savefig('pearson_correlation_matrix.png')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig('spearman_correlation_matrix.png')
    plt.close()
    
    # 2. Create interactive correlation heatmaps
    fig = px.imshow(pearson_corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Pearson Correlation Matrix')
    fig.write_html('interactive_pearson_correlation.html')
    
    fig = px.imshow(spearman_corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Spearman Correlation Matrix')
    fig.write_html('interactive_spearman_correlation.html')
    
    # 3. Find significant correlations
    # Calculate p-values for Pearson correlations
    pearson_p_values = pd.DataFrame(np.zeros_like(pearson_corr), 
                                   index=pearson_corr.index, 
                                   columns=pearson_corr.columns)
    
    for i, feature_i in enumerate(numerical_features):
        for j, feature_j in enumerate(numerical_features):
            if i != j:  # Skip diagonal
                pearson_r, p_value = stats.pearsonr(df[feature_i].dropna(), df[feature_j].dropna())
                pearson_p_values.iloc[i, j] = p_value
    
    # Identify significant correlations
    significant_corrs = []
    
    for i, feature_i in enumerate(numerical_features):
        for j, feature_j in enumerate(numerical_features):
            if i < j:  # Only consider upper triangle of correlation matrix
                pearson_r = pearson_corr.iloc[i, j]
                p_value = pearson_p_values.iloc[i, j]
                
                if p_value < 0.05:
                    significant_corrs.append({
                        'Feature 1': feature_i,
                        'Feature 2': feature_j,
                        'Pearson r': pearson_r,
                        'p-value': p_value,
                        'Correlation Strength': 'Weak' if abs(pearson_r) < 0.3 else ('Moderate' if abs(pearson_r) < 0.7 else 'Strong'),
                        'Direction': 'Positive' if pearson_r > 0 else 'Negative'
                    })
    
    significant_corrs_df = pd.DataFrame(significant_corrs)
    
    if not significant_corrs_df.empty:
        significant_corrs_df = significant_corrs_df.sort_values('Pearson r', key=abs, ascending=False)
        print("\nSignificant Correlations:")
        print(significant_corrs_df)
        
        # Save results
        significant_corrs_df.to_csv('thyroid_significant_correlations.csv', index=False)
        print("\nSignificant correlations saved to 'thyroid_significant_correlations.csv'")
        
        # Visualize top correlations with scatter plots
        top_corrs = significant_corrs_df.head(5)
        
        for _, row in top_corrs.iterrows():
            feature1 = row['Feature 1']
            feature2 = row['Feature 2']
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=feature1, y=feature2, data=df, hue='Diagnosis', alpha=0.7)
            plt.title(f'Correlation between {feature1} and {feature2} (r={row["Pearson r"]:.2f}, p={row["p-value"]:.4f})')
            plt.savefig(f'scatter_{feature1}_{feature2}.png')
            plt.close()
            
            # Create interactive scatter plot
            fig = px.scatter(df, x=feature1, y=feature2, color='Diagnosis',
                            title=f'Correlation between {feature1} and {feature2} (r={row["Pearson r"]:.2f}, p={row["p-value"]:.4f})',
                            template='plotly_white', opacity=0.7,
                            trendline='ols', trendline_scope='overall')
            
            fig.write_html(f'interactive_scatter_{feature1}_{feature2}.html')
    else:
        print("\nNo significant correlations found.")
    
    # 4. Point-biserial correlation with Diagnosis (treat Diagnosis as binary)
    diagnosis_num = pd.Series(np.where(df['Diagnosis'] == 'Malignant', 1, 0), index=df.index)
    
    point_biserial_results = []
    
    for col in numerical_features:
        correlation, p_value = stats.pointbiserialr(diagnosis_num, df[col])
        
        point_biserial_results.append({
            'Feature': col,
            'Point-biserial r': correlation,
            'p-value': p_value,
            'Significant (p<0.05)': p_value < 0.05,
            'Correlation Strength': 'Weak' if abs(correlation) < 0.3 else ('Moderate' if abs(correlation) < 0.7 else 'Strong'),
            'Direction': 'Positive' if correlation > 0 else 'Negative'
        })
    
    point_biserial_df = pd.DataFrame(point_biserial_results)
    point_biserial_df = point_biserial_df.sort_values('Point-biserial r', key=abs, ascending=False)
    
    print("\nCorrelation between Numerical Features and Diagnosis:")
    print(point_biserial_df)
    
    # Save results
    point_biserial_df.to_csv('thyroid_diagnosis_correlations.csv', index=False)
    print("\nDiagnosis correlations saved to 'thyroid_diagnosis_correlations.csv'")
    
    # Create visualization of correlation with diagnosis
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Point-biserial r', y='Feature', data=point_biserial_df, palette='viridis')
    plt.title('Correlation of Features with Malignant Diagnosis')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.savefig('diagnosis_correlation_barplot.png')
    plt.close()
    
    # Create interactive version
    fig = px.bar(point_biserial_df, x='Point-biserial r', y='Feature', 
                title='Correlation of Features with Malignant Diagnosis',
                color='Correlation Strength', template='plotly_white')
    
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.write_html('interactive_diagnosis_correlation.html')
    
    correlation_results = {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'significant': significant_corrs_df if not significant_corrs_df.empty else None,
        'diagnosis_correlation': point_biserial_df
    }
    
    return correlation_results

def risk_factor_analysis(df):
    """
    Analyze categorical risk factors and their relationship with diagnosis
    """
    print("\n=== RISK FACTOR ANALYSIS ===")
    
    # Identify potential risk factors (categorical features)
    risk_factors = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 
                   'Smoking', 'Obesity', 'Diabetes', 'Thyroid_Cancer_Risk']
    
    # Filter only those that exist in the dataset
    risk_factors = [rf for rf in risk_factors if rf in df.columns]
    
    if not risk_factors:
        print("No risk factors found in the dataset")
        return None
    
    # 1. Calculate risk ratios and odds ratios
    risk_stats = []
    
    for factor in risk_factors:
        # Get unique values (typically 'Yes'/'No' or risk levels)
        factor_values = df[factor].unique()
        
        for value in factor_values:
            # Skip if the value is missing
            if pd.isna(value):
                continue
                
            # Create contingency table
            exposed = df[factor] == value
            malignant = df['Diagnosis'] == 'Malignant'
            
            # Counts
            exposed_malignant = (exposed & malignant).sum()
            exposed_benign = exposed.sum() - exposed_malignant
            unexposed_malignant = malignant.sum() - exposed_malignant
            unexposed_benign = (~exposed & ~malignant).sum()
            
            # Avoid division by zero
            if exposed.sum() == 0 or (~exposed).sum() == 0:
                continue
                
            # Calculate risk in exposed and unexposed groups
            risk_exposed = exposed_malignant / exposed.sum()
            risk_unexposed = unexposed_malignant / (~exposed).sum()
            
            # Calculate risk ratio
            risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else np.nan
            
            # Calculate odds ratio
            odds_exposed = exposed_malignant / exposed_benign if exposed_benign > 0 else np.nan
            odds_unexposed = unexposed_malignant / unexposed_benign if unexposed_benign > 0 else np.nan
            odds_ratio = odds_exposed / odds_unexposed if (odds_unexposed > 0 and not np.isnan(odds_exposed)) else np.nan
            
            # Calculate confidence intervals for odds ratio (using log method)
            if not np.isnan(odds_ratio) and odds_ratio > 0:
                log_odds_ratio = np.log(odds_ratio)
                se_log_odds = np.sqrt(1/exposed_malignant + 1/exposed_benign + 1/unexposed_malignant + 1/unexposed_benign)
                
                ci_lower = np.exp(log_odds_ratio - 1.96 * se_log_odds)
                ci_upper = np.exp(log_odds_ratio + 1.96 * se_log_odds)
            else:
                ci_lower = np.nan
                ci_upper = np.nan
            
            # Calculate chi-square and p-value
            contingency = np.array([[exposed_malignant, exposed_benign], 
                                   [unexposed_malignant, unexposed_benign]])
            
            # Check if any expected value is < 5
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            fisher_exact = True if (expected < 5).any() else False
            
            # If any expected value < 5, use Fisher's exact test
            if fisher_exact:
                odds_ratio_fisher, p_fisher = stats.fisher_exact(contingency)
                p = p_fisher
            
            risk_stats.append({
                'Risk Factor': factor,
                'Value': value,
                'Exposed Count': exposed.sum(),
                'Risk in Exposed': risk_exposed,
                'Risk in Unexposed': risk_unexposed,
                'Risk Ratio': risk_ratio,
                'Odds Ratio': odds_ratio,
                'OR 95% CI Lower': ci_lower,
                'OR 95% CI Upper': ci_upper,
                'p-value': p,
                'Significant (p<0.05)': p < 0.05,
                'Test Used': "Fisher's exact test" if fisher_exact else "Chi-square test"
            })
    
    risk_stats_df = pd.DataFrame(risk_stats)
    
    # Sort by significance and effect size
    if not risk_stats_df.empty:
        risk_stats_df = risk_stats_df.sort_values(['Significant (p<0.05)', 'Odds Ratio'], 
                                                 ascending=[False, False])
        
        print("\nRisk Factor Analysis Results:")
        print(risk_stats_df[['Risk Factor', 'Value', 'Risk Ratio', 'Odds Ratio', 
                            'OR 95% CI Lower', 'OR 95% CI Upper', 'p-value', 'Significant (p<0.05)']])
        
        # Save results
        risk_stats_df.to_csv('thyroid_risk_factor_analysis.csv', index=False)
        print("\nRisk factor analysis saved to 'thyroid_risk_factor_analysis.csv'")
        
        # Create forest plot for odds ratios
        plt.figure(figsize=(12, len(risk_stats_df) * 0.5 + 2))
        
        # Plot points and lines for odds ratios and CIs
        plt.errorbar(
            x=risk_stats_df['Odds Ratio'],
            y=risk_stats_df.index,
            xerr=[risk_stats_df['Odds Ratio'] - risk_stats_df['OR 95% CI Lower'], 
                  risk_stats_df['OR 95% CI Upper'] - risk_stats_df['Odds Ratio']],
            fmt='o',
            capsize=5
        )
        
        # Add reference line at OR=1
        plt.axvline(x=1, color='red', linestyle='--')
        
        # Customize plot
        plt.xscale('log')
        plt.xlabel('Odds Ratio (log scale)')
        plt.yticks(risk_stats_df.index, risk_stats_df['Risk Factor'] + ': ' + risk_stats_df['Value'])
        plt.title('Forest Plot of Odds Ratios for Risk Factors')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('risk_factors_forest_plot.png')
        plt.close()
        
        # Create interactive forest plot
        fig = go.Figure()
        
        # Add data points with error bars
        fig.add_trace(go.Scatter(
            x=risk_stats_df['Odds Ratio'],
            y=risk_stats_df['Risk Factor'] + ': ' + risk_stats_df['Value'],
            mode='markers',
            error_x=dict(
                type='data',
                symmetric=False,
                array=risk_stats_df['OR 95% CI Upper'] - risk_stats_df['Odds Ratio'],
                arrayminus=risk_stats_df['Odds Ratio'] - risk_stats_df['OR 95% CI Lower']
            ),
            marker=dict(
                color=np.where(risk_stats_df['Significant (p<0.05)'], 'red', 'blue'),
                size=8
            ),
            name='Odds Ratio'
        ))
        
        # Add reference line
        fig.add_shape(
            type='line',
            x0=1, x1=1,
            y0=-1, y1=len(risk_stats_df),
            line=dict(color='red', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Forest Plot of Odds Ratios for Risk Factors',
            xaxis_title='Odds Ratio (log scale)',
            xaxis_type='log',
            template='plotly_white',
            height=600
        )
        
        fig.write_html('interactive_risk_factors_forest_plot.html')
    else:
        print("No sufficient data for risk factor analysis")
    
    # 2. Create stacked bar charts for each risk factor
    for factor in risk_factors:
        # Calculate proportion of malignant diagnoses by factor level
        cross_tab = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
        
        # Create plot
        plt.figure(figsize=(10, 6))
        cross_tab.plot(kind='bar', stacked=True)
        plt.title(f'Diagnosis by {factor}')
        plt.xlabel(factor)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Diagnosis')
        plt.tight_layout()
        plt.savefig(f'stacked_bar_{factor}.png')
        plt.close()
        
        # Create interactive version
        cross_tab_reset = cross_tab.reset_index()
        cross_tab_melted = pd.melt(cross_tab_reset, id_vars=[factor], var_name='Diagnosis', value_name='Percentage')
        
        fig = px.bar(cross_tab_melted, x=factor, y='Percentage', color='Diagnosis',
                    title=f'Diagnosis by {factor}',
                    barmode='stack', template='plotly_white')
        
        fig.update_layout(
            xaxis_title=factor,
            yaxis_title='Percentage (%)'
        )
        
        fig.write_html(f'interactive_stacked_bar_{factor}.html')
    
    return risk_stats_df if 'risk_stats_df' in locals() else None


def multivariate_analysis(df):
    """
    Perform multivariate analysis using ANOVA and logistic regression
    """
    print("\n=== MULTIVARIATE ANALYSIS ===")
    
    # 1. ANOVA for interactions between categorical and numerical variables
    # Get numerical outcome variables
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Patient_ID' in numerical_features:
        numerical_features.remove('Patient_ID')
    
    # Get categorical variables
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Diagnosis' in categorical_features:
        categorical_features.remove('Diagnosis')
    
    # Limit to top 3 numerical features for simplicity
    print("\nPerforming ANOVA for interactions between categorical and numerical variables")
    anova_results = []
    
    for num_var in numerical_features[:3]:  # Limit to first 3 numerical variables
        for cat_var in categorical_features[:3]:  # Limit to first 3 categorical variables
            # Create and fit the model
            formula = f"{num_var} ~ C({cat_var}) + C(Diagnosis) + C({cat_var}):C(Diagnosis)"
            try:
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # Extract results
                main_effect_cat = anova_table.loc[f'C({cat_var})', 'PR(>F)']
                main_effect_diagnosis = anova_table.loc['C(Diagnosis)', 'PR(>F)']
                
                # Check if interaction term exists
                interaction_term = f'C({cat_var}):C(Diagnosis)'
                if interaction_term in anova_table.index:
                    interaction_effect = anova_table.loc[interaction_term, 'PR(>F)']
                else:
                    interaction_effect = np.nan
                
                anova_results.append({
                    'Numerical Variable': num_var,
                    'Categorical Variable': cat_var,
                    'Main Effect Cat. p-value': main_effect_cat,
                    'Main Effect Diagnosis p-value': main_effect_diagnosis,
                    'Interaction p-value': interaction_effect,
                    'Cat. Effect Significant': main_effect_cat < 0.05,
                    'Diagnosis Effect Significant': main_effect_diagnosis < 0.05,
                    'Interaction Significant': interaction_effect < 0.05 if not np.isnan(interaction_effect) else False
                })
            except Exception as e:
                print(f"Error in ANOVA for {num_var} ~ {cat_var}: {str(e)}")
    
    if anova_results:
        anova_df = pd.DataFrame(anova_results)
        print("\nANOVA Results:")
        print(anova_df)
        
        # Save results
        anova_df.to_csv('thyroid_anova_results.csv', index=False)
        print("\nANOVA results saved to 'thyroid_anova_results.csv'")
        
        # Visualize significant interactions
        significant_interactions = anova_df[anova_df['Interaction Significant']]
        
        for _, row in significant_interactions.iterrows():
            num_var = row['Numerical Variable']
            cat_var = row['Categorical Variable']
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(x=cat_var, y=num_var, hue='Diagnosis', data=df)
            plt.title(f'Interaction between {cat_var} and Diagnosis on {num_var}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'interaction_{cat_var}_diagnosis_{num_var}.png')
            plt.close()
            
            # Create interaction plot
            plt.figure(figsize=(12, 8))
            
            # Calculate means for interaction plot
            means = df.groupby([cat_var, 'Diagnosis'])[num_var].mean().unstack()
            
            for diagnosis in means.columns:
                plt.plot(means.index, means[diagnosis], marker='o', label=diagnosis)
            
            plt.title(f'Interaction between {cat_var} and Diagnosis on {num_var}')
            plt.xlabel(cat_var)
            plt.ylabel(f'Mean {num_var}')
            plt.legend(title='Diagnosis')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'interaction_plot_{cat_var}_diagnosis_{num_var}.png')
            plt.close()
            
            # Create interactive version
            fig = px.box(df, x=cat_var, y=num_var, color='Diagnosis',
                        title=f'Interaction between {cat_var} and Diagnosis on {num_var}',
                        template='plotly_white')
            
            fig.write_html(f'interactive_interaction_{cat_var}_diagnosis_{num_var}.html')
    else:
        print("No ANOVA results available")
    
    # 2. Logistic Regression for predicting diagnosis
    print("\nFitting Logistic Regression model for predicting Diagnosis")
    
    try:
        # Prepare data for logistic regression
        # Encode diagnosis as binary
        y = (df['Diagnosis'] == 'Malignant').astype(int)
        
        # Select predictors - use only numerical features to avoid issues with categorical values
        X = df[numerical_features].copy()
        
        # Handle missing values - only fill numerical values with median
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        # Add categorical features after proper encoding
        for cat_col in categorical_features:
            # Skip columns with too many categories
            if df[cat_col].nunique() < 10:
                # Create dummy variables
                dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True, dummy_na=True)
                # Add to X
                X = pd.concat([X, dummies], axis=1)
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Fit model
        logit_model = sm.Logit(y, X_with_const)
        logit_result = logit_model.fit(disp=0)  # disp=0 suppresses convergence messages
        
        print("\nLogistic Regression Results:")
        print(logit_result.summary())
        
        # Extract odds ratios and confidence intervals
        odds_ratios = np.exp(logit_result.params)
        conf_intervals = np.exp(logit_result.conf_int())
        conf_intervals.columns = ['Lower 95% CI', 'Upper 95% CI']
        
        # Combine parameters, pvalues, odds ratios, and confidence intervals
        logit_summary = pd.DataFrame({
            'Coefficient': logit_result.params,
            'Std Error': logit_result.bse,
            'z-value': logit_result.tvalues,
            'p-value': logit_result.pvalues,
            'Odds Ratio': odds_ratios,
            'Lower 95% CI': conf_intervals['Lower 95% CI'],
            'Upper 95% CI': conf_intervals['Upper 95% CI'],
            'Significant (p<0.05)': logit_result.pvalues < 0.05
        })
        
        # Sort by significance and effect size
        logit_summary = logit_summary.sort_values(['Significant (p<0.05)', 'Odds Ratio'], ascending=[False, False])
        
        # Save results
        logit_summary.to_csv('thyroid_logistic_regression.csv')
        print("\nLogistic regression results saved to 'thyroid_logistic_regression.csv'")
        
        # Create forest plot for odds ratios
        # Filter out the constant term and sort by odds ratio
        plot_data = logit_summary[logit_summary.index != 'const'].sort_values('Odds Ratio', ascending=False)
        
        plt.figure(figsize=(12, len(plot_data) * 0.4 + 2))
        
        # Plot points and lines for odds ratios and CIs
        plt.errorbar(
            x=plot_data['Odds Ratio'],
            y=range(len(plot_data)),
            xerr=[plot_data['Odds Ratio'] - plot_data['Lower 95% CI'], 
                  plot_data['Upper 95% CI'] - plot_data['Odds Ratio']],
            fmt='o',
            capsize=5
        )
        
        # Add reference line at OR=1
        plt.axvline(x=1, color='red', linestyle='--')
        
        # Customize plot
        plt.xscale('log')
        plt.xlabel('Odds Ratio (log scale)')
        plt.yticks(range(len(plot_data)), plot_data.index)
        plt.title('Forest Plot of Odds Ratios from Logistic Regression')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logistic_regression_forest_plot.png')
        plt.close()
        
        # Create interactive forest plot
        fig = go.Figure()
        
        # Add data points with error bars
        fig.add_trace(go.Scatter(
            x=plot_data['Odds Ratio'],
            y=plot_data.index,
            mode='markers',
            error_x=dict(
                type='data',
                symmetric=False,
                array=plot_data['Upper 95% CI'] - plot_data['Odds Ratio'],
                arrayminus=plot_data['Odds Ratio'] - plot_data['Lower 95% CI']
            ),
            marker=dict(
                color=np.where(plot_data['Significant (p<0.05)'], 'red', 'blue'),
                size=8
            ),
            name='Odds Ratio'
        ))
        
        # Add reference line
        fig.add_shape(
            type='line',
            x0=1, x1=1,
            y0=-1, y1=len(plot_data),
            line=dict(color='red', dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Forest Plot of Odds Ratios from Logistic Regression',
            xaxis_title='Odds Ratio (log scale)',
            xaxis_type='log',
            template='plotly_white',
            height=max(600, len(plot_data) * 25)
        )
        
        fig.write_html('interactive_logistic_regression_forest_plot.html')
        
        return {
            'anova': anova_df if 'anova_df' in locals() else None,
            'logistic_regression': logit_summary
        }
    
    except Exception as e:
        print(f"Error in logistic regression: {str(e)}")
        return {
            'anova': anova_df if 'anova_df' in locals() else None,
            'logistic_regression': None
        }
    


    
def create_executive_summary(stats_results, group_comparison_results, correlation_results, risk_factor_results, multivariate_results):
    """
    Create an executive summary of key findings
    """
    print("\n=== CREATING EXECUTIVE SUMMARY ===")
    
    # Start with basic HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thyroid Cancer Statistical Analysis - Executive Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
            h3 { color: #3498db; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .highlight { background-color: #ebf5fb; padding: 15px; border-radius: 5px; }
            .significant { color: #e74c3c; font-weight: bold; }
            .container { display: flex; justify-content: space-between; }
            .half-width { width: 48%; }
        </style>
    </head>
    <body>
        <h1>Thyroid Cancer Statistical Analysis - Executive Summary</h1>
    """
    
    # 1. Key Demographics and Basic Statistics
    html_content += """
        <h2>Key Demographics and Basic Statistics</h2>
        <div class="highlight">
    """
    
    # Add diagnosis distribution if available
    if 'diagnosis_correlation' in correlation_results:
        diagnosis_corr = correlation_results['diagnosis_correlation']
        top_correlates = diagnosis_corr.sort_values('Point-biserial r', key=abs, ascending=False).head(3)
        
        html_content += f"""
            <h3>Top Correlates with Malignant Diagnosis</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Correlation</th>
                    <th>p-value</th>
                    <th>Significance</th>
                </tr>
        """
        
        for _, row in top_correlates.iterrows():
            sig_class = "significant" if row['Significant (p<0.05)'] else ""
            html_content += f"""
                <tr class="{sig_class}">
                    <td>{row['Feature']}</td>
                    <td>{row['Point-biserial r']:.4f}</td>
                    <td>{row['p-value']:.4f}</td>
                    <td>{row['Correlation Strength']} ({row['Direction']})</td>
                </tr>
            """
        
        html_content += "</table>"
    
    html_content += """
        </div>
    """
    
    # 2. Significant Differences Between Benign and Malignant Cases
    html_content += """
        <h2>Significant Differences Between Benign and Malignant Cases</h2>
        <div class="highlight">
    """
    
    if group_comparison_results and 'numerical_tests' in group_comparison_results:
        sig_diff = group_comparison_results['numerical_tests'][group_comparison_results['numerical_tests']['Significant (p<0.05)']].sort_values('Effect Size (Cohen\'s d)', ascending=False)
        
        if not sig_diff.empty:
            html_content += f"""
                <h3>Top Numerical Features Differentiating Benign and Malignant Cases</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Test</th>
                        <th>p-value</th>
                        <th>Effect Size</th>
                        <th>Effect Magnitude</th>
                    </tr>
            """
            
            for _, row in sig_diff.head(5).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['Feature']}</td>
                        <td>{row['Test']}</td>
                        <td>{row['p-value']:.6f}</td>
                        <td>{row['Effect Size (Cohen\'s d)']:.4f}</td>
                        <td>{row['Effect Magnitude']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        else:
            html_content += "<p>No significant differences found in numerical features.</p>"
    
    # Add categorical features if available
    if group_comparison_results and 'categorical_tests' in group_comparison_results:
        cat_assoc = group_comparison_results['categorical_tests'][group_comparison_results['categorical_tests']['Significant (p<0.05)']].sort_values('Cramer\'s V', ascending=False)
        
        if not cat_assoc.empty:
            html_content += f"""
                <h3>Top Categorical Features Associated with Diagnosis</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Chi-square</th>
                        <th>p-value</th>
                        <th>Cramer's V</th>
                        <th>Association Strength</th>
                    </tr>
            """
            
            for _, row in cat_assoc.head(5).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['Feature']}</td>
                        <td>{row['Chi-square']:.4f}</td>
                        <td>{row['p-value']:.6f}</td>
                        <td>{row['Cramer\'s V']:.4f}</td>
                        <td>{row['Association Strength']}</td>
                    </tr>
                """
            
            html_content += "</table>"
    
    html_content += """
        </div>
    """
    
    # 3. Key Risk Factors
    html_content += """
        <h2>Key Risk Factors</h2>
        <div class="highlight">
    """
    
    if risk_factor_results is not None and not risk_factor_results.empty:
        sig_risk_factors = risk_factor_results[risk_factor_results['Significant (p<0.05)']].sort_values('Odds Ratio', ascending=False)
        
        if not sig_risk_factors.empty:
            html_content += f"""
                <h3>Significant Risk Factors for Malignant Diagnosis</h3>
                <table>
                    <tr>
                        <th>Risk Factor</th>
                        <th>Value</th>
                        <th>Odds Ratio</th>
                        <th>95% CI</th>
                        <th>p-value</th>
                    </tr>
            """
            
            for _, row in sig_risk_factors.head(5).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['Risk Factor']}</td>
                        <td>{row['Value']}</td>
                        <td>{row['Odds Ratio']:.4f}</td>
                        <td>({row['OR 95% CI Lower']:.4f} - {row['OR 95% CI Upper']:.4f})</td>
                        <td>{row['p-value']:.6f}</td>
                    </tr>
                """
            
            html_content += "</table>"
        else:
            html_content += "<p>No significant risk factors identified.</p>"
    else:
        html_content += "<p>Risk factor analysis not available.</p>"
    
    html_content += """
        </div>
    """
    
    # 4. Key Correlations
    html_content += """
        <h2>Key Correlations</h2>
        <div class="highlight">
    """
    
    if correlation_results and 'significant' in correlation_results and correlation_results['significant'] is not None:
        sig_corrs = correlation_results['significant'].sort_values('Pearson r', key=abs, ascending=False)
        
        if not sig_corrs.empty:
            html_content += f"""
                <h3>Top Significant Correlations Between Features</h3>
                <table>
                    <tr>
                        <th>Feature 1</th>
                        <th>Feature 2</th>
                        <th>Pearson r</th>
                        <th>p-value</th>
                        <th>Strength</th>
                        <th>Direction</th>
                    </tr>
            """
            
            for _, row in sig_corrs.head(5).iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['Feature 1']}</td>
                        <td>{row['Feature 2']}</td>
                        <td>{row['Pearson r']:.4f}</td>
                        <td>{row['p-value']:.6f}</td>
                        <td>{row['Correlation Strength']}</td>
                        <td>{row['Direction']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        else:
            html_content += "<p>No significant correlations found between features.</p>"
    else:
        html_content += "<p>Correlation analysis not available.</p>"
    
    html_content += """
        </div>
    """
    
    # 5. Multivariate Analysis Results
    html_content += """
        <h2>Multivariate Analysis Results</h2>
        <div class="highlight">
    """
    
    if multivariate_results and 'logistic_regression' in multivariate_results and multivariate_results['logistic_regression'] is not None:
        logit_summary = multivariate_results['logistic_regression']
        sig_predictors = logit_summary[logit_summary['Significant (p<0.05)']].sort_values('Odds Ratio', ascending=False)
        
        if not sig_predictors.empty and len(sig_predictors) > 1:  # More than 1 to exclude just the constant
            html_content += f"""
                <h3>Significant Predictors of Malignant Diagnosis (Logistic Regression)</h3>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Coefficient</th>
                        <th>Odds Ratio</th>
                        <th>95% CI</th>
                        <th>p-value</th>
                    </tr>
            """
            
            for idx, row in sig_predictors.iterrows():
                if idx != 'const':  # Skip the constant term
                    html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{row['Coefficient']:.4f}</td>
                            <td>{row['Odds Ratio']:.4f}</td>
                            <td>({row['Lower 95% CI']:.4f} - {row['Upper 95% CI']:.4f})</td>
                            <td>{row['p-value']:.6f}</td>
                        </tr>
                    """
            
            html_content += "</table>"
        else:
            html_content += "<p>No significant predictors found in the logistic regression model (excluding constant).</p>"
    else:
        html_content += "<p>Logistic regression analysis not available.</p>"
    
    html_content += """
        </div>
    """
    
    # 6. Recommendations
    html_content += """
        <h2>Key Findings and Recommendations</h2>
        <div class="highlight">
            <h3>Key Findings</h3>
            <ul>
    """
    
    # Automatically generate key findings based on results
    key_findings = []
    
    # Finding from numerical features
    if group_comparison_results and 'numerical_tests' in group_comparison_results:
        sig_diff = group_comparison_results['numerical_tests'][group_comparison_results['numerical_tests']['Significant (p<0.05)']].sort_values('Effect Size (Cohen\'s d)', ascending=False)
        
        if not sig_diff.empty:
            top_feature = sig_diff.iloc[0]
            key_findings.append(f"<li>The most significant differentiating feature between benign and malignant cases is <strong>{top_feature['Feature']}</strong> with a {top_feature['Effect Magnitude']} effect size ({top_feature['Effect Size (Cohen\'s d)']:.2f}).</li>")
    
    # Finding from risk factors
    if risk_factor_results is not None and not risk_factor_results.empty:
        sig_risk_factors = risk_factor_results[risk_factor_results['Significant (p<0.05)']].sort_values('Odds Ratio', ascending=False)
        
        if not sig_risk_factors.empty:
            top_risk = sig_risk_factors.iloc[0]
            key_findings.append(f"<li>Patients with <strong>{top_risk['Risk Factor']}</strong> value of <strong>{top_risk['Value']}</strong> have {top_risk['Odds Ratio']:.2f} times higher odds of malignant diagnosis (95% CI: {top_risk['OR 95% CI Lower']:.2f}-{top_risk['OR 95% CI Upper']:.2f}).</li>")
    
    # Finding from correlations
    if correlation_results and 'diagnosis_correlation' in correlation_results:
        diagnosis_corr = correlation_results['diagnosis_correlation']
        top_correlate = diagnosis_corr.iloc[0]
        
        if top_correlate['Significant (p<0.05)']:
            key_findings.append(f"<li>The feature most strongly correlated with malignancy is <strong>{top_correlate['Feature']}</strong> with a {top_correlate['Direction'].lower()} correlation of {top_correlate['Point-biserial r']:.2f}.</li>")
    
    # Finding from logistic regression
    if multivariate_results and 'logistic_regression' in multivariate_results and multivariate_results['logistic_regression'] is not None:
        logit_summary = multivariate_results['logistic_regression']
        sig_predictors = logit_summary[logit_summary['Significant (p<0.05)']].sort_values('Odds Ratio', ascending=False)
        
        if not sig_predictors.empty and len(sig_predictors) > 1:  # More than 1 to exclude just the constant
            for idx, row in sig_predictors.iterrows():
                if idx != 'const':  # Skip the constant term
                    key_findings.append(f"<li>In the multivariate model, <strong>{idx}</strong> remained a significant predictor with adjusted odds ratio of {row['Odds Ratio']:.2f} (p={row['p-value']:.4f}).</li>")
                    break  # Just add the top predictor
    
    # Add all findings to HTML
    for finding in key_findings:
        html_content += finding
    
    # If no findings, add placeholder
    if not key_findings:
        html_content += "<li>No statistically significant findings were identified in the analysis.</li>"
    
    html_content += """
            </ul>
            
            <h3>Recommendations</h3>
            <ul>
                <li>Focus clinical assessment on the key differentiating features identified in this analysis.</li>
                <li>Consider the identified risk factors when stratifying patients for further testing or monitoring.</li>
                <li>Validate these findings with external datasets before implementing in clinical practice.</li>
                <li>Conduct further studies to investigate causal relationships between the identified associations.</li>
            </ul>
        </div>
    """
    
    # Close HTML document
    html_content += """
    </body>
    </html>
    """
    
    # Save the executive summary
    with open('thyroid_executive_summary.html', 'w') as f:
        f.write(html_content)
    
    print("Executive summary saved to 'thyroid_executive_summary.html'")




def main():
    """
    Main function to run the statistical analysis
    """
    # Set file path
    file_path = 'thyroid_cancer_risk_data.csv'
    
    # Load the data
    df = load_data(file_path)
    
    # 1. Basic statistics
    stats_results = basic_statistics(df)
    
    # 2. Normality tests
    normality_results = normality_tests(df)
    
    # 3. Group comparisons between benign and malignant
    group_comparison_results = group_comparisons(df)
    
    # 4. Correlation analysis
    correlation_results = correlation_analysis(df)
    
    # 5. Risk factor analysis
    risk_factor_results = risk_factor_analysis(df)
    
    # 6. Multivariate analysis
    multivariate_results = multivariate_analysis(df)
    
    # 7. Create executive summary
    create_executive_summary(stats_results, group_comparison_results, correlation_results, 
                           risk_factor_results, multivariate_results)
    
    print("\n=== STATISTICAL ANALYSIS COMPLETE ===")
    print("All results have been saved as CSV files and visualizations.")
    print("Open the 'thyroid_executive_summary.html' file for a comprehensive summary of findings.")
    
    return {
        'basic_stats': stats_results,
        'normality': normality_results,
        'group_comparisons': group_comparison_results,
        'correlations': correlation_results,
        'risk_factors': risk_factor_results,
        'multivariate': multivariate_results
    }

if __name__ == "__main__":
    results = main()