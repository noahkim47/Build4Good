import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set styles for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_data(file_path):
    """Load the thyroid cancer dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def basic_description(df):
    """Basic description of the dataset"""
    print("\n=== BASIC DATASET DESCRIPTION ===")
    
    # Count by diagnosis
    diagnosis_counts = df['Diagnosis'].value_counts()
    print("\nDiagnosis Counts:")
    print(diagnosis_counts)
    
    # Percentage by diagnosis
    diagnosis_percent = df['Diagnosis'].value_counts(normalize=True) * 100
    print("\nDiagnosis Percentages:")
    print(diagnosis_percent)
    
    # Basic summary of numerical features
    numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    print("\nNumerical Features Summary:")
    print(df[numerical_cols].describe())
    
    # Interactive bar chart
    fig = px.bar(x=diagnosis_counts.index, y=diagnosis_counts.values, 
                color=diagnosis_counts.index,
                title="Diagnosis Distribution",
                labels={"x": "Diagnosis", "y": "Count"})
    fig.update_layout(template="plotly_white")
    fig.write_html('interactive_diagnosis_distribution.html')
    print("Interactive diagnosis distribution saved as 'interactive_diagnosis_distribution.html'")
    
    # Static plot for HTML report
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Diagnosis', data=df)
    plt.title('Diagnosis Distribution')
    plt.savefig('diagnosis_distribution.png')
    plt.close()
    
    return diagnosis_counts

def gender_vs_diagnosis(df):
    """Analyze relationship between gender and diagnosis"""
    print("\n=== GENDER VS DIAGNOSIS ===")
    
    # Create contingency table
    gender_diag_table = pd.crosstab(df['Gender'], df['Diagnosis'])
    print("\nGender vs Diagnosis Contingency Table:")
    print(gender_diag_table)
    
    # Convert to percentages (rows sum to 100%)
    gender_diag_percent = pd.crosstab(df['Gender'], df['Diagnosis'], normalize='index') * 100
    print("\nGender vs Diagnosis Percentages (by row):")
    print(gender_diag_percent)
    
    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(gender_diag_table)
    print(f"\nChi-square test: chi2={chi2:.4f}, p={p:.6f}")
    print(f"Significant difference: {'Yes' if p < 0.05 else 'No'}")
    
    # Interactive plot
    # Convert crosstab to long format for plotly
    gender_diag_percent_reset = gender_diag_percent.reset_index()
    gender_diag_percent_melted = pd.melt(gender_diag_percent_reset, 
                                        id_vars=['Gender'], 
                                        value_vars=['Benign', 'Malignant'],
                                        var_name='Diagnosis', 
                                        value_name='Percentage')
    
    fig = px.bar(gender_diag_percent_melted, x='Gender', y='Percentage', color='Diagnosis',
               title='Diagnosis by Gender (%)',
               barmode='group',
               labels={'Percentage': 'Percentage (%)', 'Gender': 'Gender'},
               template='plotly_white')
    
    # Add significance annotation
    if p < 0.05:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"Chi-square test: p={p:.6f} (Significant)",
            showarrow=False,
            font=dict(color="red", size=12),
            xref="paper", yref="paper"
        )
    else:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"Chi-square test: p={p:.6f} (Not Significant)",
            showarrow=False,
            font=dict(color="black", size=12),
            xref="paper", yref="paper"
        )
    
    fig.write_html('interactive_gender_vs_diagnosis.html')
    print("Interactive gender vs diagnosis chart saved as 'interactive_gender_vs_diagnosis.html'")
    
    # Static plot for HTML report
    plt.figure(figsize=(10, 6))
    gender_diag_percent.plot(kind='bar')
    plt.title('Diagnosis by Gender')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.savefig('gender_vs_diagnosis.png')
    plt.close()
    
    # Print clear summary
    print("\nCLEAR SUMMARY:")
    for gender in gender_diag_percent.index:
        malig_percent = gender_diag_percent.loc[gender, 'Malignant']
        print(f"- {gender}: {malig_percent:.1f}% Malignant")
    
    if p < 0.05:
        print(f"  → Significant difference in malignancy rates by gender (p={p:.6f})")
    else:
        print(f"  → No significant difference in malignancy rates by gender (p={p:.6f})")
    
    return {'chi2': chi2, 'p_value': p, 'table': gender_diag_table}

def country_vs_diagnosis(df):
    """Analyze relationship between country and diagnosis"""
    print("\n=== COUNTRY VS DIAGNOSIS ===")
    
    # Get top 10 countries by frequency
    top_countries = df['Country'].value_counts().nlargest(10).index.tolist()
    df_top_countries = df[df['Country'].isin(top_countries)].copy()
    
    # Create contingency table
    country_diag_table = pd.crosstab(df_top_countries['Country'], df_top_countries['Diagnosis'])
    print("\nCountry vs Diagnosis Contingency Table (Top 10 Countries):")
    print(country_diag_table)
    
    # Convert to percentages (rows sum to 100%)
    country_diag_percent = pd.crosstab(df_top_countries['Country'], df_top_countries['Diagnosis'], normalize='index') * 100
    print("\nCountry vs Diagnosis Percentages (by row):")
    print(country_diag_percent)
    
    # Sort by malignancy percentage
    country_diag_percent = country_diag_percent.sort_values('Malignant', ascending=False)
    
    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(country_diag_table)
    print(f"\nChi-square test: chi2={chi2:.4f}, p={p:.6f}")
    print(f"Significant difference: {'Yes' if p < 0.05 else 'No'}")
    
    # Interactive malignancy rate plot
    country_diag_sorted = country_diag_percent.sort_values('Malignant', ascending=False)
    fig = px.bar(x=country_diag_sorted.index, y=country_diag_sorted['Malignant'],
               title='Malignancy Rate by Country',
               color=country_diag_sorted['Malignant'],
               color_continuous_scale='Viridis',
               labels={'x': 'Country', 'y': 'Malignancy Rate (%)', 'color': 'Rate (%)'},
               template='plotly_white')
    
    # Add overall average line
    fig.add_hline(y=df['Diagnosis'].value_counts(normalize=True)['Malignant']*100,
                line_dash="dash", line_color="red",
                annotation_text="Overall Average",
                annotation_position="bottom right")
    
    # Add significance annotation
    if p < 0.05:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"Chi-square test: p={p:.6f} (Significant)",
            showarrow=False,
            font=dict(color="red", size=12),
            xref="paper", yref="paper"
        )
    else:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"Chi-square test: p={p:.6f} (Not Significant)",
            showarrow=False,
            font=dict(color="black", size=12),
            xref="paper", yref="paper"
        )
    
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html('interactive_country_vs_diagnosis.html')
    print("Interactive country vs diagnosis chart saved as 'interactive_country_vs_diagnosis.html'")
    
    # Static plot for HTML report
    plt.figure(figsize=(12, 8))
    country_diag_percent['Malignant'].plot(kind='bar')
    plt.title('Malignancy Rate by Country')
    plt.ylabel('Percentage Malignant')
    plt.xticks(rotation=45)
    plt.axhline(y=df['Diagnosis'].value_counts(normalize=True)['Malignant']*100, 
                color='r', linestyle='--', label='Overall Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig('country_vs_diagnosis.png')
    plt.close()
    
    # Print clear summary
    print("\nCLEAR SUMMARY (Countries by Malignancy Rate):")
    for country in country_diag_percent.index:
        malig_percent = country_diag_percent.loc[country, 'Malignant']
        print(f"- {country}: {malig_percent:.1f}% Malignant")
    
    if p < 0.05:
        print(f"  → Significant difference in malignancy rates by country (p={p:.6f})")
    else:
        print(f"  → No significant difference in malignancy rates by country (p={p:.6f})")
    
    return {'chi2': chi2, 'p_value': p, 'table': country_diag_table}

def age_t_test(df):
    """Compare age between benign and malignant cases"""
    print("\n=== AGE COMPARISON ===")
    
    # Extract age by diagnosis
    benign_age = df[df['Diagnosis'] == 'Benign']['Age']
    malignant_age = df[df['Diagnosis'] == 'Malignant']['Age']
    
    # Calculate means
    benign_mean = benign_age.mean()
    malignant_mean = malignant_age.mean()
    difference = malignant_mean - benign_mean
    
    print(f"Mean age (Benign): {benign_mean:.2f} years")
    print(f"Mean age (Malignant): {malignant_mean:.2f} years")
    print(f"Difference: {difference:.2f} years")
    
    # Two-sample t-test
    t_stat, p_val = stats.ttest_ind(benign_age, malignant_age, equal_var=False)
    print(f"\nTwo-sample t-test (Welch's): t={t_stat:.4f}, p={p_val:.6f}")
    print(f"Significant difference: {'Yes' if p_val < 0.05 else 'No'}")
    
    # Interactive boxplot
    fig = px.box(df, x='Diagnosis', y='Age', color='Diagnosis',
               title='Age Distribution by Diagnosis',
               points='all', # show all points
               labels={'Age': 'Age (years)'},
               template='plotly_white')
    
    # Add mean points
    fig.add_trace(go.Scatter(
        x=['Benign', 'Malignant'],
        y=[benign_mean, malignant_mean],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='Mean'
    ))
    
    # Add significance annotation
    if p_val < 0.05:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"T-test: p={p_val:.6f} (Significant), difference = {difference:.2f} years",
            showarrow=False,
            font=dict(color="red", size=12),
            xref="paper", yref="paper"
        )
    else:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"T-test: p={p_val:.6f} (Not Significant)",
            showarrow=False,
            font=dict(color="black", size=12),
            xref="paper", yref="paper"
        )
    
    fig.write_html('interactive_age_by_diagnosis.html')
    print("Interactive age distribution chart saved as 'interactive_age_by_diagnosis.html'")
    
    # Static plot for HTML report
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Diagnosis', y='Age', data=df)
    plt.title('Age Distribution by Diagnosis')
    plt.savefig('age_by_diagnosis.png')
    plt.close()
    
    # Print clear summary
    print("\nCLEAR SUMMARY:")
    print(f"- Benign patients average age: {benign_mean:.1f} years")
    print(f"- Malignant patients average age: {malignant_mean:.1f} years")
    
    if p_val < 0.05:
        print(f"  → Age is significantly {'higher' if difference > 0 else 'lower'} in malignant cases (p={p_val:.6f})")
    else:
        print(f"  → No significant difference in age between benign and malignant cases (p={p_val:.6f})")
    
    return {'t_stat': t_stat, 'p_value': p_val, 'benign_mean': benign_mean, 'malignant_mean': malignant_mean}

def nodule_size_t_test(df):
    """Compare nodule size between benign and malignant cases"""
    print("\n=== NODULE SIZE COMPARISON ===")
    
    # Extract nodule size by diagnosis
    benign_size = df[df['Diagnosis'] == 'Benign']['Nodule_Size']
    malignant_size = df[df['Diagnosis'] == 'Malignant']['Nodule_Size']
    
    # Calculate means
    benign_mean = benign_size.mean()
    malignant_mean = malignant_size.mean()
    difference = malignant_mean - benign_mean
    difference_percent = (difference / benign_mean) * 100
    
    print(f"Mean nodule size (Benign): {benign_mean:.2f} cm")
    print(f"Mean nodule size (Malignant): {malignant_mean:.2f} cm")
    print(f"Difference: {difference:.2f} cm ({difference_percent:.1f}%)")
    
    # Two-sample t-test
    t_stat, p_val = stats.ttest_ind(benign_size, malignant_size, equal_var=False)
    print(f"\nTwo-sample t-test (Welch's): t={t_stat:.4f}, p={p_val:.6f}")
    print(f"Significant difference: {'Yes' if p_val < 0.05 else 'No'}")
    
    # Interactive violin plot
    fig = px.violin(df, x='Diagnosis', y='Nodule_Size', color='Diagnosis',
                  title='Nodule Size Distribution by Diagnosis',
                  points='all', # show all points
                  box=True, # show box plot inside
                  labels={'Nodule_Size': 'Nodule Size (cm)'},
                  template='plotly_white')
    
    # Add mean points
    fig.add_trace(go.Scatter(
        x=['Benign', 'Malignant'],
        y=[benign_mean, malignant_mean],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='Mean'
    ))
    
    # Add significance annotation
    if p_val < 0.05:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"T-test: p={p_val:.6f} (Significant), difference = {difference:.2f} cm ({difference_percent:.1f}%)",
            showarrow=False,
            font=dict(color="red", size=12),
            xref="paper", yref="paper"
        )
    else:
        fig.add_annotation(
            x=0.5, y=1.05,
            text=f"T-test: p={p_val:.6f} (Not Significant)",
            showarrow=False,
            font=dict(color="black", size=12),
            xref="paper", yref="paper"
        )
    
    fig.write_html('interactive_nodule_size_by_diagnosis.html')
    print("Interactive nodule size distribution chart saved as 'interactive_nodule_size_by_diagnosis.html'")
    
    # Static plot for HTML report
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Diagnosis', y='Nodule_Size', data=df)
    plt.title('Nodule Size Distribution by Diagnosis')
    plt.savefig('nodule_size_by_diagnosis.png')
    plt.close()
    
    # Print clear summary
    print("\nCLEAR SUMMARY:")
    print(f"- Benign nodules average size: {benign_mean:.2f} cm")
    print(f"- Malignant nodules average size: {malignant_mean:.2f} cm")
    print(f"- Malignant nodules are {abs(difference_percent):.1f}% {'larger' if difference > 0 else 'smaller'}")
    
    if p_val < 0.05:
        print(f"  → Nodule size is significantly {'larger' if difference > 0 else 'smaller'} in malignant cases (p={p_val:.6f})")
    else:
        print(f"  → No significant difference in nodule size between benign and malignant cases (p={p_val:.6f})")
    
    return {'t_stat': t_stat, 'p_value': p_val, 'benign_mean': benign_mean, 'malignant_mean': malignant_mean}

def risk_factors_analysis(df):
    """Analyze categorical risk factors"""
    print("\n=== RISK FACTORS ANALYSIS ===")
    
    # List of risk factors to analyze
    risk_factors = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 
                    'Smoking', 'Obesity', 'Diabetes', 'Thyroid_Cancer_Risk']
    
    # Check which factors exist in the dataset
    existing_factors = [factor for factor in risk_factors if factor in df.columns]
    
    if not existing_factors:
        print("No risk factors found in the dataset")
        return None
        
    results = {}
    
    # For each risk factor
    for factor in existing_factors:
        print(f"\n--- {factor} ---")
        
        # Create contingency table
        contingency = pd.crosstab(df[factor], df['Diagnosis'])
        print(f"\n{factor} vs Diagnosis:")
        print(contingency)
        
        # Calculate percentages
        percent_table = pd.crosstab(df[factor], df['Diagnosis'], normalize='index') * 100
        print(f"\n{factor} vs Diagnosis (%):")
        print(percent_table)
        
        # Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square test: chi2={chi2:.4f}, p={p:.6f}")
        print(f"Significant association: {'Yes' if p < 0.05 else 'No'}")
        
        # Interactive bar chart
        # Convert to long format for plotly
        percent_reset = percent_table.reset_index()
        percent_melted = pd.melt(percent_reset, 
                                id_vars=[factor], 
                                value_vars=['Benign', 'Malignant'],
                                var_name='Diagnosis', 
                                value_name='Percentage')
        
        # Calculate for risk ratio
        risk_ratio = None
        if len(contingency.index) == 2 and 'Yes' in contingency.index and 'No' in contingency.index:
            yes_malignant_rate = percent_table.loc['Yes', 'Malignant']
            no_malignant_rate = percent_table.loc['No', 'Malignant']
            risk_ratio = yes_malignant_rate / no_malignant_rate
            
            print(f"\nRisk analysis:")
            print(f"- Malignancy rate with {factor}=Yes: {yes_malignant_rate:.1f}%")
            print(f"- Malignancy rate with {factor}=No: {no_malignant_rate:.1f}%")
            print(f"- Risk Ratio: {risk_ratio:.2f}")
            
            if risk_ratio > 1:
                print(f"  → People with {factor} are {risk_ratio:.2f}x more likely to have malignant diagnosis")
            else:
                print(f"  → People with {factor} are {1/risk_ratio:.2f}x less likely to have malignant diagnosis")
        
        # Create interactive plot
        fig = px.bar(percent_melted, x=factor, y='Percentage', color='Diagnosis',
                   title=f'Diagnosis by {factor} (%)',
                   barmode='group',
                   labels={'Percentage': 'Percentage (%)'},
                   template='plotly_white')
        
        # Add risk ratio annotation if binary factor
        if risk_ratio is not None:
            if risk_ratio > 1:
                fig.add_annotation(
                    x=0.5, y=1.1,
                    text=f"Risk Ratio: {risk_ratio:.2f}x increased risk with {factor}",
                    showarrow=False,
                    font=dict(color="red" if p < 0.05 else "black", size=12),
                    xref="paper", yref="paper"
                )
            else:
                fig.add_annotation(
                    x=0.5, y=1.1,
                    text=f"Risk Ratio: {1/risk_ratio:.2f}x decreased risk with {factor}",
                    showarrow=False,
                    font=dict(color="red" if p < 0.05 else "black", size=12),
                    xref="paper", yref="paper"
                )
        
        # Add significance annotation
        if p < 0.05:
            fig.add_annotation(
                x=0.5, y=1.05,
                text=f"Chi-square test: p={p:.6f} (Significant)",
                showarrow=False,
                font=dict(color="red", size=12),
                xref="paper", yref="paper"
            )
        else:
            fig.add_annotation(
                x=0.5, y=1.05,
                text=f"Chi-square test: p={p:.6f} (Not Significant)",
                showarrow=False,
                font=dict(color="black", size=12),
                xref="paper", yref="paper"
            )
        
        fig.write_html(f'interactive_{factor}_vs_diagnosis.html')
        print(f"Interactive {factor} vs diagnosis chart saved as 'interactive_{factor}_vs_diagnosis.html'")
        
        # Static plot for HTML report
        plt.figure(figsize=(10, 6))
        percent_table['Malignant'].sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Malignancy Rate by {factor}')
        plt.ylabel('Percentage Malignant')
        plt.axhline(y=df['Diagnosis'].value_counts(normalize=True)['Malignant']*100, 
                    color='r', linestyle='--', label='Overall Average')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{factor}_vs_diagnosis.png')
        plt.close()
        
        # Store results
        results[factor] = {
            'chi2': chi2,
            'p_value': p,
            'contingency': contingency,
            'percentages': percent_table
        }
        
        # Add risk ratio if calculated
        if risk_ratio is not None:
            results[factor]['risk_ratio'] = risk_ratio
    
    # Summary of all risk factors
    print("\n=== RISK FACTORS SUMMARY ===")
    risk_summary = []
    
    for factor, result in results.items():
        p_value = result['p_value']
        significant = p_value < 0.05
        
        if 'risk_ratio' in result:
            risk_ratio = result['risk_ratio']
            risk_summary.append({
                'Risk Factor': factor,
                'p-value': p_value,
                'Significant': significant,
                'Risk Ratio': risk_ratio,
                'Increases Risk': risk_ratio > 1
            })
        else:
            # For non-binary factors, find highest malignancy category
            percentages = result['percentages']
            highest_category = percentages['Malignant'].idxmax()
            highest_percent = percentages.loc[highest_category, 'Malignant']
            
            risk_summary.append({
                'Risk Factor': factor,
                'p-value': p_value,
                'Significant': significant,
                'Highest Risk Category': highest_category,
                'Malignancy %': highest_percent
            })
    
    risk_summary_df = pd.DataFrame(risk_summary)
    if not risk_summary_df.empty:
        risk_summary_df = risk_summary_df.sort_values('p-value')
        print("\nRisk factors by significance:")
        print(risk_summary_df)
        
        # Create interactive summary table
        if 'Risk Ratio' in risk_summary_df.columns:
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Risk Factor', 'p-value', 'Significant', 'Risk Ratio', 'Increases Risk'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        risk_summary_df['Risk Factor'],
                        risk_summary_df['p-value'].round(6),
                        risk_summary_df['Significant'],
                        risk_summary_df['Risk Ratio'].round(2) if 'Risk Ratio' in risk_summary_df.columns else None,
                        risk_summary_df['Increases Risk'] if 'Increases Risk' in risk_summary_df.columns else None,
                    ],
                    fill_color=[['white', 'lightgrey'] * len(risk_summary_df)],
                    align='left'
                )
            )])
        else:
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Risk Factor', 'p-value', 'Significant', 'Highest Risk Category', 'Malignancy %'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        risk_summary_df['Risk Factor'],
                        risk_summary_df['p-value'].round(6),
                        risk_summary_df['Significant'],
                        risk_summary_df['Highest Risk Category'] if 'Highest Risk Category' in risk_summary_df.columns else None,
                        risk_summary_df['Malignancy %'].round(1) if 'Malignancy %' in risk_summary_df.columns else None,
                    ],
                    fill_color=[['white', 'lightgrey'] * len(risk_summary_df)],
                    align='left'
                )
            )])
        
        fig.update_layout(title='Risk Factors Summary')
        fig.write_html('interactive_risk_factors_summary.html')
        print("Interactive risk factors summary saved as 'interactive_risk_factors_summary.html'")
    
    return results

def one_way_anova(df):
    """Perform one-way ANOVA for categorical variables on numerical outcomes"""
    print("\n=== ONE-WAY ANOVA ANALYSIS ===")

    # List of categorical variables to test
    categorical_vars = ['Gender', 'Thyroid_Cancer_Risk', 'Smoking', 'Obesity', 'Family_History']

    # List of numerical variables to test
    numerical_vars = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

    # Check which variables exist in the dataset
    categorical_vars = [var for var in categorical_vars if var in df.columns]
    numerical_vars = [var for var in numerical_vars if var in df.columns]

    results = {}

    for cat_var in categorical_vars:
        if df[cat_var].nunique() > 10:
            continue

        for num_var in numerical_vars:
            print(f"\n--- ANOVA: {cat_var} effect on {num_var} ---")
            formula = f"{num_var} ~ C({cat_var})"
            try:
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                print(anova_table)
                p_value = anova_table.loc[f'C({cat_var})', 'PR(>F)']
                print(f"P-value: {p_value:.6f}")
                print(f"Significant effect: {'Yes' if p_value < 0.05 else 'No'}")
                results[f"{cat_var} on {num_var}"] = {
                    'anova_table': anova_table,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                print(f"Error processing {cat_var} on {num_var}: {e}")

    return results


def main():
    # Load dataset
    df = load_data("thyroid_cancer_risk_data.csv")

    # Basic overview
    basic_description(df)

    # Gender vs Diagnosis
    gender_vs_diagnosis(df)

    # Country vs Diagnosis
    country_vs_diagnosis(df)

    # Age comparison
    age_t_test(df)

    # Nodule size comparison
    nodule_size_t_test(df)

    # Risk factor analysis
    risk_factors_analysis(df)

    # One-way ANOVA
    one_way_anova(df)

if __name__ == "__main__":
    main()
