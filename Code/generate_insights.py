"""
Generate additional insights and visualizations for the perovskite ML analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("Generating additional visualizations...")

# Load data
df = pd.read_csv('04_feature_engineered_data.csv')
feature_importance = pd.read_csv('08_feature_importance.csv')

# Create comprehensive insights figure
fig = plt.figure(figsize=(20, 12))

# 1. PCE Distribution
ax1 = plt.subplot(3, 3, 1)
df['JV_default_PCE'].hist(bins=50, color='steelblue', edgecolor='black', alpha=0.7, ax=ax1)
ax1.axvline(df['JV_default_PCE'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["JV_default_PCE"].mean():.2f}%')
ax1.axvline(df['JV_default_PCE'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {df["JV_default_PCE"].median():.2f}%')
ax1.set_xlabel('PCE (%)')
ax1.set_ylabel('Frequency')
ax1.set_title('PCE Distribution (42,279 samples)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. PCE by FA presence
ax2 = plt.subplot(3, 3, 2)
if 'has_FA' in df.columns:
    pce_by_fa = df.groupby('has_FA')['JV_default_PCE'].agg(['mean', 'std', 'count'])
    bars = ax2.bar(['Without FA', 'With FA'], pce_by_fa['mean'], 
                   yerr=pce_by_fa['std'], capsize=5, color=['coral', 'lightgreen'],
                   edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Mean PCE (%)')
    ax2.set_title('PCE by Formamidinium (FA) Presence', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, (bar, count) in enumerate(zip(bars, pce_by_fa['count'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count:,}', ha='center', va='bottom', fontsize=9)

# 3. PCE by Pb presence
ax3 = plt.subplot(3, 3, 3)
if 'has_Pb' in df.columns:
    pce_by_pb = df.groupby('has_Pb')['JV_default_PCE'].agg(['mean', 'std', 'count'])
    bars = ax3.bar(['Without Pb', 'With Pb'], pce_by_pb['mean'], 
                   yerr=pce_by_pb['std'], capsize=5, color=['lightcoral', 'lightblue'],
                   edgecolor='black', alpha=0.7)
    ax3.set_ylabel('Mean PCE (%)')
    ax3.set_title('PCE by Lead (Pb) Presence', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, count) in enumerate(zip(bars, pce_by_pb['count'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count:,}', ha='center', va='bottom', fontsize=9)

# 4. A-site ion distribution
ax4 = plt.subplot(3, 3, 4)
if all(col in df.columns for col in ['has_MA', 'has_FA', 'has_Cs']):
    a_site_counts = {
        'MA': df['has_MA'].sum(),
        'FA': df['has_FA'].sum(),
        'Cs': df['has_Cs'].sum()
    }
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
    wedges, texts, autotexts = ax4.pie(a_site_counts.values(), labels=a_site_counts.keys(),
                                        autopct='%1.1f%%', colors=colors_pie,
                                        startangle=90, textprops={'fontsize': 10})
    ax4.set_title('A-site Ion Distribution', fontweight='bold')

# 5. X-site ion distribution
ax5 = plt.subplot(3, 3, 5)
if all(col in df.columns for col in ['has_I', 'has_Br', 'has_Cl']):
    x_site_counts = {
        'I': df['has_I'].sum(),
        'Br': df['has_Br'].sum(),
        'Cl': df['has_Cl'].sum()
    }
    colors_pie2 = ['#c9b3ff', '#ffb3e6', '#ffd699']
    wedges, texts, autotexts = ax5.pie(x_site_counts.values(), labels=x_site_counts.keys(),
                                        autopct='%1.1f%%', colors=colors_pie2,
                                        startangle=90, textprops={'fontsize': 10})
    ax5.set_title('X-site Ion Distribution', fontweight='bold')

# 6. PCE by composition combination
ax6 = plt.subplot(3, 3, 6)
if all(col in df.columns for col in ['has_FA', 'has_Pb', 'has_MA']):
    compositions = {
        'FA-Pb': ((df['has_FA'] == 1) & (df['has_Pb'] == 1)),
        'MA-Pb': ((df['has_MA'] == 1) & (df['has_Pb'] == 1)),
        'Cs-Pb': ((df['has_Cs'] == 1) & (df['has_Pb'] == 1)),
    }
    
    comp_stats = []
    for name, mask in compositions.items():
        if mask.sum() > 0:
            comp_stats.append({
                'Composition': name,
                'Mean_PCE': df.loc[mask, 'JV_default_PCE'].mean(),
                'Count': mask.sum()
            })
    
    comp_df = pd.DataFrame(comp_stats)
    bars = ax6.barh(comp_df['Composition'], comp_df['Mean_PCE'], color='teal', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Mean PCE (%)')
    ax6.set_title('Mean PCE by Composition Type', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, count) in enumerate(zip(bars, comp_df['Count'])):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f' n={count:,}', ha='left', va='center', fontsize=9)

# 7. Top 10 features
ax7 = plt.subplot(3, 3, 7)
top_10 = feature_importance.head(10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
ax7.barh(range(len(top_10)), top_10['Importance'], color=colors, edgecolor='black')
ax7.set_yticks(range(len(top_10)))
ax7.set_yticklabels([f"{i+1}. {feat[:30]}" for i, feat in enumerate(top_10['Feature'])], fontsize=9)
ax7.set_xlabel('Importance Score')
ax7.set_title('Top 10 Predictive Features', fontweight='bold')
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

# 8. PCE vs Feature Importance Top Feature
ax8 = plt.subplot(3, 3, 8)
if 'has_FA' in df.columns:
    violin_data = [df[df['has_FA'] == 0]['JV_default_PCE'].dropna(),
                   df[df['has_FA'] == 1]['JV_default_PCE'].dropna()]
    parts = ax8.violinplot(violin_data, positions=[0, 1], showmeans=True, showmedians=True)
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels(['Without FA', 'With FA'])
    ax8.set_ylabel('PCE (%)')
    ax8.set_title('PCE Distribution by FA (Violin Plot)', fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

# 9. Summary statistics table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Dataset:
  • Total samples: {len(df):,}
  • Features: {len(df.columns)}
  
PCE Statistics:
  • Mean: {df['JV_default_PCE'].mean():.2f}%
  • Median: {df['JV_default_PCE'].median():.2f}%
  • Std Dev: {df['JV_default_PCE'].std():.2f}%
  • Range: [{df['JV_default_PCE'].min():.2f}%, {df['JV_default_PCE'].max():.2f}%]
  
Top Feature:
  • has_FA (17.5% importance)
  • Mean PCE with FA: {df[df['has_FA']==1]['JV_default_PCE'].mean():.2f}%
  • Mean PCE w/o FA: {df[df['has_FA']==0]['JV_default_PCE'].mean():.2f}%
  
Model Performance:
  • Best: Random Forest
  • Test RMSE: 3.74%
  • Test R²: 0.496
"""

ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Perovskite Solar Cell PCE Analysis - Comprehensive Insights', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('10_comprehensive_insights.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 10_comprehensive_insights.png")
plt.close()

# Create a correlation heatmap for engineered features
print("\nGenerating correlation matrix...")
fig, ax = plt.subplots(figsize=(12, 10))

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target and keep only engineered features
engineered_features = [col for col in numeric_cols if col in [
    'has_MA', 'has_FA', 'has_Cs', 'has_Pb', 'has_Sn', 'has_I', 'has_Br', 'has_Cl',
    'ETL_stack_sequence_count', 'HTL_stack_sequence_count', 
    'Substrate_stack_sequence_count', 'Backcontact_stack_sequence_count',
    'JV_default_PCE'
]]

if len(engineered_features) > 2:
    corr = df[engineered_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title('Feature Correlation Matrix (Engineered Features)', 
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('11_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 11_correlation_matrix.png")
    plt.close()

print("\n✓ All additional visualizations generated!")
print("\nGenerated files:")
print("  - 10_comprehensive_insights.png")
print("  - 11_correlation_matrix.png")
