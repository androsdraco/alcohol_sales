import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, pointbiserialr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Alcohol Sales Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'campaign_date' not in st.session_state:
    st.session_state.campaign_date = '2023-01-10'

# Title
st.title("Alcohol Sales Analysis")
st.write("Analyze sales data before and after campaign periods")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📁 Data", "📊 EDA", "🔬 Hypothesis"])

# Tab 1: Data
with tab1:
    st.header("Data Input")
    
    # Data source selection
    data_option = st.radio("Choose data source:", 
                          ["Use Sample Data", "Upload CSV"], 
                          horizontal=True)
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"Data loaded: {len(df)} rows")
    
    # Campaign date selection
    st.subheader("Campaign Settings")
    campaign_date = st.date_input("Campaign Start Date", value=datetime(2023, 1, 10))
    st.session_state.campaign_date = campaign_date.strftime('%Y-%m-%d')
    
    # If no uploaded data, load sample
    if st.session_state.df is None:
        if st.button("Load Sample Data"):
            # Create sample data
            np.random.seed(42)
            n = 5000
            
            dates = pd.date_range('2021-01-01', '2023-12-31', n)
            brands = [f'Product {i}' for i in range(1, 31)]
            
            # Create data with campaign effect
            data = []
            for i in range(n):
                date = np.random.choice(dates)
                brand = np.random.choice(brands)
                
                # Different means before/after campaign
                if date < pd.Timestamp('2023-01-10'):
                    sale = np.random.lognormal(5.5, 1.2)
                else:
                    sale = np.random.lognormal(6.0, 1.1)
                
                sale = round(max(0, sale + np.random.normal(0, 50)), 2)
                
                data.append({
                    'date': date,
                    'brand': brand,
                    'sales': sale
                })
            
            df = pd.DataFrame(data)
            st.session_state.df = df
            st.success(f"Sample data loaded: {len(df)} rows")
    
    # Show data if loaded
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Basic cleaning
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Ensure required columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        if 'sales' in df.columns and df['sales'].dtype == 'object':
            df['sales'] = pd.to_numeric(df['sales'].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        
        # Remove NaNs
        df = df.dropna(subset=['date', 'sales', 'brand'], how='any')
        
        # Create campaign column
        df['campaign'] = np.where(df['date'] < pd.Timestamp(st.session_state.campaign_date), 'Before', 'After')
        
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Basic Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Unique Brands", df['brand'].nunique())
        with col2:
            st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
            st.metric("Days", (df['date'].max() - df['date'].min()).days)
        with col3:
            st.metric("Total Sales", f"${df['sales'].sum():,.0f}")
            st.metric("Avg Sale", f"${df['sales'].mean():.2f}")

# Tab 2: EDA
with tab2:
    st.header("Exploratory Data Analysis")
    
    if st.session_state.df is None:
        st.warning("Please load data in the Data tab first")
    else:
        df = st.session_state.df.copy()
        df['campaign'] = np.where(df['date'] < pd.Timestamp(st.session_state.campaign_date), 'Before', 'After')
        
        # Select EDA options
        st.subheader("Analysis Options")
        eda_options = st.multiselect(
            "Choose analyses to display:",
            ["Sales Distribution", "Time Trends", "Brand Analysis", "Campaign Comparison"],
            default=["Sales Distribution", "Campaign Comparison"]
        )
        
        if "Sales Distribution" in eda_options:
            st.write("### Sales Distribution")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Histogram
            axes[0].hist(df['sales'], bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Sales ($)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Sales Distribution')
            axes[0].grid(True, alpha=0.3)
            
            # Boxplot
            axes[1].boxplot(df['sales'])
            axes[1].set_ylabel('Sales ($)')
            axes[1].set_title('Sales Box Plot')
            axes[1].grid(True, alpha=0.3)
            
            # QQ plot
            from scipy import stats
            stats.probplot(df['sales'], dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot')
            axes[2].grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"${df['sales'].mean():.2f}")
            with col2:
                st.metric("Median", f"${df['sales'].median():.2f}")
            with col3:
                st.metric("Std Dev", f"${df['sales'].std():.2f}")
            with col4:
                st.metric("Skewness", f"{df['sales'].skew():.3f}")
        
        if "Time Trends" in eda_options:
            st.write("### Time Trends")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Daily trend
            daily = df.groupby('date')['sales'].sum().reset_index()
            axes[0].plot(daily['date'], daily['sales'], linewidth=1)
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Total Sales ($)')
            axes[0].set_title('Daily Sales Trend')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Monthly trend
            df['month'] = df['date'].dt.strftime('%Y-%m')
            monthly = df.groupby('month')['sales'].sum().reset_index()
            axes[1].bar(range(len(monthly)), monthly['sales'])
            axes[1].set_xlabel('Month')
            axes[1].set_ylabel('Total Sales ($)')
            axes[1].set_title('Monthly Sales')
            axes[1].set_xticks(range(len(monthly)))
            axes[1].set_xticklabels(monthly['month'], rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        if "Brand Analysis" in eda_options:
            st.write("### Brand Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 brands
                top_brands = df.groupby('brand')['sales'].sum().nlargest(10).sort_values()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                top_brands.plot(kind='barh', ax=ax)
                ax.set_xlabel('Total Sales ($)')
                ax.set_title('Top 10 Brands by Sales')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Brand count distribution
                brand_counts = df['brand'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                brand_counts.head(10).plot(kind='bar', ax=ax)
                ax.set_xlabel('Brand')
                ax.set_ylabel('Number of Sales')
                ax.set_title('Top 10 Brands by Transaction Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        if "Campaign Comparison" in eda_options:
            st.write("### Campaign Comparison")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Boxplot by campaign
            sns.boxplot(data=df, x='campaign', y='sales', ax=axes[0])
            axes[0].set_xlabel('Campaign Period')
            axes[0].set_ylabel('Sales ($)')
            axes[0].set_title('Sales Distribution by Campaign')
            axes[0].grid(True, alpha=0.3)
            
            # Violin plot
            sns.violinplot(data=df, x='campaign', y='sales', ax=axes[1])
            axes[1].set_xlabel('Campaign Period')
            axes[1].set_ylabel('Sales ($)')
            axes[1].set_title('Sales Distribution (Violin Plot)')
            axes[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Campaign statistics
            st.write("#### Campaign Statistics")
            campaign_stats = df.groupby('campaign').agg({
                'sales': ['count', 'sum', 'mean', 'median', 'std']
            }).round(2)
            
            st.dataframe(campaign_stats)

# Tab 3: Hypothesis Testing
with tab3:
    st.header("Hypothesis Testing")
    
    if st.session_state.df is None:
        st.warning("Please load data in the Data tab first")
    else:
        df = st.session_state.df.copy()
        df['campaign'] = np.where(df['date'] < pd.Timestamp(st.session_state.campaign_date), 'Before', 'After')
        
        st.write("### Research Question")
        st.write("**Does the marketing campaign have a statistically significant impact on sales?**")
        
        st.write("### Hypothesis")
        st.write("- **Null Hypothesis (H₀):** No difference in sales before vs after campaign")
        st.write("- **Alternative Hypothesis (H₁):** Sales are different after the campaign")
        
        # Separate data
        before = df[df['campaign'] == 'Before']['sales']
        after = df[df['campaign'] == 'After']['sales']
        
        # Display basic comparison
        st.write("### Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Campaign:**")
            st.write(f"- N = {len(before):,}")
            st.write(f"- Mean = ${before.mean():.2f}")
            st.write(f"- Median = ${before.median():.2f}")
            st.write(f"- Std Dev = ${before.std():.2f}")
        
        with col2:
            st.write("**After Campaign:**")
            st.write(f"- N = {len(after):,}")
            st.write(f"- Mean = ${after.mean():.2f}")
            st.write(f"- Median = ${after.median():.2f}")
            st.write(f"- Std Dev = ${after.std():.2f}")
        
        # Calculate differences
        mean_diff = after.mean() - before.mean()
        pct_change = (mean_diff / before.mean()) * 100
        
        st.write("### Difference")
        st.write(f"- Mean Difference: ${mean_diff:.2f}")
        st.write(f"- Percentage Change: {pct_change:.1f}%")
        
        # Statistical tests
        st.write("### Statistical Tests")
        
        # T-test
        t_stat, p_ttest = ttest_ind(after, before, equal_var=False)
        
        # Mann-Whitney
        u_stat, p_mw = mannwhitneyu(after, before, alternative='two-sided')
        
        # Effect size
        n1, n2 = len(before), len(after)
        sd_pooled = np.sqrt(((n1-1)*before.var() + (n2-1)*after.var()) / (n1+n2-2))
        cohens_d = mean_diff / sd_pooled if sd_pooled != 0 else 0
        
        # Display test results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**T-Test (Parametric)**")
            st.write(f"- t-statistic = {t_stat:.3f}")
            st.write(f"- p-value = {p_ttest:.6f}")
            if p_ttest < 0.05:
                st.success("✅ Statistically significant (p < 0.05)")
            else:
                st.warning("⚠️ Not statistically significant")
        
        with col2:
            st.write("**Mann-Whitney U (Non-parametric)**")
            st.write(f"- U-statistic = {u_stat:,.0f}")
            st.write(f"- p-value = {p_mw:.6f}")
            if p_mw < 0.05:
                st.success("✅ Statistically significant (p < 0.05)")
            else:
                st.warning("⚠️ Not statistically significant")
        
        # Effect size
        st.write("### Effect Size")
        st.write(f"- Cohen's d = {cohens_d:.3f}")
        
        if abs(cohens_d) > 0.8:
            st.write("Interpretation: Large effect size")
        elif abs(cohens_d) > 0.5:
            st.write("Interpretation: Medium effect size")
        elif abs(cohens_d) > 0.2:
            st.write("Interpretation: Small effect size")
        else:
            st.write("Interpretation: Negligible effect size")
        
        # Conclusion
        st.write("### Conclusion")
        
        if p_ttest < 0.05:
            if mean_diff > 0:
                st.success(f"""
                **The campaign had a statistically significant positive impact on sales.**
                
                Sales increased by {pct_change:.1f}% on average (from ${before.mean():.2f} to ${after.mean():.2f}).
                
                This difference is statistically significant (p = {p_ttest:.4f}) with a {cohens_d:.3f} effect size.
                """)
            else:
                st.error(f"""
                **The campaign had a statistically significant negative impact on sales.**
                
                Sales decreased by {abs(pct_change):.1f}% on average (from ${before.mean():.2f} to ${after.mean():.2f}).
                
                This difference is statistically significant (p = {p_ttest:.4f}) with a {cohens_d:.3f} effect size.
                """)
        else:
            st.info(f"""
            **No statistically significant impact detected.**
            
            Sales changed by {pct_change:+.1f}% on average (from ${before.mean():.2f} to ${after.mean():.2f}).
            
            This difference is not statistically significant (p = {p_ttest:.4f}).
            
            The observed change could be due to random variation.
            """)
        
        # Confidence intervals
        st.write("### 95% Confidence Intervals")
        
        from scipy.stats import t
        def confidence_interval(data, confidence=0.95):
            n = len(data)
            mean = np.mean(data)
            std_err = stats.sem(data)
            h = std_err * t.ppf((1 + confidence) / 2, n - 1)
            return mean - h, mean + h
        
        before_ci = confidence_interval(before)
        after_ci = confidence_interval(after)
        
        st.write(f"- Before: ${before_ci[0]:.2f} to ${before_ci[1]:.2f}")
        st.write(f"- After:  ${after_ci[0]:.2f} to ${after_ci[1]:.2f}")
        
        # Visualization of test results
        st.write("### Test Visualization")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Boxplot with p-value
        sns.boxplot(data=df, x='campaign', y='sales', ax=axes[0])
        axes[0].set_title(f'Sales by Campaign\np = {p_ttest:.4f}')
        axes[0].set_ylabel('Sales ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram overlay
        axes[1].hist(before, bins=30, alpha=0.6, label='Before', density=True)
        axes[1].hist(after, bins=30, alpha=0.6, label='After', density=True)
        axes[1].set_title('Sales Distribution Comparison')
        axes[1].set_xlabel('Sales ($)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Footer
st.divider()
st.write("**Alcohol Sales Analysis** | Basic Streamlit App")