import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, pointbiserialr, skew, kurtosis
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Alcohol Sales Analytics",
    layout="wide"
)

# Title
st.title("Alcohol Sales Analytics Dashboard")

# Generate realistic dataset with exact specifications
def generate_realistic_dataset():
    np.random.seed(42)
    
    n_records = 7500
    brands = [f'Product_{i:02d}' for i in range(1, 26)]
    
    # Date range from 2022-01-01 to 2023-03-10
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2023-03-10')
    
    # Generate dates with realistic distribution
    days_range = (end_date - start_date).days
    days_since_start = np.random.beta(2, 1, n_records) * days_range
    dates = start_date + pd.to_timedelta(days_since_start, unit='D')
    
    dates = pd.to_datetime(dates)
    brand_weights = np.random.dirichlet(np.ones(25)) * 100
    
    # Campaign starts exactly 2 months before end date
    campaign_date = end_date - pd.DateOffset(months=2)
    
    data = []
    for i in range(n_records):
        date = dates[i]
        brand_idx = np.random.choice(range(25), p=brand_weights/brand_weights.sum())
        brand = brands[brand_idx]
        
        brand_base = 500 + brand_idx * 40
        day_of_year = date.dayofyear
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        weekend_boost = 1.25 if date.weekday() >= 5 else 1.0
        
        # Campaign effect - more pronounced in second month
        if date < campaign_date:
            campaign_multiplier = 1.0
        else:
            # Different response patterns based on time since campaign start
            days_since_campaign = (date - campaign_date).days
            if days_since_campaign < 30:  # First month
                brand_response = 0.5 + (brand_idx % 10) / 20
                campaign_multiplier = 1.0 + brand_response * 0.15
            else:  # Second month
                brand_response = 0.8 + (brand_idx % 10) / 20
                campaign_multiplier = 1.0 + brand_response * 0.3
        
        base_sale = brand_base * seasonal * weekend_boost * campaign_multiplier
        sale = np.random.lognormal(np.log(base_sale), 0.4)
        sale = round(max(10, sale), 2)
        
        data.append({
            'date': date,
            'brand': brand,
            'sales': sale
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    df['campaign'] = np.where(df['date'] < campaign_date, 'Before', 'After')
    
    return df, campaign_date, end_date

# Generate data
df, campaign_date, end_date = generate_realistic_dataset()

# Create two-month periods for comparison
def create_two_month_periods(df, campaign_date, end_date):
    """Create two-month periods before and after campaign start"""
    # Create period labels
    periods = []
    
    # Calculate 6 two-month periods before campaign (1 year)
    for i in range(6, 0, -1):
        period_end = campaign_date - pd.DateOffset(days=(i-1)*60)
        period_start = period_end - pd.DateOffset(days=60)
        periods.append({
            'period': f'P-{i}',
            'start': period_start,
            'end': period_end - timedelta(days=1),  # Exclude end date
            'label': f'{(period_start).strftime("%b")}-{(period_end - timedelta(days=1)).strftime("%b")} {(period_start.year)}',
            'campaign': 'Before'
        })
    
    # Campaign period (2 months)
    campaign_period = {
        'period': 'Campaign',
        'start': campaign_date,
        'end': end_date,
        'label': f'Campaign: {campaign_date.strftime("%b %d")}-{end_date.strftime("%b %d")}',
        'campaign': 'After'
    }
    periods.append(campaign_period)
    
    # Assign each record to a period
    df['period'] = 'Other'
    df['period_label'] = 'Other'
    
    for period in periods:
        mask = (df['date'] >= period['start']) & (df['date'] <= period['end'])
        df.loc[mask, 'period'] = period['period']
        df.loc[mask, 'period_label'] = period['label']
    
    return df, periods

df, periods = create_two_month_periods(df, campaign_date, end_date)

# Dashboard Metrics
st.markdown("---")
st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_sales = df['sales'].sum()
    st.metric("Total Revenue", f"${total_sales:,.0f}")

with col2:
    avg_sale = df['sales'].mean()
    st.metric("Average Transaction", f"${avg_sale:.2f}")

with col3:
    st.metric("Total Transactions", f"{len(df):,}")

with col4:
    st.metric("Unique Products", df['brand'].nunique())

with col5:
    date_range = f"{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}"
    st.metric("Analysis Period", date_range)

st.markdown("---")

# Two-Month Periods Analysis
st.subheader("Two-Month Periods Comparison")

col1, col2 = st.columns([2, 1])

with col1:
    # Calculate period metrics
    period_data = df[df['period'] != 'Other'].groupby(['period', 'period_label', 'campaign']).agg({
        'sales': ['sum', 'mean', 'count', 'std']
    }).round(2).reset_index()
    
    period_data.columns = ['Period', 'Label', 'Campaign', 'Total_Sales', 'Avg_Sales', 'Transaction_Count', 'Std_Sales']
    
    # Sort periods chronologically
    period_order = [f'P-{i}' for i in range(6, 0, -1)] + ['Campaign']
    period_data['Period_Order'] = pd.Categorical(period_data['Period'], categories=period_order, ordered=True)
    period_data = period_data.sort_values('Period_Order')
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Bar chart for total sales by period
    x = np.arange(len(period_data))
    colors = ['steelblue' if p == 'Before' else 'forestgreen' for p in period_data['Campaign']]
    
    bars = ax1.bar(x, period_data['Total_Sales'], color=colors, alpha=0.7)
    ax1.set_xlabel('Two-Month Period')
    ax1.set_ylabel('Total Sales ($)')
    ax1.set_title('Total Sales by Two-Month Period', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(period_data['Label'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, period_data['Total_Sales']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + (0.01 * max(period_data['Total_Sales'])),
                f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Line chart for average sales
    ax2.plot(x, period_data['Avg_Sales'], marker='o', linewidth=2, markersize=8, color='darkorange')
    ax2.set_xlabel('Two-Month Period')
    ax2.set_ylabel('Average Sale ($)')
    ax2.set_title('Average Sale by Two-Month Period', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(period_data['Label'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Highlight campaign period
    campaign_idx = np.where(period_data['Period'] == 'Campaign')[0][0]
    ax1.axvspan(campaign_idx - 0.4, campaign_idx + 0.4, alpha=0.2, color='green')
    ax2.axvspan(campaign_idx - 0.4, campaign_idx + 0.4, alpha=0.2, color='green')
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.write("Period Performance Metrics")
    
    # Show last 4 periods including campaign
    recent_periods = period_data.tail(4).reset_index(drop=True)
    
    for idx, row in recent_periods.iterrows():
        if row['Period'] == 'Campaign':
            # Compare with previous period
            if idx > 0:
                prev_avg = recent_periods.iloc[idx-1]['Avg_Sales']
                change_pct = ((row['Avg_Sales'] - prev_avg) / prev_avg) * 100
                delta_display = f"{change_pct:+.1f}%"
            else:
                delta_display = None
            
            st.metric(
                row['Label'],
                f"${row['Avg_Sales']:.2f}",
                delta=delta_display
            )
            
            with st.expander(f"Details for {row['Label']}"):
                st.write(f"Total Sales: ${row['Total_Sales']:,.0f}")
                st.write(f"Transactions: {row['Transaction_Count']:,}")
                st.write(f"Avg Sale: ${row['Avg_Sales']:.2f}")
                st.write(f"Std Dev: ${row['Std_Sales']:.2f}")
        else:
            st.metric(
                row['Label'],
                f"${row['Avg_Sales']:.2f}"
            )
    
    # Overall comparison
    st.write("---")
    st.write("Campaign vs Pre-Campaign Average")
    
    campaign_avg = period_data[period_data['Period'] == 'Campaign']['Avg_Sales'].values[0]
    pre_campaign_avg = period_data[period_data['Period'] != 'Campaign']['Avg_Sales'].mean()
    overall_change = ((campaign_avg - pre_campaign_avg) / pre_campaign_avg) * 100
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Pre-Campaign Avg", f"${pre_campaign_avg:.2f}")
    with col_b:
        st.metric("Campaign Avg", f"${campaign_avg:.2f}", delta=f"{overall_change:+.1f}%")

st.markdown("---")

# Statistical Analysis with Fair Comparison
st.subheader("Statistical Analysis with Two-Month Periods")

col1, col2 = st.columns(2)

with col1:
    st.write("Period-by-Period Comparison")
    
    # Create comparison table
    comparison_data = []
    
    for i in range(1, 7):  # Compare campaign with each of the 6 pre-campaign periods
        pre_period = f'P-{i}'
        pre_avg = period_data[period_data['Period'] == pre_period]['Avg_Sales'].values[0]
        campaign_avg = period_data[period_data['Period'] == 'Campaign']['Avg_Sales'].values[0]
        change_pct = ((campaign_avg - pre_avg) / pre_avg) * 100
        
        comparison_data.append({
            'Pre-Campaign Period': f'Period {i}',
            'Pre-Campaign Avg': f"${pre_avg:.2f}",
            'Campaign Avg': f"${campaign_avg:.2f}",
            'Change': f"{change_pct:+.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

# In the Statistical Analysis section (around line 288):
with col2:
    st.write("Statistical Tests")
    
    # Extract sales data for statistical tests
    campaign_sales = df[df['period'] == 'Campaign']['sales']  # Changed 'Period' to 'period'
    all_pre_campaign_sales = df[df['campaign'] == 'Before']['sales']
    
    # Statistical tests
    t_stat, p_ttest = ttest_ind(campaign_sales, all_pre_campaign_sales, equal_var=False)
    
    # Effect size
    n1, n2 = len(campaign_sales), len(all_pre_campaign_sales)
    sd_pooled = np.sqrt(((n1-1)*campaign_sales.var() + (n2-1)*all_pre_campaign_sales.var()) / (n1+n2-2))
    cohens_d = (campaign_sales.mean() - all_pre_campaign_sales.mean()) / sd_pooled
    
    # Confidence intervals
    from scipy.stats import t
    
    def confidence_interval(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h
    
    campaign_ci = confidence_interval(campaign_sales)
    pre_campaign_ci = confidence_interval(all_pre_campaign_sales)
    
    st.metric("T-Test P-Value", f"{p_ttest:.6f}")
    st.metric("Statistical Significance", "Significant" if p_ttest < 0.05 else "Not Significant")
    st.metric("Effect Size (Cohen's d)", f"{cohens_d:.3f}")
    
    st.write("95% Confidence Intervals:")
    st.write(f"Campaign: ${campaign_ci[0]:.2f} - ${campaign_ci[1]:.2f}")
    st.write(f"Pre-Campaign: ${pre_campaign_ci[0]:.2f} - ${pre_campaign_ci[1]:.2f}")

st.markdown("---")

# Product Performance in Two-Month Context
st.subheader("Product Performance During Campaign Period")

col1, col2 = st.columns([1, 2])

with col1:
    # Analysis options
    analysis_focus = st.radio(
        "Focus Analysis On",
        ["Top Performers", "Most Improved", "Consistency Analysis", "Price Tier Analysis"]
    )
    
    # Time segmentation within campaign
    st.write("Campaign Time Segmentation")
    show_first_month = st.checkbox("Show First Month Performance", value=True)
    show_second_month = st.checkbox("Show Second Month Performance", value=True)
    
    if show_first_month and show_second_month:
        # Create month segments
        first_month_end = campaign_date + pd.DateOffset(days=30)
        
        first_month_sales = df[(df['date'] >= campaign_date) & (df['date'] < first_month_end)]
        second_month_sales = df[(df['date'] >= first_month_end) & (df['date'] <= end_date)]
        
        st.write(f"First Month: {campaign_date.strftime('%b %d')} - {first_month_end.strftime('%b %d')}")
        st.write(f"Second Month: {first_month_end.strftime('%b %d')} - {end_date.strftime('%b %d')}")

with col2:
    # Product performance during campaign
    campaign_products = df[df['period'] == 'Campaign'].groupby('brand').agg({  # Changed 'Period' to 'period'
        'sales': ['sum', 'mean', 'count', 'std']
    }).round(2)
    
    campaign_products.columns = ['Total_Sales', 'Avg_Sales', 'Transaction_Count', 'Std_Sales']
    campaign_products = campaign_products.sort_values('Total_Sales', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate position for grouped bars
    x = np.arange(len(campaign_products))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, campaign_products['Total_Sales'], width, 
                  label='Total Sales', alpha=0.7, color='steelblue')
    bars2 = ax.bar(x + width/2, campaign_products['Avg_Sales'] * 100, width,  # Scaled for visibility
                  label='Avg Sales (x100)', alpha=0.7, color='darkorange')
    
    ax.set_xlabel('Product')
    ax.set_ylabel('Sales ($) / Avg Sales (x100)')
    ax.set_title('Top 10 Products During Campaign Period', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(campaign_products.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if bars == bars1:
                value = f'${height:,.0f}'
            else:
                value = f'${height/100:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 * max([b.get_height() for b in bars])),
                   value, ha='center', va='bottom', fontsize=8)
    
    st.pyplot(fig)

st.markdown("---")

# Executive Summary with Fair Periods Context
st.subheader("Executive Summary")

summary_col1, summary_col2 = st.columns([2, 1])

with summary_col1:
    # Calculate comprehensive metrics
    campaign_period_data = period_data[period_data['Period'] == 'Campaign'].iloc[0]
    pre_campaign_periods_data = period_data[period_data['Period'] != 'Campaign']
    
    campaign_total = campaign_period_data['Total_Sales']
    campaign_avg = campaign_period_data['Avg_Sales']
    campaign_transactions = campaign_period_data['Transaction_Count']
    
    pre_campaign_avg_total = pre_campaign_periods_data['Total_Sales'].mean()
    pre_campaign_avg_sale = pre_campaign_periods_data['Avg_Sales'].mean()
    pre_campaign_avg_transactions = pre_campaign_periods_data['Transaction_Count'].mean()
    
    total_change = ((campaign_total - pre_campaign_avg_total) / pre_campaign_avg_total) * 100
    avg_change = ((campaign_avg - pre_campaign_avg_sale) / pre_campaign_avg_sale) * 100
    transaction_change = ((campaign_transactions - pre_campaign_avg_transactions) / pre_campaign_avg_transactions) * 100
    
    # Create summary points
    summary_points = []
    
    summary_points.append(f"**Campaign Period Performance**: {campaign_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')} (2 months)")
    
    summary_points.append(f"**Total Revenue**: ${campaign_total:,.0f} ({total_change:+.1f}% vs pre-campaign 2-month average of ${pre_campaign_avg_total:,.0f})")
    
    summary_points.append(f"**Average Transaction Value**: ${campaign_avg:.2f} ({avg_change:+.1f}% vs pre-campaign average of ${pre_campaign_avg_sale:.2f})")
    
    summary_points.append(f"**Transaction Volume**: {campaign_transactions:,} transactions ({transaction_change:+.1f}% vs pre-campaign average of {pre_campaign_avg_transactions:,.0f})")
    
    summary_points.append(f"**Statistical Significance**: {'Statistically significant improvement' if p_ttest < 0.05 else 'No statistically significant difference'} (p={p_ttest:.4f})")
    
    summary_points.append(f"**Effect Size**: Cohen's d = {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
    
    summary_points.append(f"**Data Fairness**: Campaign period compared against 6 previous two-month periods for equitable comparison")
    
    for point in summary_points:
        st.write(f"• {point}")

with summary_col2:
    st.write("Strategic Recommendations")
    
    # Generate context-aware recommendations
    if p_ttest < 0.05 and avg_change > 15:
        recommendations = [
            "**Scale Successful Campaign Elements**: Campaign demonstrated strong 2-month performance",
            "**Maintain Momentum**: Continue successful tactics into next quarter",
            "**Optimize Top Products**: Focus resources on highest performers",
            "**Expand Successful Segments**: Identify and replicate winning patterns",
            "**Monitor Sustained Performance**: Track if 2-month gains persist"
        ]
    elif p_ttest < 0.05 and avg_change > 0:
        recommendations = [
            "**Refine Campaign Execution**: Positive but moderate results suggest optimization potential",
            "**Focus on High-Performing Periods**: Analyze what worked best within the 2 months",
            "**Test Variations**: Experiment with different approaches in next campaign",
            "**Improve Consistency**: Work on maintaining performance throughout entire period",
            "**Plan Follow-up Analysis**: Monitor longer-term effects"
        ]
    elif p_ttest < 0.05 and avg_change < 0:
        recommendations = [
            "**Review Campaign Strategy**: Negative impact suggests need for adjustment",
            "**Analyze Timing**: Consider if 2-month period was optimal",
            "**Segment Performance**: Identify what didn't work and why",
            "**Test Alternative Approaches**: Different messaging or timing may work better",
            "**Consider Market Conditions**: External factors may have influenced results"
        ]
    else:
        recommendations = [
            "**Extend Analysis Period**: Consider longer observation window",
            "**Increase Data Collection**: More comprehensive data needed",
            "**Segment Analysis**: Break down by product type or customer segment",
            "**Test Different Timeframes**: Alternative 2-month periods may show different results",
            "**Evaluate Campaign Objectives**: Revisit goals and measurement criteria"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Data fairness note
    st.write("---")
    st.write("**Note on Fair Comparison**:")
    st.write("Analysis uses two-month periods to ensure equitable comparison. The campaign period (2 months) is compared against the average of 6 previous two-month periods, avoiding misleading comparisons with longer or shorter timeframes.")

# Footer with Export Options
st.markdown("---")
col_export1, col_export2, col_export3 = st.columns(3)

with col_export1:
    if st.button("Download Executive Summary"):
        summary_data = {
            'Analysis Date': [datetime.now().strftime('%Y-%m-%d')],
            'Campaign Period': [f"{campaign_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"],
            'Campaign Duration': ["2 months"],
            'Campaign Total Sales': [f"${campaign_total:,.0f}"],
            'Campaign Average Sale': [f"${campaign_avg:.2f}"],
            'Pre-Campaign 2-Month Average Sales': [f"${pre_campaign_avg_total:,.0f}"],
            'Percentage Change vs Pre-Campaign Average': [f"{total_change:+.1f}%"],
            'Statistical Significance (P-Value)': [f"{p_ttest:.6f}"],
            'Effect Size (Cohen d)': [f"{cohens_d:.3f}"],
            'Data Comparison Method': ["2-month period averages for fair comparison"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Click to Download Summary",
            data=csv,
            file_name="alcohol_sales_executive_summary.csv",
            mime="text/csv"
        )

with col_export2:
    if st.button("Download Two-Month Period Data"):
        export_data = period_data.copy()
        export_data = export_data[['Period', 'Label', 'Campaign', 'Total_Sales', 'Avg_Sales', 'Transaction_Count']]
        export_data.columns = ['Period_ID', 'Time_Period', 'Campaign_Status', 'Total_Sales', 'Average_Sale', 'Transaction_Count']
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Click to Download Period Data",
            data=csv,
            file_name="two_month_period_analysis.csv",
            mime="text/csv"
        )

with col_export3:
    if st.button("Download Campaign Product Performance"):
        product_data = df[df['period'] == 'Campaign'].groupby('brand').agg({ 
            'sales': ['sum', 'mean', 'count']
        }).round(2)
        
        product_data.columns = ['Total_Sales', 'Average_Sale', 'Transaction_Count']
        product_data = product_data.sort_values('Total_Sales', ascending=False)
        csv = product_data.to_csv()
        st.download_button(
            label="Click to Download Product Data",
            data=csv,
            file_name="campaign_product_performance.csv",
            mime="text/csv"
        )

# Final footer
st.markdown("---")
st.write(f"Alcohol Sales Analytics Dashboard | Data Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')} | "
         f"Campaign Period: {campaign_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (2 months) | "
         f"Comparison Method: Two-month period averages for fair analysis | "
         f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
