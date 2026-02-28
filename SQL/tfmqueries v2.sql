USE TFMsql2026;
select * from sales;
select * from usd_rates;
select * from date_dim;

-- 1. Overall Average Sales vs. Campaign Average
SELECT 
    s.campaign,
    AVG(s.sales) AS campaign_avg_sale,
    (SELECT AVG(sales) FROM sales WHERE sales > 0) AS overall_avg_sale,
    AVG(s.sales) - (SELECT AVG(sales) FROM sales WHERE sales > 0) AS diff_from_overall
FROM sales s
WHERE s.sales > 0 AND s.campaign IS NOT NULL
GROUP BY s.campaign;

-- 2. Seasonal Sales Performance
SELECT 
    d.season,
    COUNT(DISTINCT s.date) AS days_with_sales,
    SUM(s.sales) AS total_sales,
    AVG(s.sales) AS avg_sale_per_transaction,
    SUM(s.sales) / COUNT(DISTINCT s.date) AS avg_daily_sales,
    MAX(s.sales) AS max_single_sale
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed from d.date_id to d.id
WHERE s.sales > 0
GROUP BY d.season
ORDER BY avg_daily_sales DESC;

-- 3. Top-Selling Product per Campaign
SELECT 
    s.campaign,
    s.brand,
    SUM(s.sales) AS brand_total
FROM sales s
WHERE s.sales > 0
GROUP BY s.campaign, s.brand
HAVING SUM(s.sales) = (
    SELECT MAX(brand_total)
    FROM (
        SELECT 
            campaign,
            brand,
            SUM(sales) AS brand_total
        FROM sales
        WHERE sales > 0
        GROUP BY campaign, brand
    ) AS inner_totals
    WHERE inner_totals.campaign = s.campaign
)
ORDER BY s.campaign;

-- 4. Campaign Performance by Season
SELECT 
    d.season,
    s.campaign,
    SUM(s.sales) AS total_sales,
    AVG(s.sales) AS avg_sale,
    COUNT(*) AS transaction_count
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0 AND s.campaign IS NOT NULL
GROUP BY d.season, s.campaign
ORDER BY d.season, 
         FIELD(s.campaign, 'Before', 'During', 'After'); 

-- 5. Campaign Sales Summary with Overall Totals
SELECT 
    campaign AS group_name,
    'Campaign' AS level,
    SUM(sales) AS total_sales,
    AVG(sales) AS avg_sale,
    COUNT(*) AS num_transactions,
    MAX(sales) AS max_sale
FROM sales
WHERE sales > 0 AND campaign IS NOT NULL
GROUP BY campaign

UNION ALL

SELECT 
    'ALL' AS group_name,
    'Overall' AS level,
    SUM(sales) AS total_sales,
    AVG(sales) AS avg_sale,
    COUNT(*) AS num_transactions,
    MAX(sales) AS max_sale
FROM sales
WHERE sales > 0 AND campaign IS NOT NULL

ORDER BY level DESC, group_name;

-- 6. Best Day of Each Campaign
SELECT 
    s.campaign,
    s.date,
    d.weekday,
    SUM(s.sales) AS daily_total
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0
GROUP BY s.campaign, s.date, d.weekday
HAVING SUM(s.sales) = (
    SELECT MAX(daily_total)
    FROM (
        SELECT 
            campaign,
            date,
            SUM(sales) AS daily_total
        FROM sales
        WHERE sales > 0
        GROUP BY campaign, date
    ) AS daily
    WHERE daily.campaign = s.campaign
)
ORDER BY s.campaign;

-- 7. Exchange Rate Summary by Currency
SELECT 
    'USD_CAD' AS currency,
    MIN(usd_cad) AS min_rate,
    MAX(usd_cad) AS max_rate,
    AVG(usd_cad) AS avg_rate
FROM usd_rates
WHERE usd_cad > 0

UNION ALL

SELECT 
    'USD_EUR',
    MIN(usd_eur),
    MAX(usd_eur),
    AVG(usd_eur)
FROM usd_rates
WHERE usd_eur > 0

UNION ALL

SELECT 
    'USD_GBP',
    MIN(usd_gbp),
    MAX(usd_gbp),
    AVG(usd_gbp)
FROM usd_rates
WHERE usd_gbp > 0

UNION ALL

SELECT 
    'USD_MXN',
    MIN(usd_mxn),
    MAX(usd_mxn),
    AVG(usd_mxn)
FROM usd_rates
WHERE usd_mxn > 0

UNION ALL

SELECT 
    'ALL CURRENCIES' AS currency,
    MIN(min_rate) AS overall_min,
    MAX(max_rate) AS overall_max,
    AVG(avg_rate) AS overall_avg
FROM (
    SELECT 
        MIN(usd_cad) AS min_rate,
        MAX(usd_cad) AS max_rate,
        AVG(usd_cad) AS avg_rate
    FROM usd_rates
    UNION ALL
    SELECT MIN(usd_eur), MAX(usd_eur), AVG(usd_eur) FROM usd_rates
    UNION ALL
    SELECT MIN(usd_gbp), MAX(usd_gbp), AVG(usd_gbp) FROM usd_rates
    UNION ALL
    SELECT MIN(usd_mxn), MAX(usd_mxn), AVG(usd_mxn) FROM usd_rates
) AS all_currencies;

-- 8. Per-weekday and per-campaign aggregates
SELECT 
    d.weekday,
    s.campaign,
    SUM(s.sales) AS total_sales,
    AVG(s.sales) AS avg_sale,
    COUNT(*) AS transaction_count
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0 AND s.campaign IS NOT NULL
GROUP BY d.weekday, s.campaign

UNION ALL

SELECT 
    'ALL' AS weekday,
    'ALL' AS campaign,
    SUM(s.sales) AS total_sales,
    AVG(s.sales) AS avg_sale,
    COUNT(*) AS transaction_count
FROM sales s
WHERE s.sales > 0 AND s.campaign IS NOT NULL

ORDER BY 
    FIELD(weekday, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday', 'ALL'),
    campaign;

-- 9. Daily Sales with Exchange Rates
SELECT 
    s.date,
    d.weekday,
    s.brand,
    s.sales AS sales_usd,
    r.usd_eur,
    s.sales * r.usd_eur AS sales_eur,
    s.campaign
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
JOIN usd_rates r ON s.date_id = r.date_id   -- correct (both reference date_dim.id)
WHERE s.sales > 0
ORDER BY s.date DESC, s.brand;

-- 10. Rolling 7-Day Average Sales
WITH daily_sales AS (
    SELECT 
        s.date,
        SUM(s.sales) AS total_sales
    FROM sales s
    WHERE s.sales > 0
    GROUP BY s.date
)
SELECT 
    ds.date,
    d.weekday,
    ds.total_sales,
    AVG(ds.total_sales) OVER (ORDER BY ds.date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7day_avg,
    ds.total_sales - AVG(ds.total_sales) OVER (ORDER BY ds.date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS diff_from_avg
FROM daily_sales ds
JOIN date_dim d ON ds.date = d.date   -- correct (using actual date column)
ORDER BY ds.date;

-- 11. Per-Brand Weekly Performance with Change
SELECT 
    s.brand,
    d.week_number,
    MIN(d.date) AS week_start,
    SUM(s.sales) AS brand_total,
    AVG(s.sales) AS brand_avg,
    COUNT(*) AS brand_transactions,
    LAG(SUM(s.sales)) OVER (PARTITION BY s.brand ORDER BY d.week_number) AS prev_week_total,
    SUM(s.sales) - LAG(SUM(s.sales)) OVER (PARTITION BY s.brand ORDER BY d.week_number) AS total_change,
    (SUM(s.sales) - LAG(SUM(s.sales)) OVER (PARTITION BY s.brand ORDER BY d.week_number)) 
        / NULLIF(LAG(SUM(s.sales)) OVER (PARTITION BY s.brand ORDER BY d.week_number), 0) * 100 AS pct_change
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0
GROUP BY s.brand, d.week_number
ORDER BY s.brand, d.week_number;

-- 12. Identify Weeks with Highest/Lowest Growth
WITH weekly_sales AS (
    SELECT 
        d.week_number,
        MIN(d.date) AS week_start,
        SUM(s.sales) AS total_sales
    FROM sales s
    JOIN date_dim d ON s.date_id = d.id   -- changed
    WHERE s.sales > 0
    GROUP BY d.week_number
),
weekly_with_prev AS (
    SELECT 
        week_number,
        week_start,
        total_sales,
        LAG(total_sales) OVER (ORDER BY week_number) AS prev_sales,
        total_sales - LAG(total_sales) OVER (ORDER BY week_number) AS change_abs,
        (total_sales - LAG(total_sales) OVER (ORDER BY week_number)) / NULLIF(LAG(total_sales) OVER (ORDER BY week_number), 0) * 100 AS change_pct
    FROM weekly_sales
)
SELECT 
    week_number,
    week_start,
    total_sales,
    prev_sales,
    change_abs,
    change_pct
FROM weekly_with_prev
WHERE change_pct IS NOT NULL
ORDER BY change_pct DESC
LIMIT 5;

-- 13. Compare Campaign vs. Non-Campaign by Quarter
SELECT 
    d.quarter,
    s.campaign,
    SUM(s.sales) AS total_sales,
    AVG(s.sales) AS avg_sale,
    COUNT(*) AS num_sales
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0 AND s.campaign IS NOT NULL
GROUP BY d.quarter, s.campaign
ORDER BY d.quarter, s.campaign;

-- 14. Holiday vs. Non-Holiday Sales Comparison
SELECT 
    d.is_holiday,
    COUNT(DISTINCT s.date) AS num_days,
    SUM(s.sales) AS total_sales,
    AVG(s.sales) AS avg_sale_per_transaction,
    COUNT(*) AS total_transactions,
    SUM(s.sales) / COUNT(DISTINCT s.date) AS avg_daily_sales
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0
GROUP BY d.is_holiday;

SELECT 
    d.holiday_name,
    d.date,
    SUM(s.sales) AS daily_total,
    AVG(s.sales) AS avg_sale
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0 AND d.is_holiday = 1
GROUP BY d.holiday_name, d.date
ORDER BY d.date;

-- 15. Cumulative Sales Over Time with Campaign Milestones
SELECT 
    s.date,
    d.weekday,
    s.campaign,
    SUM(s.sales) AS daily_sales,
    SUM(SUM(s.sales)) OVER (ORDER BY s.date) AS cumulative_sales
FROM sales s
JOIN date_dim d ON s.date_id = d.id   -- changed
WHERE s.sales > 0
GROUP BY s.date, d.weekday, s.campaign
ORDER BY s.date;