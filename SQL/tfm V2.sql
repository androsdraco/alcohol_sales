CREATE DATABASE IF NOT EXISTS TFMsql2026;
USE TFMsql2026;
SET GLOBAL local_infile = 1;

-- 1. Create tables 

CREATE TABLE sales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    brand VARCHAR(100),
    sales DECIMAL(10,2),
    campaign VARCHAR(50),
    date_id INT
);

CREATE TABLE usd_rates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    usd_cad DECIMAL(10,6),
    usd_eur DECIMAL(10,6),
    usd_gbp DECIMAL(10,6),
    usd_mxn DECIMAL(10,6),
    UNIQUE KEY unique_date (date),
    date_id INT
);

CREATE TABLE date_dim (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    month_name VARCHAR(20),
    day INT,
    weekday VARCHAR(20),
    week_number INT,
    quarter INT,
    holiday_name VARCHAR(100),
    is_holiday BOOLEAN,
    season VARCHAR(20),
    date_id INT
);

-- 2. Load data 

LOAD DATA LOCAL INFILE 'C:/Users/Laptop/OneDrive/Documents/TFM/exports/Sales_table.csv'
INTO TABLE sales
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(@row_index, @date, brand, sales, campaign, date_id)
SET date = STR_TO_DATE(@date, '%Y-%m-%d');

LOAD DATA LOCAL INFILE 'C:/Users/Laptop/OneDrive/Documents/TFM/exports/currency_table.csv'
INTO TABLE usd_rates
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(@date, usd_cad, usd_eur, usd_gbp, usd_mxn, date_id)   -- added date_key
SET date = STR_TO_DATE(@date, '%m/%d/%Y');

LOAD DATA LOCAL INFILE 'C:/Users/Laptop/OneDrive/Documents/TFM/exports/date_table.csv'
INTO TABLE date_dim
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(@row_num, date, month_name, day, weekday, week_number, quarter, @holiday_name, @is_holiday, season, date_id)
SET 
    holiday_name = NULLIF(@holiday_name, ''),
    is_holiday = CASE WHEN @is_holiday = 'True' THEN 1 ELSE 0 END;