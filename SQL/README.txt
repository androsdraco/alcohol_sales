Para correr archivo local favor de realizar lo siguiente en MYSQL Workbench

ir a Database > Connect to database > "Elegir database donde se correra el database" > Advanced > Others > Pegar: OPT_LOCAL_INFILE=1

descargar los csv y utilizar el PATH en el archivo "TFM V2" segun corresponda

currency_table.csv --> LOAD DATA LOCAL INFILE "PATH" INTO TABLE usd_rates

Sales_table.csv --> LOAD DATA LOCAL INFILE "PATH" INTO TABLE sales

date_table.csv --> LOAD DATA LOCAL INFILE "PATH" INTO TABLE date_dim

