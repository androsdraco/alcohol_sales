import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, pointbiserialr, skew, kurtosis, sem, t, norm
import warnings
import re
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Análisis Estadístico de Ventas",
    layout="wide"
)

# Título principal
st.title("Análisis Estadístico Completo de Ventas")

# ==================== CONFIGURACIÓN ====================
st.header("Configuración del Análisis")

# Cargar archivo desde ubicación fija
file_path = r"CSV/Alcohol sales.csv"

# Verificar si el archivo existe
if not os.path.exists(file_path):
    st.error(f"No se encontró el archivo en la ubicación: {file_path}")
    st.write("Por favor, asegúrate de que el archivo 'Alcohol sales.csv' esté en la carpeta CSV")
    st.stop()

st.write(f"Archivo cargado: {file_path}")

# Fecha de inicio de campaña
campaign_date = st.date_input(
    "Fecha de Inicio de Campaña",
    value=pd.Timestamp('2023-01-10'),
    help="Selecciona la fecha cuando comenzó la campaña"
)

# Convertir a Timestamp
campaign_date = pd.Timestamp(campaign_date)

# ==================== FUNCIONES AUXILIARES ====================
def clean_sales_value(value):
    """Limpia y convierte valores de venta a float"""
    if pd.isna(value):
        return np.nan
    
    value_str = str(value)
    value_str = re.sub(r'[$,€£¥\s]', '', value_str)
    
    if ',' in value_str and '.' in value_str:
        value_str = value_str.replace(',', '')
    elif ',' in value_str and '.' not in value_str:
        parts = value_str.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            value_str = value_str.replace(',', '.')
        else:
            value_str = value_str.replace(',', '')
    
    try:
        return float(value_str)
    except:
        return np.nan

def confidence_interval(data, confidence=0.95):
    """Calcula intervalo de confianza para la media"""
    n = len(data)
    if n < 2:
        return np.mean(data), np.nan, np.nan
    
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def interpret_cohens_d(d):
    """Interpreta el tamaño del efecto de Cohen's d"""
    if abs(d) < 0.2:
        return "Efecto muy pequeño"
    elif abs(d) < 0.5:
        return "Efecto pequeño"
    elif abs(d) < 0.8:
        return "Efecto moderado"
    else:
        return "Efecto grande"

# ==================== CARGA Y PROCESAMIENTO DE DATOS ====================
@st.cache_data
def load_and_process_data(file_path, campaign_date):
    """Carga y procesa los datos"""
    try:
        # Leer archivo
        df = pd.read_csv(file_path)
        
        # Buscar columna de fecha
        date_cols = ['date', 'Date', 'DATE', 'fecha', 'Fecha']
        for date_col in date_cols:
            if date_col in df.columns:
                try:
                    df['date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                    if df['date'].isna().any():
                        df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%y', errors='coerce')
                    break
                except:
                    continue
        
        # Buscar columna de ventas
        sales_cols = ['sales', 'Sales', 'ventas', 'Ventas']
        for sales_col in sales_cols:
            if sales_col in df.columns:
                df['sales'] = df[sales_col]
                break
        
        # Limpiar datos de ventas
        df['sales'] = df['sales'].apply(clean_sales_value)
        df = df.dropna(subset=['sales'])
        df = df[df['sales'] >= 0]
        
        # Buscar columna de marca
        brand_cols = ['brand', 'Brand', 'marca', 'Marca', 'producto', 'Producto']
        for brand_col in brand_cols:
            if brand_col in df.columns:
                df['brand'] = df[brand_col].astype(str)
                break
        
        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=['date'])
        
        # Crear variable Campaign (Before/After)
        df['Campaign'] = 'Before'
        df.loc[df['date'] >= campaign_date, 'Campaign'] = 'After'
        
        # Crear variables temporales adicionales
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        return df
        
    except Exception as e:
        st.error(f"Error cargando archivo: {str(e)}")
        return None

# Cargar datos
with st.spinner("Cargando y procesando datos..."):
    df = load_and_process_data(file_path, campaign_date)

if df is None:
    st.stop()

# ==================== INFORMACIÓN DEL DATASET ====================
st.markdown("---")
st.header("Información del Dataset")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Forma del Dataset", f"{df.shape[0]} filas × {df.shape[1]} columnas")
    st.metric("Registros totales", f"{len(df):,}")
    st.metric("Marcas únicas", f"{df['brand'].nunique()}")

with col_info2:
    st.metric("Rango de fechas", 
              f"{df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}")
    st.metric("Días totales", f"{(df['date'].max() - df['date'].min()).days}")
    st.metric("Fechas únicas", f"{df['date'].nunique()}")

with col_info3:
    # Estadísticas de ventas
    st.metric("Ventas totales", f"${df['sales'].sum():,.0f}")
    st.metric("Venta promedio", f"${df['sales'].mean():.2f}")
    st.metric("Mediana de ventas", f"${df['sales'].median():.2f}")

# Estadísticas descriptivas detalladas
st.subheader("Estadísticas Descriptivas de Ventas")

col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.write("**Medidas de tendencia central:**")
    stats_summary = df['sales'].describe()
    
    for stat, value in stats_summary.items():
        if stat == 'count':
            st.write(f"- {stat.capitalize()}: {value:,.0f}")
        else:
            st.write(f"- {stat.capitalize()}: ${value:,.2f}")
    
    # Medidas de forma
    st.write(f"\n**Medidas de forma:**")
    st.write(f"- Asimetría (skewness): {skew(df['sales'].dropna()):.4f}")
    st.write(f"- Curtosis (kurtosis): {kurtosis(df['sales'].dropna()):.4f}")

with col_stats2:
    # Cálculo de cuartiles y outliers
    Q1 = df['sales'].quantile(0.25)
    Q3 = df['sales'].quantile(0.75)
    IQR = Q3 - Q1
    
    st.write("**Análisis de Cuartiles:**")
    st.write(f"- Q1 (25%): ${Q1:.2f}")
    st.write(f"- Q3 (75%): ${Q3:.2f}")
    st.write(f"- Rango Intercuartílico (IQR): ${IQR:.2f}")
    
    # Límites para outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['sales'] < lower_bound) | (df['sales'] > upper_bound)]
    
    st.write(f"\n**Detección de Outliers (método IQR):**")
    st.write(f"- Límite inferior: ${lower_bound:.2f}")
    st.write(f"- Límite superior: ${upper_bound:.2f}")
    st.write(f"- Outliers potenciales: {len(outliers)} registros ({len(outliers)/len(df)*100:.1f}%)")

# ==================== ANÁLISIS TEMPORAL ====================
st.markdown("---")
st.header("Análisis Temporal de Ventas")

col_time1, col_time2 = st.columns(2)

with col_time1:
    # Estadísticas anuales
    st.subheader("Estadísticas por Año")
    yearly_stats = df.groupby('year')['sales'].agg(['count', 'sum', 'mean', 'median']).round(2)
    st.dataframe(yearly_stats.style.format({
        'count': '{:,.0f}',
        'sum': '${:,.2f}',
        'mean': '${:.2f}',
        'median': '${:.2f}'
    }))

with col_time2:
    # Patrones mensuales
    st.subheader("Patrones Mensuales")
    monthly_avg = df.groupby('month')['sales'].mean().sort_values(ascending=False)
    
    fig_month, ax_month = plt.subplots(figsize=(10, 5))
    bars = ax_month.bar(range(1, 13), monthly_avg.sort_index().values)
    ax_month.set_xlabel('Mes')
    ax_month.set_ylabel('Venta Promedio ($)')
    ax_month.set_title('Venta Promedio por Mes')
    ax_month.set_xticks(range(1, 13))
    ax_month.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, val in zip(bars, monthly_avg.sort_index().values):
        ax_month.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(monthly_avg.values)*0.01,
                     f'${val:.0f}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig_month)

# Patrones por día de la semana
st.subheader("Patrones por Día de la Semana")
dow_avg = df.groupby('day_of_week')['sales'].mean()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_avg = dow_avg.reindex(days_order)

fig_dow, ax_dow = plt.subplots(figsize=(10, 5))
bars = ax_dow.bar(range(len(dow_avg)), dow_avg.values)
ax_dow.set_xlabel('Día de la Semana')
ax_dow.set_ylabel('Venta Promedio ($)')
ax_dow.set_title('Venta Promedio por Día de la Semana')
ax_dow.set_xticks(range(len(dow_avg)))
ax_dow.set_xticklabels(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
ax_dow.grid(True, alpha=0.3, axis='y')

# Añadir valores
for bar, val in zip(bars, dow_avg.values):
    ax_dow.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dow_avg.values)*0.01,
               f'${val:.0f}', ha='center', va='bottom', fontsize=9)

st.pyplot(fig_dow)

# ==================== ANÁLISIS DE HIPÓTESIS MULTIVARIADO ====================
st.markdown("---")
st.header("Pruebas de Hipótesis Multivariadas")

st.subheader("Comparación de Múltiples Variables: Antes vs Después")

# Identificar columnas numéricas
Num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remover columnas que no queremos analizar
exclude_cols = ['year', 'month', 'day', 'is_weekend']
Num_cols = [col for col in Num_cols if col not in exclude_cols]

if 'sales' not in Num_cols and 'sales' in df.columns:
    Num_cols.insert(0, 'sales')

st.write(f"Variables numéricas identificadas: {', '.join(Num_cols)}")
st.write(f"Total de variables a analizar: {len(Num_cols)}")

# Realizar pruebas de hipótesis para cada columna
test_results = []

with st.spinner("Realizando pruebas de hipótesis..."):
    for col in Num_cols:
        # Skip si no es columna numérica
        if col == 'Campaign' or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Separar datos por Campaign
        before_data = df[df['Campaign'] == 'Before'][col].dropna()
        after_data = df[df['Campaign'] == 'After'][col].dropna()
        
        if len(before_data) >= 10 and len(after_data) >= 10:
            # Calcular estadísticas básicas
            before_mean = before_data.mean()
            after_mean = after_data.mean()
            mean_diff = after_mean - before_mean
            
            # Calcular cambio porcentual
            try:
                pct_change = (mean_diff / abs(before_mean)) * 100 if before_mean != 0 else np.nan
            except (TypeError, ZeroDivisionError):
                pct_change = np.nan
            
            # 1. Prueba t de Student
            t_stat, p_val_ttest = ttest_ind(after_data, before_data, equal_var=False)
            
            # 2. Prueba U de Mann-Whitney (no paramétrica)
            u_stat, p_val_mw = mannwhitneyu(after_data, before_data, alternative='two-sided')
            
            # 3. Tamaño del efecto (Cohen's d)
            n1, n2 = len(before_data), len(after_data)
            if n1 > 1 and n2 > 1:
                var1 = before_data.var()
                var2 = after_data.var()
                sd_pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                cohens_d = mean_diff / sd_pooled if sd_pooled != 0 else np.nan
            else:
                cohens_d = np.nan
            
            # 4. Correlación punto-biserial
            try:
                combined_data = df[[col, 'Campaign']].dropna()
                combined_data['Campaign_numeric'] = combined_data['Campaign'].map({'Before': 0, 'After': 1})
                pb_corr, pb_pval = pointbiserialr(combined_data[col], combined_data['Campaign_numeric'])
            except:
                pb_corr, pb_pval = np.nan, np.nan
            
            test_results.append({
                'Variable': col,
                'Media_Antes': before_mean,
                'Media_Despues': after_mean,
                'Diferencia_Media': mean_diff,
                'Cambio_Porcentual': pct_change,
                'Valor_p_TTest': p_val_ttest,
                'Valor_p_MannWhitney': p_val_mw,
                'Correlacion_PuntoBiserial': pb_corr,
                'Cohens_d': cohens_d,
                'Significativo_TTest': p_val_ttest < 0.05,
                'Significativo_MW': p_val_mw < 0.05
            })

if test_results:
    results_df = pd.DataFrame(test_results)
    
    # Ordenar por tamaño del efecto absoluto
    results_df['Tamaño_Efecto_Absoluto'] = results_df['Cohens_d'].abs()
    results_df = results_df.sort_values('Tamaño_Efecto_Absoluto', ascending=False)
    
    # Mostrar resultados en Streamlit
    st.subheader("Resultados de Pruebas de Hipótesis")
    
    # Formatear para mejor visualización
    display_cols = ['Variable', 'Media_Antes', 'Media_Despues', 'Cambio_Porcentual', 
                    'Correlacion_PuntoBiserial', 'Cohens_d', 'Valor_p_TTest', 'Significativo_TTest']
    
    display_df = results_df[display_cols].round({
        'Media_Antes': 2,
        'Media_Despues': 2,
        'Cambio_Porcentual': 1,
        'Correlacion_PuntoBiserial': 3,
        'Cohens_d': 3,
        'Valor_p_TTest': 4
    })
    
    # Añadir símbolos para significancia
    def add_sig_symbol(p_val, sig):
        if sig:
            return f"{p_val:.4f}*"
        return f"{p_val:.4f}"
    
    display_df['Valor_p_TTest'] = [add_sig_symbol(p, s) for p, s in 
                                   zip(results_df['Valor_p_TTest'], results_df['Significativo_TTest'])]
    
    # Mostrar tabla
    st.dataframe(display_df.style.format({
        'Media_Antes': '${:.2f}',
        'Media_Despues': '${:.2f}',
        'Cambio_Porcentual': '{:.1f}%',
        'Correlacion_PuntoBiserial': '{:.3f}',
        'Cohens_d': '{:.3f}'
    }))
    
    # Resumen estadístico
    st.subheader("Resumen Estadístico")
    
    n_significant_ttest = results_df['Significativo_TTest'].sum()
    n_significant_mw = results_df['Significativo_MW'].sum()
    
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.metric("Variables con diferencias significativas (T-Test)", 
                 f"{n_significant_ttest} de {len(results_df)}",
                 f"{n_significant_ttest/len(results_df)*100:.1f}%")
    
    with col_sum2:
        st.metric("Variables con diferencias significativas (Mann-Whitney)", 
                 f"{n_significant_mw} de {len(results_df)}",
                 f"{n_significant_mw/len(results_df)*100:.1f}%")
    
    # Top 3 variables más afectadas
    st.subheader("Top 3 Variables Más Afectadas por la Campaña")
    
    top_3 = results_df.head(3)
    for i, row in top_3.iterrows():
        col_top1, col_top2, col_top3 = st.columns(3)
        
        with col_top1:
            st.metric(f"{row['Variable']} - Media Antes", f"${row['Media_Antes']:.2f}")
        
        with col_top2:
            st.metric(f"{row['Variable']} - Media Después", f"${row['Media_Despues']:.2f}")
        
        with col_top3:
            if pd.notna(row['Cambio_Porcentual']):
                st.metric(f"{row['Variable']} - Cambio", f"{row['Cambio_Porcentual']:+.1f}%")
            else:
                st.metric(f"{row['Variable']} - Tamaño Efecto", f"d = {row['Cohens_d']:.3f}")
    
    # Visualización de resultados significativos
    st.subheader("Visualización de Resultados Significativos")
    
    significant_vars = results_df[results_df['Significativo_TTest']]['Variable'].tolist()
    
    if significant_vars:
        # Mostrar gráficos en pestañas
        tabs = st.tabs(significant_vars[:3])  # Mostrar máximo 3 pestañas
        
        for idx, var in enumerate(significant_vars[:3]):
            with tabs[idx]:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Boxplot
                sns.boxplot(data=df, x='Campaign', y=var, ax=axes[0])
                axes[0].set_title(f'Diagrama de Cajas: {var} por Campaña')
                axes[0].set_ylabel(var)
                axes[0].set_xlabel('Periodo')
                
                # Violin plot
                sns.violinplot(data=df, x='Campaign', y=var, ax=axes[1])
                axes[1].set_title(f'Distribución: {var} por Campaña')
                axes[1].set_ylabel(var)
                axes[1].set_xlabel('Periodo')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Mostrar estadísticas específicas
                var_stats = results_df[results_df['Variable'] == var].iloc[0]
                
                col_var1, col_var2 = st.columns(2)
                
                with col_var1:
                    st.write("**Estadísticas de la Variable:**")
                    st.write(f"- Media Antes: ${var_stats['Media_Antes']:.2f}")
                    st.write(f"- Media Después: ${var_stats['Media_Despues']:.2f}")
                    st.write(f"- Diferencia: ${var_stats['Diferencia_Media']:.2f}")
                    
                    if pd.notna(var_stats['Cambio_Porcentual']):
                        st.write(f"- Cambio: {var_stats['Cambio_Porcentual']:+.1f}%")
                
                with col_var2:
                    st.write("**Resultados Estadísticos:**")
                    st.write(f"- Valor p (T-Test): {var_stats['Valor_p_TTest']:.4f}")
                    st.write(f"- Cohen's d: {var_stats['Cohens_d']:.3f}")
                    st.write(f"- Interpretación tamaño efecto: {interpret_cohens_d(var_stats['Cohens_d'])}")
                    st.write(f"- Correlación punto-biserial: {var_stats['Correlacion_PuntoBiserial']:.3f}")
    else:
        st.info("No se encontraron variables con diferencias estadísticamente significativas")

else:
    st.warning("No se pudieron realizar pruebas de hipótesis - verifique que haya datos suficientes")

# ==================== EXPORTACIÓN DE RESULTADOS ====================
st.markdown("---")
st.header("Exportación de Resultados")

# Crear resumen ejecutivo
summary_data = {
    'Fecha_analisis': [datetime.now().strftime('%Y-%m-%d %H:%M')],
    'Fecha_campaña': [campaign_date.strftime('%Y-%m-%d')],
    'Muestra_total': [len(df)],
    'Antes_campaña': [len(df[df['Campaign'] == 'Before'])],
    'Despues_campaña': [len(df[df['Campaign'] == 'After'])],
    'Variables_analizadas': [len(Num_cols)],
    'Variables_significativas': [n_significant_ttest if 'n_significant_ttest' in locals() else 0],
    'Venta_promedio_total': [f"${df['sales'].mean():.2f}"],
    'Asimetría_ventas': [f"{skew(df['sales'].dropna()):.4f}"],
    'Curtosis_ventas': [f"{kurtosis(df['sales'].dropna()):.4f}"]
}

summary_df = pd.DataFrame(summary_data)

# Botones para descargar diferentes reportes
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    if st.button("Generar Reporte Ejecutivo"):
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Descargar Reporte Ejecutivo (CSV)",
            data=csv,
            file_name=f"reporte_ejecutivo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

with col_exp2:
    if test_results:
        if st.button("Generar Reporte de Pruebas de Hipótesis"):
            results_csv = results_df.to_csv(index=False)
            st.download_button(
                label="Descargar Resultados Pruebas (CSV)",
                data=results_csv,
                file_name=f"resultados_hipotesis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

with col_exp3:
    if st.button("Generar Reporte Completo"):
        # Crear reporte completo combinado
        full_report = {
            'Resumen_Ejecutivo': summary_df,
            'Estadisticas_Descriptivas': df.describe().reset_index(),
            'Resultados_Hipotesis': results_df if test_results else pd.DataFrame()
        }
        
        # Crear Excel con múltiples hojas
        from io import BytesIO
        import openpyxl
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            df.describe().to_excel(writer, sheet_name='Estadisticas')
            if test_results:
                results_df.to_excel(writer, sheet_name='Pruebas_Hipotesis', index=False)
        
        st.download_button(
            label="Descargar Reporte Completo (Excel)",
            data=output.getvalue(),
            file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==================== PIE DE PÁGINA ====================
st.markdown("---")
st.write(f"Análisis estadístico completado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.write("Herramienta de análisis estadístico multivariado - Versión 2.0")
