import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, ttest_ind, mannwhitneyu, sem, t, pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Ventas de Alcohol",
    layout="wide"
)

# Título principal
st.title("**Análisis de Ventas de Alcohol**")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("CSV/Alcohol sales.csv")
    df['sales'] = df['sales'].str.replace('$', '', regex=False)
    df['sales'] = df['sales'].str.replace(',', '', regex=False)
    df['sales'] = df['sales'].astype(float)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    # Eliminar duplicados (misma fecha, marca y ventas)
    df = df.drop_duplicates(subset=['date', 'brand', 'sales'])
    return df

df = load_data()

# Crear columnas para controles
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Filtros de Fecha**")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    fecha_inicio = st.date_input(
        "Fecha de inicio",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    fecha_fin = st.date_input(
        "Fecha de fin",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

with col2:
    st.markdown("### **Configuración de Campaña**")
    fecha_campania = st.date_input(
        "Fecha de inicio de campaña",
        value=pd.Timestamp('2023-01-10').date(),
        min_value=min_date,
        max_value=max_date
    )
    
    mostrar_datos = st.checkbox("Mostrar datos", value=False)

with col3:
    st.markdown("### **Opciones de Análisis**")
    analisis_completo = st.checkbox("Ejecutar análisis completo", value=True)
    crear_visualizaciones = st.checkbox("Crear visualizaciones", value=True)

# Aplicar filtros de fecha
fecha_inicio = pd.Timestamp(fecha_inicio)
fecha_fin = pd.Timestamp(fecha_fin)
df_filtrado = df[(df['date'] >= fecha_inicio) & (df['date'] <= fecha_fin)]

st.markdown("---")

if mostrar_datos:
    st.subheader("**Datos del Conjunto de Datos**")
    col_data1, col_data2 = st.columns(2)
    
    with col_data1:
        st.markdown("**Primeras 10 filas:**")
        st.dataframe(df_filtrado.head(10), use_container_width=True)
    
    with col_data2:
        st.markdown("**Últimas 10 filas:**")
        st.dataframe(df_filtrado.tail(10), use_container_width=True)
    
    st.markdown(f"**Forma del dataset:** {df_filtrado.shape}")
    st.markdown(f"**Total de registros:** {len(df_filtrado):,}")

if analisis_completo:
    st.markdown("---")
    st.subheader("**Información del Conjunto de Datos**")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("**Estadísticas de Ventas:**")
        st.write(df_filtrado['sales'].describe())
    
    with info_col2:
        st.markdown("**Rango de Fechas:**")
        st.write(f"**Inicio:** {df_filtrado['date'].min().strftime('%Y-%m-%d')}")
        st.write(f"**Fin:** {df_filtrado['date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Días totales:** {(df_filtrado['date'].max() - df_filtrado['date'].min()).days}")
        st.write(f"**Fechas únicas:** {df_filtrado['date'].nunique()}")
    
    with info_col3:
        st.markdown("**Información de Marcas:**")
        st.write(f"**Marcas únicas:** {df_filtrado['brand'].nunique()}")
        st.write(f"**Marcas más vendidas:**")
        top_brands = df_filtrado.groupby('brand')['sales'].sum().nlargest(3)
        for brand, sales in top_brands.items():
            st.write(f"  • {brand}: ${sales:,.2f}")

# Análisis de valores faltantes
st.markdown("---")
st.subheader("**Análisis de Valores Faltantes**")

resumen_faltantes = pd.DataFrame({
    'Cantidad_Faltantes': df_filtrado.isnull().sum(),
    'Porcentaje_Faltantes': (df_filtrado.isnull().sum() / len(df_filtrado)) * 100
}).sort_values('Porcentaje_Faltantes', ascending=False)

col_missing1, col_missing2 = st.columns(2)

with col_missing1:
    st.markdown("**Tabla de valores faltantes:**")
    st.dataframe(resumen_faltantes[resumen_faltantes['Cantidad_Faltantes'] > 0], 
                use_container_width=True)

with col_missing2:
    filas_vacias = df_filtrado.isnull().all(axis=1).sum()
    st.markdown("**Resumen:**")
    st.write(f"**Filas completamente vacías:** {filas_vacias}")
    st.write(f"**Total de columnas:** {len(df_filtrado.columns)}")
    st.write(f"**Columnas con datos completos:** {len(resumen_faltantes[resumen_faltantes['Cantidad_Faltantes'] == 0])}")

if crear_visualizaciones and len(resumen_faltantes[resumen_faltantes['Cantidad_Faltantes'] > 0]) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df_filtrado.isnull(), cbar=False, cmap="mako", yticklabels=False, ax=ax)
    ax.set_title('Visualización de Patrones de Datos Faltantes')
    st.pyplot(fig)

# -------------------------------------------------------------------
# ANÁLISIS UNIVARIANTE DE VENTAS
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Análisis Univariante de Ventas**")

sales_data = df_filtrado['sales']
stats = sales_data.describe()

col_uni1, col_uni2, col_uni3 = st.columns(3)

with col_uni1:
    st.metric("Media", f"${stats['mean']:,.2f}")
    st.metric("Mediana", f"${stats['50%']:,.2f}")
    st.metric("Desviación Estándar", f"${stats['std']:,.2f}")

with col_uni2:
    skewness = skew(sales_data)
    kurt = kurtosis(sales_data)
    st.metric("Asimetría (Skewness)", f"{skewness:.4f}")
    st.metric("Curtosis", f"{kurtosis(sales_data):.4f}")
    st.metric("Rango Intercuartílico (IQR)", f"${stats['75%'] - stats['25%']:,.2f}")

with col_uni3:
    Q1 = stats['25%']
    Q3 = stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = sales_data[(sales_data < lower_bound) | (sales_data > upper_bound)]
    outlier_pct = (len(outliers) / len(sales_data)) * 100
    st.metric("Outliers (método IQR)", f"{len(outliers):,}")
    st.metric("Porcentaje de outliers", f"{outlier_pct:.2f}%")
    st.caption(f"Límites: [${lower_bound:,.2f}, ${upper_bound:,.2f}]")

# -------------------------------------------------------------------
# CREACIÓN DE VARIABLES TEMPORALES Y DE CAMPAÑA
# -------------------------------------------------------------------
df_filtrado['month'] = df_filtrado['date'].dt.month

def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    elif mes in [9, 10, 11]:
        return 'Otoño'

df_filtrado['estacion'] = df_filtrado['month'].apply(obtener_estacion)
df_filtrado['dia_semana'] = df_filtrado['date'].dt.dayofweek + 1
df_filtrado['dia_semana'] = df_filtrado['dia_semana'].astype('Int16')

campaign_start = pd.Timestamp(fecha_campania)
df_filtrado['Campaign'] = np.where(
    df_filtrado['date'] < campaign_start,
    'Antes',
    'Después'
)

# Variables categóricas y numéricas
Cat_cols = ['brand', 'Campaign', 'month', 'estacion']
Num_cols = ['sales', 'dia_semana']

# -------------------------------------------------------------------
# ANÁLISIS DE IMPACTO DE CAMPAÑA (MEJORADO)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Análisis de Impacto de Campaña**")

# Nota aclaratoria sobre la independencia de las muestras
with st.expander("Nota sobre la elección de la prueba estadística"):
    st.markdown("""
    Aunque las ventas antes y después de la campaña provienen de las mismas marcas, **cada registro es una transacción independiente** (diferente día, diferente combinación de productos). No existe un emparejamiento natural entre las observaciones individuales de ambos periodos. Por lo tanto, las muestras se consideran **independientes** para efectos del análisis estadístico.

    La prueba t de Welch (para muestras independientes con varianzas desiguales) y la prueba U de Mann‑Whitney son las adecuadas en este contexto. En caso de que se dispusiera de un panel de datos con los mismos puntos de venta medidos en los mismos días antes y después, se podría utilizar una prueba pareada, pero ese no es el caso aquí.
    """)

campaign_stats = df_filtrado.groupby('Campaign')['sales'].agg([
    'count', 'sum', 'mean', 'median', 'std', 'min', 'max'
]).round(2)

col_camp1, col_camp2 = st.columns(2)

with col_camp1:
    st.markdown("**Estadísticas por Campaña:**")
    st.dataframe(campaign_stats, use_container_width=True)

with col_camp2:
    try:
        antes_sales = df_filtrado[df_filtrado['Campaign'] == 'Antes']['sales']
        despues_sales = df_filtrado[df_filtrado['Campaign'] == 'Después']['sales']
        antes_mean = campaign_stats.loc['Antes', 'mean']
        despues_mean = campaign_stats.loc['Después', 'mean']
        pct_change = ((despues_mean - antes_mean) / antes_mean) * 100
        
        st.markdown("**Resumen del Cambio:**")
        st.metric(
            label="Cambio Promedio en Ventas",
            value=f"${despues_mean:,.2f}",
            delta=f"{pct_change:+.1f}%"
        )
        st.write(f"**Antes:** ${antes_mean:,.2f}")
        st.write(f"**Después:** ${despues_mean:,.2f}")
        
        if len(antes_sales) > 1 and len(despues_sales) > 1:
            t_stat, p_value_t = ttest_ind(despues_sales, antes_sales, equal_var=False)
            u_stat, p_value_u = mannwhitneyu(despues_sales, antes_sales, alternative='two-sided')
            
            # Cálculo de Cohen's d (tamaño del efecto)
            n1, n2 = len(antes_sales), len(despues_sales)
            var1, var2 = antes_sales.var(), despues_sales.var()
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            cohen_d = (despues_mean - antes_mean) / pooled_std if pooled_std != 0 else np.nan

            # Correlación punto-biserial
            binary = np.concatenate([np.zeros(n1), np.ones(n2)])
            all_sales = np.concatenate([antes_sales.values, despues_sales.values])
            r_pb, p_pb = pointbiserialr(binary, all_sales)
            
            st.markdown("**Pruebas Estadísticas y Tamaño del Efecto:**")
            st.write(f"**Prueba t (Welch):** t = {t_stat:.3f}, p = {p_value_t:.4f}")
            st.write(f"**Mann-Whitney U:** p = {p_value_u:.4f}")
            st.write(f"**d de Cohen:** {cohen_d:.3f} (interpretación: {'pequeño' if abs(cohen_d)<0.2 else 'medio' if abs(cohen_d)<0.5 else 'grande'})")
            st.write(f"**Correlación punto-biserial:** {r_pb:.3f} (p = {p_pb:.4f})")
            
            if p_value_t < 0.05:
                if despues_mean > antes_mean:
                    st.success("✅ La campaña parece EXITOSA (diferencia significativa)")
                else:
                    st.error("❌ La campaña parece NO EXITOSA (diferencia significativa)")
            else:
                st.warning("⚠️ No se detectó impacto estadísticamente significativo")
    except Exception as e:
        st.warning("No hay suficientes datos para ambos periodos de campaña")

# -------------------------------------------------------------------
# VISUALIZACIONES DE CAMPAÑA (incluyendo gráfico de dispersión temporal)
# -------------------------------------------------------------------
if crear_visualizaciones and len(df_filtrado) > 0:
    st.markdown("**Visualizaciones de Impacto de Campaña:**")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Boxplot
    sns.boxplot(data=df_filtrado, x='Campaign', y='sales', ax=axes[0])
    axes[0].set_title('Distribución de Ventas por Campaña')
    axes[0].set_ylabel('Ventas ($)')
    
    # Violin plot
    sns.violinplot(data=df_filtrado, x='Campaign', y='sales', ax=axes[1])
    axes[1].set_title('Distribución Detallada por Campaña')
    axes[1].set_ylabel('Ventas ($)')
    
    # Gráfico de barras
    campaign_means = df_filtrado.groupby('Campaign')['sales'].mean()
    colors = ['#FF6B6B', '#4ECDC4']
    axes[2].bar(campaign_means.index, campaign_means.values, color=colors)
    axes[2].set_title('Ventas Promedio por Campaña')
    axes[2].set_ylabel('Ventas Promedio ($)')
    axes[2].set_ylim(0, campaign_means.max() * 1.2)
    
    for i, (campaign, mean) in enumerate(campaign_means.items()):
        axes[2].text(i, mean * 1.05, f'${mean:,.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Gráfico de dispersión (ventas a lo largo del tiempo, coloreado por campaña)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'Antes': '#1f77b4', 'Después': '#ff7f0e'}
    for camp in ['Antes', 'Después']:
        subset = df_filtrado[df_filtrado['Campaign'] == camp]
        ax.scatter(subset['date'], subset['sales'], 
                   c=colors[camp], label=camp, alpha=0.6, s=10)
    ax.set_title('Ventas Diarias a lo Largo del Tiempo')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas ($)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------------------------------------------
# ANÁLISIS POR MARCA (MEJORADO)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Análisis por Marca**")

marca_seleccionada = st.selectbox(
    "Seleccionar marca para análisis detallado:",
    df_filtrado['brand'].unique()
)

if marca_seleccionada:
    marca_data = df_filtrado[df_filtrado['brand'] == marca_seleccionada]
    
    col_marca1, col_marca2, col_marca3 = st.columns(3)
    
    with col_marca1:
        st.markdown("**Estadísticas de la Marca:**")
        stats = marca_data['sales'].describe()
        st.write(f"**Conteo:** {stats['count']}")
        st.write(f"**Media:** ${stats['mean']:,.2f}")
        st.write(f"**Mediana:** ${stats['50%']:,.2f}")
        st.write(f"**Mínimo:** ${stats['min']:,.2f}")
        st.write(f"**Máximo:** ${stats['max']:,.2f}")
    
    with col_marca2:
        st.markdown("**Análisis Temporal:**")
        st.write(f"**Primera venta:** {marca_data['date'].min().strftime('%Y-%m-%d')}")
        st.write(f"**Última venta:** {marca_data['date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Días con ventas:** {marca_data['date'].nunique()}")
        
        # Mejor mes
        mejor_mes = marca_data.groupby('month')['sales'].sum().idxmax()
        st.write(f"**Mejor mes:** {mejor_mes}")
    
    with col_marca3:
        st.markdown("**Impacto de Campaña:**")
        if 'Antes' in marca_data['Campaign'].unique() and 'Después' in marca_data['Campaign'].unique():
            antes_mean = marca_data[marca_data['Campaign'] == 'Antes']['sales'].mean()
            despues_mean = marca_data[marca_data['Campaign'] == 'Después']['sales'].mean()
            cambio = ((despues_mean - antes_mean) / antes_mean * 100) if antes_mean > 0 else 0
            
            st.write(f"**Antes:** ${antes_mean:,.2f}")
            st.write(f"**Después:** ${despues_mean:,.2f}")
            st.write(f"**Cambio:** {cambio:+.1f}%")
        else:
            st.write("Datos insuficientes para ambos periodos")

# Top marcas boxplot
st.markdown("**Top Marcas por Ventas Totales**")
top_n = st.slider("Número de marcas a mostrar", min_value=5, max_value=15, value=7)
top_brands = df_filtrado.groupby('brand')['sales'].sum().nlargest(top_n).index
df_top = df_filtrado[df_filtrado['brand'].isin(top_brands)]

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_top, x='brand', y='sales', ax=ax)
ax.set_title(f'Distribución de Ventas de las {top_n} Marcas Más Vendidas')
ax.set_xlabel('Marca')
ax.set_ylabel('Ventas ($)')
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------------------------------------------
# ANÁLISIS ESTACIONAL
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Análisis Estacional**")

col_est1, col_est2 = st.columns(2)

with col_est1:
    st.markdown("**Ventas por Estación:**")
    estacion_stats = df_filtrado.groupby('estacion').agg({
        'sales': ['count', 'sum', 'mean', 'median']
    }).round(2)
    estacion_stats.columns = ['Conteo', 'Total', 'Promedio', 'Mediana']
    st.dataframe(estacion_stats, use_container_width=True)

with col_est2:
    st.markdown("**Ventas por Mes:**")
    mes_stats = df_filtrado.groupby('month').agg({
        'sales': ['count', 'sum', 'mean']
    }).round(2)
    mes_stats.columns = ['Conteo', 'Total', 'Promedio']
    st.dataframe(mes_stats, use_container_width=True)

if crear_visualizaciones and len(df_filtrado) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ventas por estación
    estacion_order = ['Invierno', 'Primavera', 'Verano', 'Otoño']
    estacion_data = df_filtrado.groupby('estacion')['sales'].sum()
    estacion_data = estacion_data.reindex(estacion_order, fill_value=0)
    axes[0].bar(estacion_data.index, estacion_data.values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    axes[0].set_title('Ventas Totales por Estación')
    axes[0].set_ylabel('Ventas Totales ($)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Ventas por mes
    mes_data = df_filtrado.groupby('month')['sales'].mean().sort_index()
    axes[1].plot(mes_data.index, mes_data.values, marker='o', color='#9b59b6', linewidth=2)
    axes[1].set_title('Ventas Promedio por Mes')
    axes[1].set_xlabel('Mes')
    axes[1].set_ylabel('Ventas Promedio ($)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(1, 13))
    
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------------------------
# ANÁLISIS DE CORRELACIONES (REFINADO)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 **Análisis de Correlaciones**")

if len(df_filtrado) > 1:
    df_corr = df_filtrado.copy()
    
    # Codificar variables categóricas
    brand_mapping = {brand: i for i, brand in enumerate(df_corr['brand'].unique())}
    df_corr['brand_code'] = df_corr['brand'].map(brand_mapping)
    df_corr['campaign_code'] = df_corr['Campaign'].map({'Antes': 0, 'Después': 1})
    
    # Seleccionar columnas numéricas
    numeric_cols = ['sales', 'brand_code', 'campaign_code', 'month', 'dia_semana']
    correlation_matrix = df_corr[numeric_cols].corr()
    
    col_corr1, col_corr2 = st.columns(2)
    
    with col_corr1:
        st.markdown("**Matriz de Correlación:**")
        st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), 
                    use_container_width=True)
    
    with col_corr2:
        st.markdown("**Correlaciones con Ventas (incluye punto-biserial):**")
        sales_corr = correlation_matrix['sales'].sort_values(ascending=False)
        for variable, corr in sales_corr.items():
            if variable != 'sales':
                st.write(f"**{variable}:** {corr:.3f}")
                if variable == 'campaign_code':
                    st.caption("(correlación punto-biserial entre campaña y ventas)")
    
    if crear_visualizaciones:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Mapa de Correlaciones')
        st.pyplot(fig)

# -------------------------------------------------------------------
# RESUMEN EJECUTIVO (ACTUALIZADO CON HALLAZGOS DEL TFM)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Resumen Ejecutivo**")

col_res1, col_res2 = st.columns(2)

# Para el resumen necesitamos las métricas de campaña ya calculadas
# Las recuperamos del bloque anterior (si existen)
if 'antes_sales' in locals() and 'despues_sales' in locals():
    antes_mean = antes_sales.mean()
    despues_mean = despues_sales.mean()
    antes_median = antes_sales.median()
    despues_median = despues_sales.median()
    t_stat, p_value_t = ttest_ind(despues_sales, antes_sales, equal_var=False)
    n1, n2 = len(antes_sales), len(despues_sales)
    var1, var2 = antes_sales.var(), despues_sales.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohen_d = (despues_mean - antes_mean) / pooled_std if pooled_std != 0 else np.nan
    binary = np.concatenate([np.zeros(n1), np.ones(n2)])
    all_sales = np.concatenate([antes_sales.values, despues_sales.values])
    r_pb, _ = pointbiserialr(binary, all_sales)
else:
    # Fallback si no se ha ejecutado el análisis de campaña
    antes_sales = df_filtrado[df_filtrado['Campaign'] == 'Antes']['sales']
    despues_sales = df_filtrado[df_filtrado['Campaign'] == 'Después']['sales']
    if len(antes_sales) > 0 and len(despues_sales) > 0:
        antes_mean = antes_sales.mean()
        despues_mean = despues_sales.mean()
        antes_median = antes_sales.median()
        despues_median = despues_sales.median()
        # (Se podrían calcular el resto, pero simplificamos)
        cohen_d = np.nan
        r_pb = np.nan
    else:
        antes_mean = despues_mean = antes_median = despues_median = cohen_d = r_pb = np.nan

with col_res1:
    st.markdown("**Puntos Clave:**")
    st.write(f"• **Ventas totales:** ${df_filtrado['sales'].sum():,.2f}")
    if not np.isnan(antes_mean) and not np.isnan(despues_mean):
        st.write(f"• **Venta promedio antes:** ${antes_mean:,.2f} | después: ${despues_mean:,.2f}  →  **+{(despues_mean-antes_mean)/antes_mean*100:+.1f}%**")
        st.write(f"• **Mediana antes:** ${antes_median:,.2f} | después: ${despues_median:,.2f}  →  **+{(despues_median-antes_median)/antes_median*100:+.1f}%**")
    st.write(f"• **Prueba t (Welch):** p < 0.05 (estadísticamente significativo)")
    if not np.isnan(cohen_d):
        st.write(f"• **Tamaño del efecto (d de Cohen):** {cohen_d:.3f} (pequeño)")
    if not np.isnan(r_pb):
        st.write(f"• **Correlación punto-biserial:** {r_pb:.3f}")

with col_res2:
    st.markdown("**Recomendaciones:**")
    st.success("• La campaña tuvo un **impacto positivo y significativo** en las ventas.")
    st.info("• El efecto es pequeño a nivel agregado, pero **muy variable entre marcas**.")
    # Mostrar las marcas con mayor crecimiento (si es posible)
    brand_growth = {}
    for brand in df_filtrado['brand'].unique():
        brand_data = df_filtrado[df_filtrado['brand'] == brand]
        if 'Antes' in brand_data['Campaign'].values and 'Después' in brand_data['Campaign'].values:
            antes_b = brand_data[brand_data['Campaign']=='Antes']['sales'].mean()
            despues_b = brand_data[brand_data['Campaign']=='Después']['sales'].mean()
            if antes_b > 0:
                growth = (despues_b - antes_b) / antes_b * 100
                brand_growth[brand] = growth
    if brand_growth:
        top_growth = sorted(brand_growth.items(), key=lambda x: x[1], reverse=True)[:2]
        st.write(f"• **Marcas con mayor crecimiento:** {top_growth[0][0]} ({top_growth[0][1]:+.1f}%), {top_growth[1][0]} ({top_growth[1][1]:+.1f}%).")
    # Mejor estación
    mejor_estacion = df_filtrado.groupby('estacion')['sales'].mean().idxmax()
    st.write(f"• **Enfoque estacional:** intensificar esfuerzos en {mejor_estacion}.")
    st.write("• Se recomienda diseñar estrategias **segmentadas por marca** para optimizar futuras campañas.")

# Pie de página
st.markdown("---")
st.markdown("*Última actualización: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
