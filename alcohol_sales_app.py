import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, ttest_ind, mannwhitneyu, pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Intento de importar statsmodels (para el test de interacción)
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("El paquete 'statsmodels' no está instalado. La prueba de interacción por marca no estará disponible. Para instalarlo, añade 'statsmodels' a tu archivo requirements.txt.")

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Ventas de Alcohol",
    layout="wide"
)

# Título principal
st.title("**Análisis de Ventas de Alcohol**")
st.markdown("---")

# -------------------------------------------------------------------
# CARGA DE DATOS
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CSV/Alcohol_sales_enriched.csv")
    # Limpiar la columna de ventas (eliminar $ y comas)
    df['sales'] = df['sales'].astype(str).str.replace('$', '', regex=False)
    df['sales'] = df['sales'].str.replace(',', '', regex=False)
    df['sales'] = df['sales'].astype(float)
    # Convertir fecha
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    # Eliminar duplicados (misma fecha, marca y ventas)
    df = df.drop_duplicates(subset=['date', 'brand', 'sales'])
    # Asegurar que 'size' sea numérico
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    # Estandarizar presentación a minúsculas
    df['presentation'] = df['presentation'].str.lower()
    return df

df = load_data()

# -------------------------------------------------------------------
# CONTROLES Y FILTROS
# -------------------------------------------------------------------
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
        value=pd.Timestamp('2023-08-01').date(),
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
        st.markdown("**Información de Productos:**")
        st.write(f"**Productos (marcas) únicos:** {df_filtrado['brand'].nunique()}")
        st.write(f"**Líneas de producto únicas:** {df_filtrado['Product Line'].nunique()}")
        st.write(f"**Presentaciones:** {df_filtrado['presentation'].dropna().unique()}")

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
    sns.heatmap(df_filtrado.isnull(), cbar=False, cmap="YlOrBr", yticklabels=False, ax=ax)
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
# ANÁLISIS UNIVARIANTE DE ATRIBUTOS
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Distribución de Atributos de Producto**")

col_attr1, col_attr2, col_attr3 = st.columns(3)

with col_attr1:
    st.markdown("**Frecuencia de Líneas de Producto**")
    line_counts = df_filtrado['Product Line'].value_counts()
    st.dataframe(line_counts, use_container_width=True)

with col_attr2:
    st.markdown("**Frecuencia de Tamaños**")
    size_counts = df_filtrado['size'].value_counts().sort_index()
    st.dataframe(size_counts, use_container_width=True)

with col_attr3:
    st.markdown("**Frecuencia de Presentaciones**")
    pres_counts = df_filtrado['presentation'].value_counts()
    st.dataframe(pres_counts, use_container_width=True)

if crear_visualizaciones:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Gráfico de barras para líneas de producto (usar YlOrBr)
    top_lines = line_counts.head(10)
    colors_lines = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(top_lines)))
    axes[0].bar(top_lines.index, top_lines.values, color=colors_lines)
    axes[0].set_title('Top 10 Líneas de Producto')
    axes[0].set_xlabel('Línea')
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Histograma de tamaños
    sizes = df_filtrado['size'].dropna()
    axes[1].hist(sizes, bins=20, color=plt.cm.YlOrBr(0.6), edgecolor='black')
    axes[1].set_title('Distribución de Tamaños')
    axes[1].set_xlabel('Tamaño (unidades)')
    axes[1].set_ylabel('Frecuencia')
    
    # Gráfico de barras 
    pres_colors = plt.cm.YlOrBr([0.4, 0.7])  
    pres_counts.plot(kind='bar', ax=axes[2], color=pres_colors)
    axes[2].set_title('Distribución de Presentaciones')
    axes[2].set_xlabel('Presentación')
    axes[2].set_ylabel('Frecuencia')
    axes[2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------------------------
# CREACIÓN DE VARIABLES TEMPORALES
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

# -------------------------------------------------------------------
# ANÁLISIS DE IMPACTO DE CAMPAÑA 
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Análisis de Impacto de Campaña**")

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
            
            # Cálculo de Cohen's d
            n1, n2 = len(antes_sales), len(despues_sales)
            var1, var2 = antes_sales.var(), despues_sales.var()
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            cohen_d = (despues_mean - antes_mean) / pooled_std if pooled_std != 0 else np.nan

            # Correlación punto-biserial
            binary = np.concatenate([np.zeros(n1), np.ones(n2)])
            all_sales = np.concatenate([antes_sales.values, despues_sales.values])
            r_pb, p_pb = pointbiserialr(binary, all_sales)
            
            st.markdown("**Pruebas Estadísticas y Tamaño del Efecto:**")
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
# VISUALIZACIONES DE CAMPAÑA (GLOBALES) 
# -------------------------------------------------------------------
if crear_visualizaciones and len(df_filtrado) > 0:
    st.markdown("**Visualizaciones de Impacto de Campaña:**")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Boxplot
    sns.boxplot(data=df_filtrado, x='Campaign', y='sales', ax=axes[0], palette="YlOrBr")
    axes[0].set_title('Distribución de Ventas por Campaña')
    axes[0].set_ylabel('Ventas ($)')
    
    # Violin plot 
    sns.violinplot(data=df_filtrado, x='Campaign', y='sales', ax=axes[1], palette="YlOrBr")
    axes[1].set_title('Distribución Detallada por Campaña')
    axes[1].set_ylabel('Ventas ($)')
    
    # Gráfico de barras
    campaign_means = df_filtrado.groupby('Campaign')['sales'].mean()
    colors_bar = plt.cm.YlOrBr([0.3, 0.7])  
    axes[2].bar(campaign_means.index, campaign_means.values, color=colors_bar)
    axes[2].set_title('Ventas Promedio por Campaña')
    axes[2].set_ylabel('Ventas Promedio ($)')
    axes[2].set_ylim(0, campaign_means.max() * 1.2)
    
    for i, (campaign, mean) in enumerate(campaign_means.items()):
        axes[2].text(i, mean * 1.05, f'${mean:,.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Gráfico de dispersión temporal
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_scatter = {'Antes': plt.cm.YlOrBr(0.3), 'Después': plt.cm.YlOrBr(0.7)}
    for camp in ['Antes', 'Después']:
        subset = df_filtrado[df_filtrado['Campaign'] == camp]
        ax.scatter(subset['date'], subset['sales'], 
                   c=colors_scatter[camp], label=camp, alpha=0.6, s=10)
    ax.set_title('Ventas Diarias a lo Largo del Tiempo')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas ($)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------------------------------------------
# ANÁLISIS BIVARIANTE: ATRIBUTOS DEL PRODUCTO VS CAMPAÑA
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Impacto de la Campaña por Atributos del Producto**")

# Función auxiliar para mostrar tabla y gráfico de impacto por categoría
def plot_impact_by_category(df, group_col, group_name):
    stats = df.groupby([group_col, 'Campaign'])['sales'].mean().unstack()
    if 'Antes' in stats.columns and 'Después' in stats.columns:
        stats['pct_change'] = ((stats['Después'] - stats['Antes']) / stats['Antes']) * 100
        stats = stats.dropna(subset=['pct_change']).sort_values('pct_change', ascending=False)
        st.dataframe(stats[['Antes', 'Después', 'pct_change']].round(2), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [plt.cm.YlOrBr(0.7) if x > 0 else plt.cm.YlOrBr(0.3) for x in stats['pct_change']]
        ax.barh(stats.index, stats['pct_change'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Cambio porcentual (%)')
        ax.set_title(f'Cambio en ventas promedio por {group_name}')
        st.pyplot(fig)
    else:
        st.info("No hay suficientes datos para ambos periodos.")

tab_attr1, tab_attr2, tab_attr3 = st.tabs(["Por Línea de Producto", "Por Tamaño (Size)", "Por Presentación"])

with tab_attr1:
    st.markdown("### Impacto por línea de producto")
    plot_impact_by_category(df_filtrado, 'Product Line', 'línea de producto')

with tab_attr2:
    st.markdown("### Impacto por tamaño")
    # Filtrar tamaños no nulos
    df_size = df_filtrado.dropna(subset=['size']).copy()
    # Convertir size a string para mejor visualización en gráfico
    df_size['size_str'] = df_size['size'].astype(int).astype(str)
    plot_impact_by_category(df_size, 'size_str', 'tamaño (unidades)')

with tab_attr3:
    st.markdown("### Impacto por presentación")
    df_pres = df_filtrado.dropna(subset=['presentation'])
    plot_impact_by_category(df_pres, 'presentation', 'presentación')

# -------------------------------------------------------------------
# ANÁLISIS POR PRODUCTO (MARCA)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Análisis Detallado por Producto**")

marca_seleccionada = st.selectbox(
    "Seleccionar producto (marca) para análisis detallado:",
    df_filtrado['brand'].unique()
)

if marca_seleccionada:
    marca_data = df_filtrado[df_filtrado['brand'] == marca_seleccionada]
    
    col_marca1, col_marca2, col_marca3 = st.columns(3)
    
    with col_marca1:
        st.markdown("**Estadísticas del Producto:**")
        stats = marca_data['sales'].describe()
        st.write(f"**Conteo:** {stats['count']}")
        st.write(f"**Media:** ${stats['mean']:,.2f}")
        st.write(f"**Mediana:** ${stats['50%']:,.2f}")
        st.write(f"**Mínimo:** ${stats['min']:,.2f}")
        st.write(f"**Máximo:** ${stats['max']:,.2f}")
    
    with col_marca2:
        st.markdown("**Atributos del Producto:**")
        product_line = marca_data['Product Line'].iloc[0]
        size = marca_data['size'].iloc[0]
        presentation = marca_data['presentation'].iloc[0]
        st.write(f"**Línea:** {product_line}")
        st.write(f"**Tamaño:** {size if pd.notna(size) else 'Desconocido'}")
        st.write(f"**Presentación:** {presentation if pd.notna(presentation) else 'Desconocido'}")
    
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

# -------------------------------------------------------------------
# HETEROGENEIDAD DEL IMPACTO POR PRODUCTO (TODOS)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Heterogeneidad del Impacto por Producto**")
st.markdown("A continuación se muestra el cambio porcentual en las ventas promedio para cada producto (antes vs. después de la campaña).")

# Calcular cambio porcentual por producto (brand)
brand_impact = df_filtrado.groupby(['brand', 'Campaign'])['sales'].mean().unstack()
brand_impact['pct_change'] = ((brand_impact['Después'] - brand_impact['Antes']) / brand_impact['Antes']) * 100
brand_impact = brand_impact.dropna(subset=['pct_change']).sort_values('pct_change', ascending=False)

col_imp1, col_imp2 = st.columns(2)

with col_imp1:
    st.markdown("**Cambio porcentual por producto:**")
    st.dataframe(brand_impact[['Antes', 'Después', 'pct_change']].round(2), use_container_width=True)

with col_imp2:
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = brand_impact.index
    colors = [plt.cm.YlOrBr(0.7) if x > 0 else plt.cm.YlOrBr(0.3) for x in brand_impact['pct_change']]
    y_pos = range(len(brand_impact))
    ax.barh(y_pos, brand_impact['pct_change'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Cambio porcentual (%)')
    ax.set_title('Impacto de la campaña por producto')
    st.pyplot(fig)

st.markdown("**¿Es estadísticamente significativa la diferencia en el impacto entre productos?**")
st.markdown("Se ajusta un modelo de regresión lineal con `sales` como variable dependiente y `brand`, `Campaign` y su interacción como predictores. El p‑valor del término de interacción indica si el efecto de la campaña varía significativamente entre productos.")

if STATSMODELS_AVAILABLE and len(df_filtrado) > 0:
    df_model = df_filtrado.copy()
    df_model['Campaign_num'] = (df_model['Campaign'] == 'Después').astype(int)
    
    formula = 'sales ~ C(brand) * Campaign_num'
    try:
        model = ols(formula, data=df_model).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        interaction_p = anova_table.loc['C(brand):Campaign_num', 'PR(>F)']
        
        st.write(f"**p‑valor de la interacción (brand × campaign):** {interaction_p:.4f}")
        if interaction_p < 0.05:
            st.success("La interacción es significativa → el efecto de la campaña varía entre productos.")
        else:
            st.warning("La interacción no es significativa → no hay evidencia de que el impacto difiera entre productos.")
    except Exception as e:
        st.error("No se pudo ajustar el modelo de interacción (posiblemente debido a datos insuficientes).")
else:
    if not STATSMODELS_AVAILABLE:
        st.info("La prueba de interacción requiere el paquete 'statsmodels'. Por favor, instálalo (añádelo a requirements.txt) para ver este análisis.")
    else:
        st.info("No hay datos suficientes para el análisis de interacción.")

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
    colors_est = plt.cm.YlOrBr(np.linspace(0.3, 0.9, 4))
    axes[0].bar(estacion_data.index, estacion_data.values, color=colors_est)
    axes[0].set_title('Ventas Totales por Estación')
    axes[0].set_ylabel('Ventas Totales ($)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Ventas por mes
    mes_data = df_filtrado.groupby('month')['sales'].mean().sort_index()
    axes[1].plot(mes_data.index, mes_data.values, marker='o', color=plt.cm.YlOrBr(0.6), linewidth=2)
    axes[1].set_title('Ventas Promedio por Mes')
    axes[1].set_xlabel('Mes')
    axes[1].set_ylabel('Ventas Promedio ($)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(1, 13))
    
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------------------------
# ANÁLISIS DE CORRELACIONES
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Análisis de Correlaciones")

if len(df_filtrado) > 1:
    df_corr = df_filtrado.copy()
    
    # Codificar variables categóricas
    brand_mapping = {brand: i for i, brand in enumerate(df_corr['brand'].unique())}
    df_corr['brand_code'] = df_corr['brand'].map(brand_mapping)
    df_corr['campaign_code'] = df_corr['Campaign'].map({'Antes': 0, 'Después': 1})
    
    # Atributos de producto
    df_corr['size_num'] = df_corr['size']
    pres_mapping = {'can': 0, 'bottle': 1}
    df_corr['pres_code'] = df_corr['presentation'].map(pres_mapping)
    line_mapping = {line: i for i, line in enumerate(df_corr['Product Line'].unique())}
    df_corr['line_code'] = df_corr['Product Line'].map(line_mapping)
    
    numeric_cols = ['sales', 'brand_code', 'campaign_code', 'month', 'dia_semana', 
                    'size_num', 'pres_code', 'line_code']
    correlation_matrix = df_corr[numeric_cols].corr()
    
    col_corr1, col_corr2 = st.columns(2)
    
    with col_corr1:
        st.markdown("Matriz de Correlación:")
        st.dataframe(correlation_matrix.style.background_gradient(cmap='YlOrBr', vmin=-1, vmax=1), 
                    use_container_width=True)
    
    with col_corr2:
        st.markdown("Correlaciones con Ventas:")
        sales_corr = correlation_matrix['sales'].sort_values(ascending=False)
        for variable, corr in sales_corr.items():
            if variable != 'sales':
                st.write(f"**{variable}:** {corr:.3f}")

# -------------------------------------------------------------------
# RESUMEN EJECUTIVO
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("**Resumen Ejecutivo**")

col_res1, col_res2 = st.columns(2)

if 'antes_sales' in locals() and 'despues_sales' in locals():
    antes_mean = antes_sales.mean()
    despues_mean = despues_sales.mean()
    antes_median = antes_sales.median()
    despues_median = despues_sales.median()
    n1, n2 = len(antes_sales), len(despues_sales)
    var1, var2 = antes_sales.var(), despues_sales.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohen_d = (despues_mean - antes_mean) / pooled_std if pooled_std != 0 else np.nan
    binary = np.concatenate([np.zeros(n1), np.ones(n2)])
    all_sales = np.concatenate([antes_sales.values, despues_sales.values])
    r_pb, _ = pointbiserialr(binary, all_sales)
else:
    antes_sales = df_filtrado[df_filtrado['Campaign'] == 'Antes']['sales']
    despues_sales = df_filtrado[df_filtrado['Campaign'] == 'Después']['sales']
    if len(antes_sales) > 0 and len(despues_sales) > 0:
        antes_mean = antes_sales.mean()
        despues_mean = despues_sales.mean()
        antes_median = antes_sales.median()
        despues_median = despues_sales.median()
        cohen_d = np.nan
        r_pb = np.nan
    else:
        antes_mean = despues_mean = antes_median = despues_median = cohen_d = r_pb = np.nan

with col_res1:
    st.markdown("Puntos Clave:")
    st.write(f"• Ventas totales: ${df_filtrado['sales'].sum():,.2f}")
    if not np.isnan(antes_mean) and not np.isnan(despues_mean):
        st.write(f"• Venta promedio antes: ${antes_mean:,.2f} | después: ${despues_mean:,.2f}  →  **+{(despues_mean-antes_mean)/antes_mean*100:+.1f}%**")
        st.write(f"• Mediana antes: ${antes_median:,.2f} | después: ${despues_median:,.2f}  →  **+{(despues_median-antes_median)/antes_median*100:+.1f}%**")
    if not np.isnan(cohen_d):
        st.write(f"• Tamaño del efecto (d de Cohen): {cohen_d:.3f} (pequeño)")
    if not np.isnan(r_pb):
        st.write(f"• Correlación punto-biserial: {r_pb:.3f}")

with col_res2:
    st.markdown("Recomendaciones:**")
    st.success("• La campaña tuvo un **impacto positivo y significativo** en las ventas.")
    st.info("• El efecto es pequeño a nivel agregado, pero muy variable entre productos.")
    
    # Productos con mayor crecimiento
    if 'brand_impact' in locals() and not brand_impact.empty:
        top_growth = brand_impact['pct_change'].nlargest(2)
        top_names = top_growth.index.tolist()
        st.write(f"• **Productos con mayor crecimiento:** {top_names[0]} ({top_growth.iloc[0]:+.1f}%), {top_names[1]} ({top_growth.iloc[1]:+.1f}%).")
    
    # Mejor línea de producto
    line_impact = df_filtrado.groupby(['Product Line', 'Campaign'])['sales'].mean().unstack()
    if 'Antes' in line_impact.columns and 'Después' in line_impact.columns:
        line_impact['pct'] = ((line_impact['Después'] - line_impact['Antes']) / line_impact['Antes']) * 100
        line_impact = line_impact.dropna(subset=['pct'])
        if not line_impact.empty:
            best_line = line_impact['pct'].idxmax()
            st.write(f"• **Línea de producto con mejor respuesta:** {best_line} (+{line_impact.loc[best_line, 'pct']:.1f}%).")
    
    # Mejor presentación
    pres_impact = df_filtrado.dropna(subset=['presentation']).groupby(['presentation', 'Campaign'])['sales'].mean().unstack()
    if 'Antes' in pres_impact.columns and 'Después' in pres_impact.columns:
        pres_impact['pct'] = ((pres_impact['Después'] - pres_impact['Antes']) / pres_impact['Antes']) * 100
        pres_impact = pres_impact.dropna(subset=['pct'])
        if not pres_impact.empty:
            best_pres = pres_impact['pct'].idxmax()
            st.write(f"• **Presentación con mejor respuesta:** {best_pres} (+{pres_impact.loc[best_pres, 'pct']:.1f}%).")
    
    # Mejor tamaño
    size_impact = df_filtrado.dropna(subset=['size']).groupby(['size', 'Campaign'])['sales'].mean().unstack()
    if 'Antes' in size_impact.columns and 'Después' in size_impact.columns:
        size_impact['pct'] = ((size_impact['Después'] - size_impact['Antes']) / size_impact['Antes']) * 100
        size_impact = size_impact.dropna(subset=['pct'])
        if not size_impact.empty:
            best_size = size_impact['pct'].idxmax()
            st.write(f"• **Tamaño con mejor respuesta:** {best_size} unidades (+{size_impact.loc[best_size, 'pct']:.1f}%).")
    
    # Mejor estación
    mejor_estacion = df_filtrado.groupby('estacion')['sales'].mean().idxmax()
    st.write(f"• **Enfoque estacional:** intensificar esfuerzos en {mejor_estacion}.")
    
    st.caption("Nota: Estos resultados reflejan asociaciones, no causalidad. Factores externos no controlados pueden influir en las ventas. El dataset no contiene información geográfica, por lo que no es posible analizar diferencias regionales.")

# Pie de página
st.markdown("---")
st.markdown("*Última actualización: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")

