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
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Crear columnas para controles
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("###**Filtros de Fecha**")
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
st.subheader("**Anális de Valores Faltantes**")

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

# Crear características temporales
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

# Análisis de distribución por campaña
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
        
        # Pruebas estadísticas
        antes_sales = df_filtrado[df_filtrado['Campaign'] == 'Antes']['sales']
        despues_sales = df_filtrado[df_filtrado['Campaign'] == 'Después']['sales']
        
        if len(antes_sales) > 1 and len(despues_sales) > 1:
            t_stat, p_value_t = ttest_ind(despues_sales, antes_sales, equal_var=False)
            u_stat, p_value_u = mannwhitneyu(despues_sales, antes_sales, alternative='two-sided')
            
            st.markdown("**Pruebas Estadísticas:**")
            st.write(f"**Prueba t (Welch):** p = {p_value_t:.4f}")
            st.write(f"**Mann-Whitney U:** p = {p_value_u:.4f}")
            
            if p_value_t < 0.05:
                if despues_mean > antes_mean:
                    st.success("✅ La campaña parece EXITOSA (diferencia significativa)")
                else:
                    st.error("❌ La campaña parece NO EXITOSA (diferencia significativa)")
            else:
                st.warning("⚠️ No se detectó impacto estadísticamente significativo")
    except:
        st.warning("No hay suficientes datos para ambos periodos de campaña")

# Visualizaciones de campaña
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

# Análisis por marca
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

# Análisis estacional
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

# Análisis de correlaciones
st.markdown("---")
st.subheader("🔗 **Análisis de Correlaciones**")

if len(df_filtrado) > 1:
    # Crear variables numéricas para correlación
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
        st.markdown("**Correlaciones con Ventas:**")
        sales_corr = correlation_matrix['sales'].sort_values(ascending=False)
        for variable, corr in sales_corr.items():
            if variable != 'sales':
                st.write(f"**{variable}:** {corr:.3f}")
    
    if crear_visualizaciones:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Mapa de Correlaciones')
        st.pyplot(fig)

# Resumen ejecutivo
st.markdown("---")
st.subheader("**Resumen Ejecutivo**")

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.markdown("**Puntos Clave:**")
    
    # Ventas totales
    ventas_totales = df_filtrado['sales'].sum()
    st.write(f"• **Ventas totales:** ${ventas_totales:,.2f}")
    
    # Ventas promedio
    ventas_promedio = df_filtrado['sales'].mean()
    st.write(f"• **Venta promedio:** ${ventas_promedio:,.2f}")
    
    # Mejor marca
    mejor_marca = df_filtrado.groupby('brand')['sales'].sum().idxmax()
    ventas_mejor_marca = df_filtrado.groupby('brand')['sales'].sum().max()
    st.write(f"• **Mejor marca:** {mejor_marca} (${ventas_mejor_marca:,.2f})")
    
    # Mejor día de la semana
    if 'dia_semana' in df_filtrado.columns:
        mejor_dia = df_filtrado.groupby('dia_semana')['sales'].mean().idxmax()
        dias = {1: 'Lunes', 2: 'Martes', 3: 'Miércoles', 4: 'Jueves', 
               5: 'Viernes', 6: 'Sábado', 7: 'Domingo'}
        st.write(f"• **Mejor día para ventas:** {dias.get(mejor_dia, 'N/A')}")

with col_res2:
    st.markdown("**Recomendaciones:**")
    
    try:
        if 'Antes' in df_filtrado['Campaign'].unique() and 'Después' in df_filtrado['Campaign'].unique():
            antes_sales = df_filtrado[df_filtrado['Campaign'] == 'Antes']['sales']
            despues_sales = df_filtrado[df_filtrado['Campaign'] == 'Después']['sales']
            
            if len(antes_sales) > 1 and len(despues_sales) > 1:
                t_stat, p_value_t = ttest_ind(despues_sales, antes_sales, equal_var=False)
                
                if p_value_t < 0.05:
                    if despues_sales.mean() > antes_sales.mean():
                        st.success("• **Campaña exitosa:** Continuar con estrategias similares")
                    else:
                        st.error("• **Revisar campaña:** Analizar posibles mejoras")
                else:
                    st.warning("• **Datos insuficientes:** Considerar extender periodo de prueba")
            else:
                st.info("• **Más datos necesarios:** Recolectar más información para análisis confiable")
    except:
        st.info("• **Análisis pendiente:** Ejecutar análisis completo para recomendaciones")
    
    # Recomendación basada en estacionalidad
    if 'estacion' in df_filtrado.columns:
        mejor_estacion = df_filtrado.groupby('estacion')['sales'].mean().idxmax()
        st.write(f"• **Enfoque estacional:** Intensificar esfuerzos en {mejor_estacion}")

# Pie de página
st.markdown("---")
st.markdown("*Última actualización: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
