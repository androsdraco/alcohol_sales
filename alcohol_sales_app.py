import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, t
from datetime import datetime, timedelta
import warnings
import re
import os
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Análisis de Ventas de Alcohol",
    layout="wide"
)

# Título
st.title("Panel de Análisis de Ventas de Alcohol")

# Sección de configuración en la parte superior
st.header("Configuración de Datos")

# Cargar archivo desde ubicación fija
file_path = r"CSV/Alcohol sales.csv"

# Verificar si el archivo existe
if not os.path.exists(file_path):
    st.error(f"No se encontró el archivo en la ubicación: {file_path}")
    st.write("Por favor, asegúrate de que el archivo 'Alcohol sales.csv' esté en la carpeta CSV")
    st.stop()

st.write(f"Archivo cargado: {file_path}")

# Configuración de campaña
st.header("Configuración de Campaña")

col3, col4, col5 = st.columns(3)

with col3:
    campaign_date = st.date_input(
        "Fecha de Inicio de Campaña",
        value=pd.Timestamp('2023-01-10'),
        help="Selecciona la fecha cuando comenzó tu campaña"
    )

with col4:
    analysis_end_date = st.date_input(
        "Fecha Final de Análisis",
        value=pd.Timestamp('2023-03-10'),
        help="Selecciona la fecha final para tu análisis"
    )

with col5:
    period_option = st.selectbox(
        "Longitud del Periodo",
        ["1 día", "1 semana"],
        help="Selecciona la longitud del periodo para comparación"
    )

# Convertir opción de periodo a días
period_days = {"1 día": 1, "1 semana": 7}
comparison_days = period_days[period_option]

# Convertir a Timestamps
campaign_date = pd.Timestamp(campaign_date)
analysis_end_date = pd.Timestamp(analysis_end_date)

# Función para limpiar datos de ventas
def clean_sales_data(df):
    if df is None:
        return None
    
    df = df.copy()
    
    if 'sales' not in df.columns:
        return None
    
    def clean_sale_value(value):
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
    
    df['sales'] = df['sales'].apply(clean_sale_value)
    df = df.dropna(subset=['sales'])
    df = df[df['sales'] >= 0]
    
    return df

# Función para cargar datos
def load_data():
    try:
        # Leer archivo desde ubicación fija
        df = pd.read_csv(file_path)
        
        # Manejar columna de fecha
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
        df = clean_sales_data(df)
        if df is None:
            return None
        
        # Buscar columna de marca
        brand_cols = ['brand', 'Brand', 'marca', 'Marca', 'producto', 'Producto']
        for brand_col in brand_cols:
            if brand_col in df.columns:
                df['brand'] = df[brand_col].astype(str)
                break
        
        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=['date'])
        
        # Filtrar datos hasta la fecha final de análisis
        df = df[df['date'] <= analysis_end_date]
        
        # Ordenar por fecha
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error cargando archivo: {str(e)}")
        return None

# Cargar los datos
df = load_data()

if df is None:
    st.stop()

# Mostrar información básica del archivo
st.write(f"Total de registros: {len(df)}")
st.write(f"Rango de fechas: {df['date'].min().date()} al {df['date'].max().date()}")

# Análisis Exploratorio de Datos (EDA)
st.markdown("---")
st.header("Análisis Exploratorio de Datos")

# Métricas básicas
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = df['sales'].sum()
    st.metric("Ventas Totales", f"${total_sales:,.0f}")

with col2:
    avg_sale = df['sales'].mean()
    st.metric("Venta Promedio", f"${avg_sale:.2f}")

with col3:
    transaction_count = len(df)
    st.metric("Total Transacciones", f"{transaction_count:,}")

with col4:
    unique_products = df['brand'].nunique()
    st.metric("Productos Únicos", f"{unique_products}")

# Gráficos EDA
col5, col6 = st.columns(2)

with col5:
    # Distribución de ventas
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['sales'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Monto de Venta ($)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Montos de Venta')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col6:
    # Ventas por día de la semana
    df['dia_semana'] = df['date'].dt.day_name()
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)
    
    ventas_por_dia = df.groupby('dia_semana')['sales'].mean().reindex(dias_orden)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(ventas_por_dia)), ventas_por_dia.values, alpha=0.7, color='forestgreen')
    ax.set_xlabel('Día de la Semana')
    ax.set_ylabel('Venta Promedio ($)')
    ax.set_title('Venta Promedio por Día de la Semana')
    ax.set_xticks(range(len(ventas_por_dia)))
    ax.set_xticklabels(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars, ventas_por_dia.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'${val:.0f}', ha='center', va='bottom')
    
    st.pyplot(fig)

# Top productos
st.subheader("Top 10 Productos por Ventas")

top_products = df.groupby('brand')['sales'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(top_products)), top_products.values, alpha=0.7, color='darkorange')
ax.set_yticks(range(len(top_products)))
ax.set_yticklabels(top_products.index)
ax.set_xlabel('Ventas Totales ($)')
ax.set_title('Top 10 Productos por Ventas Totales')
ax.grid(True, alpha=0.3, axis='x')

# Añadir valores en las barras
for i, (bar, val) in enumerate(zip(bars, top_products.values)):
    ax.text(val + max(top_products.values)*0.01, bar.get_y() + bar.get_height()/2, 
            f'${val:,.0f}', va='center')

st.pyplot(fig)

# Función para crear periodos iguales antes y después
def create_periods(df, campaign_date, end_date, period_days):
    # Calcular duración de la campaña
    campaign_duration = (end_date - campaign_date).days + 1
    
    # Calcular número máximo de periodos iguales (máximo 6)
    max_periods = min(6, campaign_duration // period_days)
    
    periods = []
    
    # Crear periodos de campaña
    for i in range(max_periods):
        start_date = campaign_date + timedelta(days=i*period_days)
        end_period = min(start_date + timedelta(days=period_days-1), end_date)
        
        label = f"Campaña {i+1}"
        if period_days == 1:
            label += f": {start_date.strftime('%d/%m')}"
        else:
            label += f": {start_date.strftime('%d/%m')}-{end_period.strftime('%d/%m')}"
        
        periods.append({
            'period': f'despues_{i+1}',
            'start': start_date,
            'end': end_period,
            'label': label,
            'campaign': 'despues'
        })
    
    # Crear periodos antes (mismo número que periodos después)
    for i in range(max_periods):
        end_date_before = campaign_date - timedelta(days=1)
        start_date = end_date_before - timedelta(days=(i+1)*period_days - 1)
        
        if start_date < df['date'].min():
            break
            
        label = f"Antes {i+1}"
        if period_days == 1:
            label += f": {start_date.strftime('%d/%m')}"
        else:
            end_period = start_date + timedelta(days=period_days-1)
            label += f": {start_date.strftime('%d/%m')}-{end_period.strftime('%d/%m')}"
        
        periods.append({
            'period': f'antes_{i+1}',
            'start': start_date,
            'end': start_date + timedelta(days=period_days-1),
            'label': label,
            'campaign': 'antes'
        })
    
    # Ordenar cronológicamente
    periods.sort(key=lambda x: x['start'])
    
    # Asignar periodos a los datos
    df['periodo'] = 'no incluido'
    df['etiqueta_periodo'] = 'no incluido'
    df['grupo'] = 'no incluido'
    
    for period in periods:
        mask = (df['date'] >= period['start']) & (df['date'] <= period['end'])
        df.loc[mask, 'periodo'] = period['period']
        df.loc[mask, 'etiqueta_periodo'] = period['label']
        df.loc[mask, 'grupo'] = period['campaign']
    
    return df, periods, max_periods

# Crear periodos
df, periods, num_periods = create_periods(df, campaign_date, analysis_end_date, comparison_days)

# Análisis comparativo
st.markdown("---")
st.header("Análisis Comparativo: Antes vs Después")

st.write(f"Método de comparación: {num_periods} periodos de {period_option} antes y {num_periods} periodos de {period_option} después")

# Datos para análisis comparativo
df_periodos = df[df['grupo'] != 'no incluido'].copy()

if len(df_periodos) > 0:
    # Calcular métricas por periodo
    period_metrics = df_periodos.groupby(['etiqueta_periodo', 'grupo']).agg({
        'sales': ['sum', 'mean', 'count']
    }).round(2).reset_index()
    
    period_metrics.columns = ['Periodo', 'Grupo', 'Ventas_Totales', 'Venta_Promedio', 'Transacciones']
    
    # Ordenar periodos
    orden_periodos = []
    for i in range(num_periods, 0, -1):
        orden_periodos.append(f'Antes {i}')
    for i in range(1, num_periods + 1):
        orden_periodos.append(f'Campaña {i}')
    
    period_metrics['Orden'] = pd.Categorical(
        period_metrics['Periodo'].str.extract(r'(\w+) (\d+)')[0] + ' ' + 
        period_metrics['Periodo'].str.extract(r'(\w+) (\d+)')[1],
        categories=orden_periodos,
        ordered=True
    )
    period_metrics = period_metrics.sort_values('Orden')
    
    # Gráfico comparativo
    col7, col8 = st.columns(2)
    
    with col7:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(period_metrics))
        colors = ['steelblue' if g == 'antes' else 'forestgreen' for g in period_metrics['Grupo']]
        
        bars = ax.bar(x, period_metrics['Ventas_Totales'], color=colors, alpha=0.7, width=0.6)
        ax.set_xlabel('Periodo')
        ax.set_ylabel('Ventas Totales ($)')
        ax.set_title('Comparación de Ventas por Periodo')
        ax.set_xticks(x)
        ax.set_xticklabels(period_metrics['Periodo'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Línea divisoria
        ax.axvline(x=num_periods-0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Etiquetas
        for bar, val in zip(bars, period_metrics['Ventas_Totales']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(period_metrics['Ventas_Totales'])*0.01,
                   f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig)
    
    with col8:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separar datos antes y después
        antes_data = period_metrics[period_metrics['Grupo'] == 'antes']['Venta_Promedio'].values
        despues_data = period_metrics[period_metrics['Grupo'] == 'despues']['Venta_Promedio'].values
        
        x = np.arange(len(antes_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, antes_data, width, label='Antes', alpha=0.7, color='steelblue')
        bars2 = ax.bar(x + width/2, despues_data, width, label='Después', alpha=0.7, color='forestgreen')
        
        ax.set_xlabel('Número de Periodo')
        ax.set_ylabel('Venta Promedio ($)')
        ax.set_title('Venta Promedio: Antes vs Después')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{i+1}' for i in range(len(antes_data))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Añadir valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + max(np.concatenate([antes_data, despues_data]))*0.01,
                       f'${height:.0f}', ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)
    
    # Análisis estadístico
    st.subheader("Análisis Estadístico")
    
    col9, col10 = st.columns(2)
    
    with col9:
        # Extraer datos para prueba t
        ventas_antes = df_periodos[df_periodos['grupo'] == 'antes']['sales']
        ventas_despues = df_periodos[df_periodos['grupo'] == 'despues']['sales']
        
        if len(ventas_antes) > 1 and len(ventas_despues) > 1:
            t_stat, p_valor = ttest_ind(ventas_despues, ventas_antes, equal_var=False)
            
            # Tamaño del efecto
            n1, n2 = len(ventas_despues), len(ventas_antes)
            pooled_sd = np.sqrt(((n1-1)*ventas_despues.var() + (n2-1)*ventas_antes.var()) / (n1+n2-2))
            if pooled_sd != 0:
                cohens_d = (ventas_despues.mean() - ventas_antes.mean()) / pooled_sd
            else:
                cohens_d = 0
            
            st.metric("Valor p", f"{p_valor:.6f}")
            st.metric("Significancia", "Significativo" if p_valor < 0.05 else "No significativo")
            st.metric("Tamaño efecto (d)", f"{cohens_d:.3f}")
            
            # Calcular intervalos de confianza
            def intervalo_confianza(data, confianza=0.95):
                n = len(data)
                media = np.mean(data)
                error = np.std(data, ddof=1) / np.sqrt(n)
                margen = error * t.ppf((1 + confianza)/2, n-1)
                return media - margen, media + margen
            
            ic_antes = intervalo_confianza(ventas_antes)
            ic_despues = intervalo_confianza(ventas_despues)
            
            st.write("Intervalos 95% confianza:")
            st.write(f"Antes: ${ic_antes[0]:.2f} - ${ic_antes[1]:.2f}")
            st.write(f"Después: ${ic_despues[0]:.2f} - ${ic_despues[1]:.2f}")
    
    with col10:
        # Métricas agregadas
        ventas_totales_antes = df_periodos[df_periodos['grupo'] == 'antes']['sales'].sum()
        ventas_totales_despues = df_periodos[df_periodos['grupo'] == 'despues']['sales'].sum()
        
        promedio_antes = df_periodos[df_periodos['grupo'] == 'antes']['sales'].mean()
        promedio_despues = df_periodos[df_periodos['grupo'] == 'despues']['sales'].mean()
        
        transacciones_antes = len(df_periodos[df_periodos['grupo'] == 'antes'])
        transacciones_despues = len(df_periodos[df_periodos['grupo'] == 'despues'])
        
        cambio_ventas = ((ventas_totales_despues - ventas_totales_antes) / ventas_totales_antes) * 100
        cambio_promedio = ((promedio_despues - promedio_antes) / promedio_antes) * 100
        cambio_transacciones = ((transacciones_despues - transacciones_antes) / transacciones_antes) * 100
        
        st.write("Métricas Agregadas:")
        st.write(f"Ventas totales antes: ${ventas_totales_antes:,.0f}")
        st.write(f"Ventas totales después: ${ventas_totales_despues:,.0f}")
        st.write(f"Cambio ventas: {cambio_ventas:+.1f}%")
        
        st.write(f"Venta promedio antes: ${promedio_antes:.2f}")
        st.write(f"Venta promedio después: ${promedio_despues:.2f}")
        st.write(f"Cambio promedio: {cambio_promedio:+.1f}%")
        
        st.write(f"Transacciones antes: {transacciones_antes:,}")
        st.write(f"Transacciones después: {transacciones_despues:,}")
        st.write(f"Cambio transacciones: {cambio_transacciones:+.1f}%")
    
    # Análisis de productos durante campaña
    st.subheader("Análisis de Productos Durante Campaña")
    
    # Mejores productos durante campaña
    productos_campaña = df_periodos[df_periodos['grupo'] == 'despues']
    
    if len(productos_campaña) > 0:
        # Top productos por crecimiento
        ventas_antes_por_producto = df_periodos[df_periodos['grupo'] == 'antes'].groupby('brand')['sales'].sum()
        ventas_despues_por_producto = productos_campaña.groupby('brand')['sales'].sum()
        
        crecimiento_productos = pd.DataFrame({
            'antes': ventas_antes_por_producto,
            'despues': ventas_despues_por_producto
        }).fillna(0)
        
        crecimiento_productos['cambio'] = ((crecimiento_productos['despues'] - crecimiento_productos['antes']) / 
                                          crecimiento_productos['antes'].replace(0, np.nan)) * 100
        crecimiento_productos['cambio'] = crecimiento_productos['cambio'].replace([np.inf, -np.inf], np.nan)
        
        # Productos con mayor crecimiento
        top_crecimiento = crecimiento_productos.dropna().nlargest(10, 'cambio')
        
        col11, col12 = st.columns(2)
        
        with col11:
            # Gráfico de mejores productos
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if len(top_crecimiento) > 0:
                bars = ax.barh(range(len(top_crecimiento)), top_crecimiento['cambio'].values, 
                              alpha=0.7, color='green')
                ax.set_yticks(range(len(top_crecimiento)))
                ax.set_yticklabels(top_crecimiento.index)
                ax.set_xlabel('Crecimiento (%)')
                ax.set_title('Top 10 Productos por Crecimiento')
                ax.grid(True, alpha=0.3, axis='x')
                
                for i, (bar, val) in enumerate(zip(bars, top_crecimiento['cambio'].values)):
                    ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                           f'{val:+.1f}%', va='center')
            
            st.pyplot(fig)
        
        with col12:
            # Productos con mayores ventas durante campaña
            top_ventas_campaña = productos_campaña.groupby('brand')['sales'].sum().nlargest(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(top_ventas_campaña)), top_ventas_campaña.values, 
                          alpha=0.7, color='darkorange')
            ax.set_yticks(range(len(top_ventas_campaña)))
            ax.set_yticklabels(top_ventas_campaña.index)
            ax.set_xlabel('Ventas Totales ($)')
            ax.set_title('Top 10 Productos Durante Campaña')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, val) in enumerate(zip(bars, top_ventas_campaña.values)):
                ax.text(val + max(top_ventas_campaña.values)*0.01, bar.get_y() + bar.get_height()/2, 
                       f'${val:,.0f}', va='center')
            
            st.pyplot(fig)
    
    # Resumen ejecutivo
    st.markdown("---")
    st.header("Resumen Ejecutivo")
    
    col13, col14 = st.columns(2)
    
    with col13:
        st.write("Resultados Clave:")
        
        conclusiones = []
        
        if p_valor < 0.05:
            if cambio_ventas > 0:
                conclusiones.append(f"Impacto positivo significativo: La campaña aumentó ventas en {cambio_ventas:+.1f}%")
            else:
                conclusiones.append(f"Impacto negativo significativo: La campaña redujo ventas en {abs(cambio_ventas):.1f}%")
        else:
            conclusiones.append("Sin impacto significativo: No hay evidencia estadística de cambio")
        
        if cambio_promedio > 10:
            conclusiones.append(f"Aumento en ticket promedio: Los clientes gastan {cambio_promedio:+.1f}% más por transacción")
        
        if cambio_transacciones > 10:
            conclusiones.append(f"Más transacciones: Volumen aumentó {cambio_transacciones:+.1f}%")
        
        if len(productos_campaña) > 0:
            mejor_producto = top_crecimiento.index[0] if len(top_crecimiento) > 0 else "N/A"
            if len(top_crecimiento) > 0:
                mejor_crecimiento = top_crecimiento['cambio'].iloc[0]
                conclusiones.append(f"Producto estrella: {mejor_producto} creció {mejor_crecimiento:+.1f}%")
        
        for conclusion in conclusiones:
            st.write(f"• {conclusion}")
    
    with col14:
        st.write("Recomendaciones:")
        
        recomendaciones = []
        
        if p_valor < 0.05 and cambio_ventas > 0:
            recomendaciones.append("Continuar campaña: Los resultados justifican continuar")
            recomendaciones.append("Optimizar elementos exitosos: Identificar qué funcionó mejor")
            recomendaciones.append("Expandir a otros productos: Aplicar estrategias similares")
        elif p_valor < 0.05 and cambio_ventas < 0:
            recomendaciones.append("Reevaluar estrategia: Revisar elementos de la campaña")
            recomendaciones.append("Realizar investigación: Entender por qué no funcionó")
            recomendaciones.append("Probar nuevos enfoques: Experimentar con diferentes tácticas")
        else:
            recomendaciones.append("Ampliar periodo de prueba: Dar más tiempo para ver efectos")
            recomendaciones.append("Aumentar presupuesto: Incrementar alcance de la campaña")
            recomendaciones.append("Segmentar mejor: Enfocarse en clientes más receptivos")
        
        for i, recomendacion in enumerate(recomendaciones, 1):
            st.write(f"{i}. {recomendacion}")

else:
    st.warning("No hay suficientes datos para el análisis comparativo")

# Pie de página
st.markdown("---")
st.write(f"Análisis completado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

