# ----- GF-0657: PORGRAMACIÓN EN SIG (2024) | PROFESOR: MANUEL VARGAS TAREA 03 | AARON BLANCO (B91088) -----


# ----- Carga y configuración de los paquetes -----

import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify
import rasterio
import numpy as np
import rasterio.plot
import folium
import branca

from io import BytesIO

from rasterio.mask import mask
from os import name
from matplotlib import colors
from folium import Choropleth, Popup, Tooltip, GeoJson, GeoJsonTooltip
from streamlit_folium import folium_static, st_folium
from folium.raster_layers import ImageOverlay
from branca.colormap import LinearColormap, linear

# Configuración de pandas para mostrar separadores de miles, 2 dígitos decimales y evitar la notación científica.
pd.set_option('display.float_format', '{:,.2f}'.format)


# ----- Fuentes de datos -----

#Datos de cantidad de botaderos por provincia
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_02/refs/heads/main/datos_entrada/rellenos_provincias.csv
datos_botaderos_provincias = 'datos/rellenos_provincias.csv'

#Datos de botaderos
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_02/823b56533d8ce881c0aa48ccc09a44440c97688b/datos_entrada/rellenos.csv
datos_botaderos = 'datos/rellenos.csv' 

#Datos de IDHD
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_02/refs/heads/main/datos_entrada/indice_desarrollo_humano_desigualdad.csv
datos_idhd = 'datos/indice_desarrollo_humano_desigualdad.csv'

#Datos espaciales de los botaderos
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_03/refs/heads/main/Datos_entrada/Botaderos.gpkg
datos_botaderos_gdf = 'datos/Botaderos.gpkg'

#Datos espaciales de las provincias
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_03/refs/heads/main/Datos_entrada/Provincias.gpkg
datos_provincias_gdf = 'datos/Provincias.gpkg'

#Datos espaciales de los cantones
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_03/refs/heads/main/Datos_entrada/Cantones.gpkg
datos_cantones_gdf = 'datos/Cantones.gpkg'

# #Datos espaciales del límite de Costa Rica
# #El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_03/refs/heads/main/Datos_entrada/limite_cr.gpkg
# datos_limite_gdf = 'datos/limite_cr.gpkg'

# Datos ráster de población
#El archivo original está en: https://raw.githubusercontent.com/Aaronblancod/tarea_03/refs/heads/main/Datos_entrada/cri_pop_2020.tif
datos_poblacion = 'datos/cri_pop_2020.tif'


# ----- Funciones para recuperar los datos -----


# Función para cargar los datos y almacenarlos en caché 
# para mejorar el rendimiento

# Función para cargar los datos csv en un dataframe de pandas
@st.cache_data
def cargar_botaderos_provincias():
    botaderos_provincias = pd.read_csv(datos_botaderos_provincias)
    return botaderos_provincias

@st.cache_data
def cargar_botaderos():
    botaderos = pd.read_csv(datos_botaderos)
    return botaderos

@st.cache_data
def cargar_idhd(): 
    idhd = pd.read_csv(datos_idhd)
    return idhd

# Función para cargar los datos geoespaciales en un geodataframe de geopandas
@st.cache_data
def cargar_botaderos_gdf():
    botaderos_gdf = gpd.read_file(datos_botaderos_gdf)
    return botaderos_gdf

@st.cache_data
def cargar_provincias_gdf():
    provincias_gdf = gpd.read_file(datos_provincias_gdf)
    return provincias_gdf

@st.cache_data
def cargar_cantones_gdf():
    cantones_gdf = gpd.read_file(datos_cantones_gdf)
    return cantones_gdf

# @st.cache_data
# def cargar_limite_gdf():
#     limite_gdf = gpd.read_file(datos_limite_gdf)
#     return limite_gdf

# Función para cargar los datos raster con rasterio
@st.cache_resource
def cargar_poblacion():
    poblacion = rasterio.open(datos_poblacion)
    return poblacion


# ----- TÍTULO DE LA APLICACIÓN -----
st.title('Gestión y problemáticas en torno a los residuos sólidos en Costa Rica')


# ----- Carga de datos -----

# Cargar datos de botaderos
estado_carga_botaderos = st.text('Cargando datos de los botaderos...')
botaderos = cargar_botaderos()
estado_carga_botaderos.text('Los datos de los botaderos fueron cargados.')

# Cargar datos de botaderos por provincia
estado_carga_botaderos_provincias = st.text('Cargando datos de los botaderos por provincia...')
botaderos_provincias = cargar_botaderos_provincias()
estado_carga_botaderos_provincias.text('Los datos de los botaderos por provincias fueron cargados.')

# Cargar datos del Indice de Desarrollo Humano Cantonal ajustado por Desigualdad (IDHD)
estado_carga_idhd= st.text('Cargando datos del Indice de Desarrollo Humano Cantonal ajustado por Desigualdad (IDHD)...')
idhd = cargar_idhd()
estado_carga_idhd.text('Los datos del Indice de Desarrollo Humano Cantonal ajustado por Desigualdad (IDHD) fueron cargados.')

# Cargar datos geoespcailes de botaderos
estado_carga_botaderos_gdf = st.text('Cargando datos geoespaciales de los botaderos...')
botaderos_gdf = cargar_botaderos_gdf()
estado_carga_botaderos_gdf.text('Los datos geoespaciales de los botaderos fueron cargados.')

# Cargar datos geoespcailes de las provicnias
estado_carga_provincias_gdf = st.text('Cargando datos geoespaciales de las provincias...')
provincias_gdf = cargar_provincias_gdf()
estado_carga_provincias_gdf.text('Los datos geoespaciales de las provincias fueron cargados.')

# Cargar datos geoespcailes de los cantones
estado_carga_cantones_gdf = st.text('Cargando datos geoespaciales de los cantones...')
cantones_gdf = cargar_cantones_gdf()
estado_carga_cantones_gdf.text('Los datos geoespaciales de los cantones fueron cargados.')

# # Cargar datos geoespcailes del límite de Costa Rica
# estado_carga_limite_gdf = st.text('Cargando datos geoespaciales de los límites...')
# limite_gdf = cargar_limite_gdf()
# estado_carga_limite_gdf.text('Los datos geoespaciales de los límites fueron cargados.')

# Cargar datos del raster de poblacion
estado_carga_poblacion = st.text('Cargando datos sobre la población de Costa Rica del WorldPop...')
poblacion = cargar_poblacion()
estado_carga_poblacion.text('Los datos sobre la población de Costa Rica del WorldPop fueron cargados.')


# ----- Preparación de datos -----


# ----- Preparación de dataframes -----

# Establecimiento de los indices para los dataframes
# Se usa la columna provincia o canton como índice
botaderos.set_index('canton', inplace=True)
idhd.set_index('canton', inplace=True)

# Unión de los dataframes botaderos y idhd
botaderos_idhd = botaderos.join(idhd, how="left")

# Restablecer el índice
botaderos.reset_index(inplace=True)
idhd.reset_index(inplace=True)
botaderos_idhd.reset_index(inplace=True)

# Filtración de las columnas relevantes del conjunto de datos de botaderos
columnas = [
    'id',
    'provincia',
    'canton', 
    'tipo', 
    'X', 
    'Y'
]
botaderos = botaderos[columnas]

# Filtración de las columnas relevantes del conjunto de datos de botaderos
columnas_idhd = [
    'cod_provin',
    'provincia',
    'canton', 
    'tipo', 
    '2010',
    '2011',
    '2012',
    '2013',
    '2014',
    '2015',
    '2016',
    '2017',
    '2018',
    '2019',
    '2020',
    'X',
    'Y'
]
botaderos_idhd = botaderos_idhd[columnas_idhd]

# ----- procesamiento de los datos según la selección -----
# Obtener la lista de cantones con IDHD según provincia
lista_provincias = botaderos_idhd['provincia'].unique().tolist()
lista_provincias.sort()

# Añadir la opción "Todos" al inicio de la lista
opciones_provincias = ['Todos'] + lista_provincias

# Crear el selectbox en la barra lateral
provincia_seleccionado = st.sidebar.selectbox(
    'Selecciona una provincia',
    opciones_provincias
)

# ----- Filtrar datos según la selección -----

if provincia_seleccionado != 'Todos':
    # Filtrar los datos para el provincia seleccionado
    botaderos_idhd_filtrados = botaderos_idhd[botaderos_idhd['provincia'] == provincia_seleccionado]
    
    # Obtener el Código del cantón seleccionado
    codigo_seleccionado = botaderos_idhd_filtrados['cod_provin'].iloc[0]
else:
    # No aplicar filtro
    botaderos_idhd_filtrados = botaderos_idhd.copy()
    codigo_seleccionado = None


# ----- Preparación de geodataframes -----

# Revisión de que ambos GeoDataFrames estén en el mismo CRS
botaderos_gdf = botaderos_gdf.to_crs("EPSG:4326")
provincias_gdf = provincias_gdf.to_crs("EPSG:4326")

# Reproyección de ambos GeoDataFrames a CRTM05 (EPSG:5367)
botaderos_gdf = botaderos_gdf.to_crs("EPSG:5367")
provincias_gdf = provincias_gdf.to_crs("EPSG:5367")

# Calculo del área de cada provincia en kilómetros cuadrados
provincias_gdf["area_km2"] = provincias_gdf.geometry.area / 1e6  # Área en m² a km²

# Realización del spatial join asegurando que se preserva la columna "provincia"
botaderos_por_provincia = gpd.sjoin(
    botaderos_gdf,
    provincias_gdf,
    how="inner",
    predicate="within",
    lsuffix="_left",
    rsuffix="_right"  # Para evitar conflictos de nombres
)

# Conteo del número de botaderos por provincia
conteo_botaderos = (
    botaderos_por_provincia.groupby("provincia__right")  # Nota: Nombre ajustado según el join
    .size()
    .reset_index(name="num_botaderos")
)

# Unión del conteo al GeoDataFrame de provincias
provincias_botaderos_gdf = provincias_gdf.merge(
    conteo_botaderos,
    left_on="provincia",
    right_on="provincia__right",
    how="left"
)

# Relleno de valores nulos (provincias sin botaderos) con 0
provincias_botaderos_gdf["num_botaderos"] = provincias_botaderos_gdf["num_botaderos"].fillna(0)

# Calculo de la densidad de botaderos por km²
provincias_botaderos_gdf["densidad_botaderos"] = (
    provincias_botaderos_gdf["num_botaderos"] / provincias_botaderos_gdf["area_km2"]
)

# ----- procesamiento de los datos geoespaciales según la selección -----
# Unir los datos del IDHD con el GeoDataFrame de cantones
cantones_gdf_merged = cantones_gdf.merge(
    botaderos_idhd_filtrados, 
    how='inner', 
    left_on='canton', 
    right_on='canton'
)

# Reemplazar valores nulos por cero en las columnas con IDHD
# Lista de las columnas que deseas modificar
anios1 = [str(anio1) for anio1 in range(2010, 2021)]  # Crea la lista ['2010', '2011', ..., '2020']

# Reemplazar valores nulos por 0 en estas columnas
cantones_gdf_merged[anios1] = cantones_gdf_merged[anios1].fillna(0)

# Cambio de CRS
cantones_gdf_merged = cantones_gdf_merged.to_crs(epsg=4326) #Cambio


#Calculo de la media del IDHD
columnas_idhd = anios1

# Calcular la media del IDHD y agregarla como nueva columna
cantones_gdf_merged['media_IDHD'] = cantones_gdf_merged[columnas_idhd].mean(axis=1)

# Unir los datos de la localización de botaderos con el dataframe filtrado
botaderos_gdf_merged = botaderos_gdf.merge(
    botaderos_idhd_filtrados, 
    how='inner', 
    left_on='canton', 
    right_on='canton'
)

# Filtración de las columnas relevantes del conjunto de datos de botaderos filtrado
columnas_bf = [
    'canton',
    'geometry',
    'tipo_x'
]
botaderos_gdf_merged = botaderos_gdf_merged[columnas_bf]
botaderos_gdf_merged = botaderos_gdf_merged.rename(columns={'tipo_x': 'tipo de botadero'})

# Unir los datos de la provincia con el dataframe filtrado
provincias_gdf_merged = provincias_gdf.merge(
    botaderos_idhd_filtrados, 
    how='inner', 
    left_on='provincia', 
    right_on='provincia'
)

# Filtración de las columnas relevantes del conjunto de datos de provincias filtrado
columnas_provin = [
    'provincia',
    'geometry'
]
provincias_gdf_merged = provincias_gdf_merged[columnas_provin]


# ----- Preparación de los datos raster -----

# Lectura del ráster
out_image = poblacion.read(1)  # Leer la primera banda
out_transform = poblacion.transform  # Obtener la transformación

# Filtrar los valores menores o iguales a 0 y enmascararlos
out_image[out_image <= 0] = np.nan  # Enmascarar los valores menores o iguales a 0 para excluirlos

# Normalizar los valores restantes del ráster para visualización
raster_normalized = (out_image - np.nanmin(out_image)) / (np.nanmax(out_image) - np.nanmin(out_image))

# Obtener los límites geográficos del ráster
bounds = rasterio.transform.array_bounds(out_image.shape[0], out_image.shape[1], out_transform)
extent = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]  # Formato [lat_min, lon_min, lat_max, lon_max]

# Crear una paleta de colores personalizada (escala de naranjas)
cmap = colors.LinearSegmentedColormap.from_list("purple_scale", ["#F7E1FF", "#C084F5", "#7A23E0", "#3E007A"])
raster_colormap = cmap(raster_normalized)  # Aplicar la paleta al ráster normalizado

# Crear una leyenda personalizada para el ráster
from branca.colormap import LinearColormap
colormap = LinearColormap(
    colors=["#F7E1FF", "#C084F5", "#7A23E0", "#3E007A"],
    vmin=np.nanmin(out_image),  # Mínimo de los datos válidos
    vmax=np.nanmax(out_image),  # Máximo de los datos válidos
    caption="Población Total (Normalizada)"
)

# Agregar el ráster de población al mapa como una superposición
image_overlay = ImageOverlay(
    image=raster_colormap,
    bounds=extent,
    opacity=0.4,
    interactive=True,
    name="Población Total"
)


# ----- TABLAS -----


# ----- Tabla de botaderos registrados de Costa Rica con tipo de gestión y localización -----

# Mostrar la tabla
st.subheader('Localización y tipo de gestión de los botaderos registrados a escala nacional')
st.dataframe(botaderos, hide_index=True)


# ----- GRÁFICOS -----


# ----- Gráfico de pastel de la distribución del número de botaderos por provincia -----

# Ordenar por número de botaderos en orden descendente
botaderos_provincias_sorted = botaderos_provincias.sort_values(by='botaderos', ascending=False)

# Definir colores en tonalidades de café
cafes = ['#4E2C0A', '#8B4513', '#A0522D', '#C68642', '#DEB887', '#F4A460', '#7B3F00']

# Crear el gráfico de pastel con colores personalizados
fig1 = px.pie(botaderos_provincias_sorted,
              names='provincia',
              values='botaderos',
              #title="Distribución del Número de Botaderos por Provincia",
              labels={'botaderos': 'Número de Botaderos', 'provincia': 'Provincia'},
              color_discrete_sequence=cafes
              )

# Agregar título a la leyenda
fig1.update_layout(legend_title_text='Provincia')

# Mostrar el gráfico
st.subheader('Distribución porcentual de los botaderos según provincia')
st.plotly_chart(fig1)


# ----- Gráfico de barras de la cantidad y tipo de botadero según provincia -----

# Contabilizar la cantidad de cada tipo de botadero por provincia
tipo_botadero_provincia_conteo = botaderos.groupby(['provincia', 'tipo']).size().reset_index(name='cantidad')

# Calcular la cantidad total de botaderos por provincia
total_botaderos_provincia = tipo_botadero_provincia_conteo.groupby('provincia')['cantidad'].sum().reset_index()

# Unir el conteo de tipos con el total para ordenar
tipo_botadero_provincia_conteo = tipo_botadero_provincia_conteo.merge(total_botaderos_provincia, on='provincia', suffixes=('', '_total'))

# Ordenar el DataFrame completo por la cantidad total de botaderos en orden descendente
tipo_botadero_provincia_conteo = tipo_botadero_provincia_conteo.sort_values(by='cantidad_total', ascending=False)

# Configurar el orden de las provincias en el gráfico
ordered_provinces = tipo_botadero_provincia_conteo['provincia'].unique()

# Definir colores de tonalidades de café
cafes = ['#4E2C0A', '#DEB887', '#7B3F00', '#C68642']

# Crear el gráfico de barras en Plotly con la paleta de colores de café
fig2 = px.bar(tipo_botadero_provincia_conteo,
             x='provincia',
             y='cantidad',
             color='tipo',
             #title='Cantidad de cada tipo de botadero de residuos por provincia',
             labels={'provincia': 'Provincia', 'cantidad': 'Cantidad de botaderos', 'tipo': 'Tipo de botadero'},
             color_discrete_sequence=cafes
             )

# Aplicar el orden personalizado al eje x
fig2.update_xaxes(categoryorder='array', categoryarray=ordered_provinces)

# Mostrar el gráfico
st.subheader('Cantidad y tipo de botadero según provincia')
st.plotly_chart(fig2)


# ----- MAPA -----

# Crear el mapa base con controles de zoom desactivados
base_map = folium.Map(
    location=[10, -84],  # Centro del mapa
    zoom_start=8,
    zoomControl=False  # Desactiva los controles de zoom
)

# Agregar la capa de provincias con densidad de botaderos (coropletas)
folium.Choropleth(
    geo_data=provincias_botaderos_gdf,
    data=provincias_botaderos_gdf,
    columns=['provincia', 'densidad_botaderos'],
    key_on='feature.properties.provincia',
    fill_color='Reds',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Densidad de botaderos por provincia"
).add_to(base_map)

# Agregar los registros de la localización de los botaderos
for _, row in botaderos_gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color='red',
        fill=True,
        fill_opacity=0.8,
        tooltip=folium.Tooltip(f"Cantón: {row['canton']}<br>Tipo: {row['tipo']}"),
    ).add_to(base_map)

# Agregar la capa ráster y la leyenda al mapa
image_overlay.add_to(base_map)
colormap.add_to(base_map)

# Añadir capa base de Esri Satellite
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(base_map)

# Agregar un control de capas al mapa
folium.LayerControl().add_to(base_map)

# Mostrar el mapa
st.subheader('Relación entre la distribución de la población y la gestión de residuos a escala nacional')
st_folium(base_map, width=400, height=600)


# ----- Sección interactiva -----

#st.title('Selección de datos')
st.subheader('Datos seleccionables sobre la gestión de residuos y su relación con el Índice de Desarrollo Humano ajustado por Desigualdad (IDHD) según provincia')




# ----- Tabla de la selección -----

# Mostrar la tabla
st.subheader('Datos sobre gestión y su relación con el IDHD según provincia')
st.dataframe(botaderos_idhd_filtrados, hide_index=True)


# ----- Gráfico de líneas de la evolución del IDHD en cada cantón que contiene al menos un botadero -----

# Seleccionar las columnas de años
anios = botaderos_idhd_filtrados.columns[3:15]

# Transformar los datos a formato largo
botaderos_idhd_largo = botaderos_idhd_filtrados.melt(
    id_vars=['canton', 'provincia'],
    value_vars=anios,
    var_name='Año',
    value_name='IDHD'
)

# Crear el gráfico de líneas
fig3 = px.line(
    botaderos_idhd_largo,
    x='Año',
    y='IDHD',
    color='canton',
    markers=True,
    #title='Evolución del Índice de Desarrollo Humano ajustado por Desigualdad entre 2010 y 2020 por cantón',
    labels={
        'Año': 'Año',
        'IDHD': 'IDHD',
        'canton': 'Cantón'
    },
    hover_data={
        'canton': True,
        'Año': True,
        'IDHD': ':.4f'
    }
)

# Atributos globales y configurar leyenda
fig3.update_layout(
    #title={'text': 'Evolución del Índice de Desarrollo Humano ajustado por Desigualdad entre 2010 y 2020 por cantón', 'x': 0.5},
    legend_title_text='Cantón',
    xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
)

# Mostrar el gráfico
st.subheader('Evolución del IDHD entre 2010 y 2020 por cantón según provincia')
st.plotly_chart(fig3)

# ----- MAPA SELECCIONADO -----


# ----- Mapa interactivo: Cantones con presencia de botaderos y su relación con el IDHD -----

# Crear una paleta de colores personalizada (rojo a verde)
colormap = LinearColormap(
    colors=['red', 'yellow', 'green'],  # Colores desde rojo a verde
    vmin=cantones_gdf_merged['media_IDHD'].min(),  # Valor mínimo
    vmax=cantones_gdf_merged['media_IDHD'].max()   # Valor máximo
)

# Centro del mapa basado en el GeoDataFrame
centro = [cantones_gdf_merged.geometry.centroid.y.mean(), cantones_gdf_merged.geometry.centroid.x.mean()]

# Crear el mapa base
mapa = folium.Map(
    location=centro,
    zoom_start=9,
    zoomControl=True  # Mantén los controles de zoom para una mejor usabilidad
)

# Agregar la capa de coropletas de cantones
GeoJson(
    cantones_gdf_merged,
    name="Cantones con la Media IDHD (2010-2020)",
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['media_IDHD']),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7,
    },
    tooltip=GeoJsonTooltip(
        fields=["canton", "media_IDHD"],
        aliases=["Cantón:", "Media IDHD:"],
        localize=True
    )
).add_to(mapa)

# Agregar la leyenda de la paleta de colores
colormap.caption = "Media del IDHD (2010-2020)"
colormap.add_to(mapa)

# Agregar una capa base de Esri Satellite
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(mapa)

# Agregar marcadores para los botaderos
for _, row in botaderos_gdf_merged.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color='white',
        fill=True,
        fill_opacity=0.8,
        tooltip=f"Cantón: {row['canton']}<br>Tipo de botadero: {row['tipo de botadero']}"
    ).add_to(mapa)

# Agregar la capa de contorno de provincias
#GeoJson(
#    provincias_gdf_merged,
#    name="Provincias",
#    style_function=lambda feature: {
#        'fillColor': 'none',  # Sin relleno
#        'color': 'black',      # Color del borde
#        'weight': 1,
#        'fillOpacity': 0
#    },
#    tooltip=GeoJsonTooltip(
#        fields=["provincia"],
#        aliases=["Provincia:"],
#        localize=True
#    )
#).add_to(mapa)

# Agregar un control de capas
folium.LayerControl().add_to(mapa)

# Mostrar el mapa con st_folium
st.subheader('Relación entre la presencia de botaderos y el promedio del IDHD entre 2010 y 2020 por cantón según provincia')
st_folium(mapa, width=400, height=600)
