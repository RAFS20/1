import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import io
import base64
import re
import smtplib, ssl
from email.message import EmailMessage
import urllib.parse
import os
import json  # Para manejo seguro de JSON

def clean_column_names(cols):
    normalized = []
    for c in cols:
        c = c.strip()
        c = re.sub(r"[ÁÀÂÄ]", "A", c, flags=re.IGNORECASE)
        c = re.sub(r"[ÉÈÊË]", "E", c, flags=re.IGNORECASE)
        c = re.sub(r"[ÍÌÎÏ]", "I", c, flags=re.IGNORECASE)
        c = re.sub(r"[ÓÒÔÖ]", "O", c, flags=re.IGNORECASE)
        c = re.sub(r"[ÚÙÛÜ]", "U", c, flags=re.IGNORECASE)
        c = re.sub(r"Ñ", "N", c, flags=re.IGNORECASE)
        c = c.lower()
        c = re.sub(r"[^0-9a-z_]+", "_", c)
        normalized.append(c)
    return normalized

def load_data(file_path="datos.csv"):
    try:
        data = pd.read_csv(file_path, dtype=str)
        data.columns = clean_column_names(data.columns)

        required_cols = ["date", "fecha_inicio_del_programa", "channel_name", "program_name",
                         "start_time", "end_time", "share", "macrogenero", "genero",
                         "subgenero", "share_predicho", "palabras_clave", "escala_likert"]
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError("Faltan columnas: " + ", ".join(missing_cols))

        if "descripcion" not in data.columns:
            data["descripcion"] = ""

        def try_parse_date(series):
            parsed = pd.to_datetime(series, format="%Y-%m-%d", errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
            return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)

        data["date"] = try_parse_date(data["date"])
        data["fecha_inicio_del_programa"] = try_parse_date(data["fecha_inicio_del_programa"])

        data["share"] = pd.to_numeric(data["share"], errors='coerce')
        data["share_predicho"] = pd.to_numeric(data["share_predicho"], errors='coerce')
        data["escala_likert"] = pd.to_numeric(data["escala_likert"], errors='coerce')

        data["channel_name"] = data["channel_name"].str.upper()

        data = data.drop_duplicates(subset=["date", "start_time", "end_time", "channel_name", "program_name"])

        data = data[(~data["date"].isna()) & (~data["fecha_inicio_del_programa"].isna())]

        return data
    except Exception as e:
        print("Error al leer el CSV:", e)
        return pd.DataFrame()

# Cargar datos iniciales
datos_iniciales = load_data()

# Definir rango de fechas
if not datos_iniciales.empty:
    min_date = min(datos_iniciales["date"].min(), datos_iniciales["fecha_inicio_del_programa"].min())
    if pd.isnull(min_date):
        min_date = datetime.date(2000, 1, 1)
    else:
        min_date = min_date.date()
    max_date = datetime.date.today() + datetime.timedelta(days=365)
else:
    min_date = datetime.date(2000, 1, 1)
    max_date = datetime.date.today() + datetime.timedelta(days=365)

# Opciones de canales
canales_disponibles = datos_iniciales["channel_name"].unique().tolist() if not datos_iniciales.empty else []
canales_disponibles_sorted = sorted(canales_disponibles)
canal_default = "LA 1" if "LA 1" in canales_disponibles else (canales_disponibles_sorted[0] if canales_disponibles else "Todos")

def filtrar_datos(fecha, canal, programa):
    df = datos_iniciales.copy()
    if df.empty:
        return df
    if canal != "Todos":
        df = df[df["channel_name"] == canal]
    if programa and programa != "Todos":
        df = df[df["program_name"] == programa]
    if fecha:
        fecha_dt = pd.to_datetime(fecha)
        df = df[df["date"] == fecha_dt]
    return df

def filtrar_para_graficos(canal, programa):
    df = datos_iniciales.copy()
    if df.empty:
        return df
    if canal != "Todos":
        df = df[df["channel_name"] == canal]
    if programa and programa != "Todos":
        df = df[df["program_name"] == programa]
    return df

def remove_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    return series[(series >= lower) & (series <= upper)]

external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css',
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# Definir el número de escenarios
NUM_ESCENARIOS = 4

def crear_escenario(n):
    return html.Div(style={
        "border":"1px solid #ccc",
        "padding":"20px",
        "borderRadius":"5px",
        "backgroundColor":"#ffffff",
        "boxShadow":"0 2px 5px rgba(0,0,0,0.1)"
    }, children=[
        html.H4(f"Escenario {n}", style={"color":"#2c3e50","fontWeight":"500"}),
        dash_table.DataTable(
            id=f"tablaEditable_{n}",
            columns=[],
            data=[],
            editable=True,
            page_size=10,
            style_table={
                'overflowX': 'auto', 
                'backgroundColor':'#ffffff',
                'border':'1px solid #ccc',
                'borderRadius':'5px',
                'boxShadow':'0 2px 5px rgba(0,0,0,0.1)'
            },
            style_cell={'fontFamily':'Roboto','fontSize':'14px','textAlign':'left','padding':'5px'},
            style_header={'fontWeight':'500','backgroundColor':'#ecf0f1','borderBottom':'2px solid #ccc'},
        ),
        html.Button(f"Ejecutar Predicción", id=f"btnEjecutar_{n}", n_clicks=0, 
                    style={
                        "backgroundColor":"#2980b9",
                        "color":"white",
                        "border":"none",
                        "padding":"10px",
                        "cursor":"pointer",
                        "borderRadius":"3px",
                        "width":"100%",
                        "fontWeight":"500",
                        "marginTop":"10px"
                    }),
        # Área de feedback específica para cada escenario
        html.Div(id=f"mensajeFeedback_{n}", style={"marginTop":"10px", "textAlign":"center", "fontWeight":"500", "color":"#e74c3c"})
    ])

app.layout = html.Div(style={
    "margin": "0",
    "padding": "20px",
    "backgroundColor":"#f7f7f7",
    "fontFamily":"Roboto"
}, children=[
    html.H1("Panel Analista", style={
        "textAlign":"center", 
        "color":"#2c3e50",
        "marginBottom":"30px",
        "fontWeight":"700"
    }),
    # Filtros Globales
    html.Div(style={"marginBottom":"30px"}, children=[
        html.Div(style={"display":"flex", "gap":"20px", "flexWrap":"wrap", "justifyContent":"center"}, children=[
            html.Div(style={"width":"250px"}, children=[
                html.Label("Seleccionar Fecha:", style={"color":"#34495e","marginTop":"10px"}),
                dcc.DatePickerSingle(
                    id='fecha',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    date=datetime.date.today(),
                    display_format='DD/MM/YYYY',
                    style={"marginBottom":"20px", "marginTop":"5px"}
                ),
            ]),
            html.Div(style={"width":"250px"}, children=[
                html.Label("Seleccionar Canal:", style={"color":"#34495e"}),
                dcc.Dropdown(
                    id="canal",
                    options=[{"label":"Todos","value":"Todos"}] + [{"label":c,"value":c} for c in canales_disponibles_sorted],
                    value=canal_default,
                    style={"marginBottom":"20px", "marginTop":"5px"}
                ),
            ]),
            html.Div(style={"width":"250px"}, children=[
                html.Label("Seleccionar Programa:", style={"color":"#34495e"}),
                dcc.Dropdown(
                    id="programa",
                    options=[{"label":"Todos","value":"Todos"}],
                    value="Todos",
                    style={"marginBottom":"20px", "marginTop":"5px"}
                ),
            ]),
            html.Div(style={"width":"250px"}, children=[
                html.Label("Correo del Destinatario:", style={"color":"#34495e"}),
                dcc.Input(id="emailRecipient", type="email", placeholder="ejemplo@dominio.com", 
                          value="rafernandezsalguero@mediapro.tv",
                          style={"width":"100%","marginBottom":"20px", "marginTop":"5px","padding":"5px","border":"1px solid #ccc","borderRadius":"3px"}),
            ]),
            html.Div(style={"width":"250px"}, children=[
                html.Label("Escenario:", style={"color":"#34495e"}),
                dcc.Input(id="escenario", type="text", placeholder="Nombre del escenario",
                          value="resultados",
                          style={"width":"100%","marginBottom":"20px", "marginTop":"5px","padding":"5px","border":"1px solid #ccc","borderRadius":"3px"}),
            ]),
            # Añadir el componente 'descargarDatos' en los filtros globales
            html.Div(style={"width":"250px"}, children=[
                html.Label("Descargar Datos:", style={"color":"#34495e"}),
                html.A("Descargar Datos", id="descargarDatos", download="", href="#", target="_blank",
                       style={
                           "display":"inline-block",
                           "marginTop":"5px",
                           "color":"white",
                           "backgroundColor":"#27ae60",
                           "padding":"10px",
                           "textDecoration":"none",
                           "borderRadius":"3px",
                           "textAlign":"center",
                           "fontWeight":"500"
                        })
            ]),
        ]),
    ]),
    # Malla de Escenarios 2x2
    html.Div(style={"display":"grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap":"20px"}, children=[
        crear_escenario(n) for n in range(1, NUM_ESCENARIOS + 1)
    ]),
    # Sección de Gráficos y Tabla General
    html.Div(style={"marginTop":"40px"}, children=[
        html.H4("Gráficos y Tabla General", style={"marginBottom":"20px","color":"#2c3e50","fontWeight":"500"}),
        html.Div(style={"display":"flex","flexWrap":"wrap","gap":"20px"}, children=[
            html.Div(style={"flex":"1 1 45%"}, children=[
                html.H4("SHARE_predicho Promedio Diario", style={"color":"#2c3e50","fontWeight":"500"}),
                html.Pre(id="shareDia", style={
                    "backgroundColor":"#ffffff",
                    "padding":"15px",
                    "borderRadius":"5px",
                    "boxShadow":"0 2px 5px rgba(0,0,0,0.1)",
                    "fontWeight":"500",
                    "fontSize":"16px",
                    "textAlign":"center",
                    "color":"#16a085"
                })
            ]),
            html.Div(style={"flex":"1 1 45%"}, children=[
                html.H4("SHARE_predicho Promedio Mensual", style={"color":"#2c3e50","fontWeight":"500"}),
                html.Pre(id="shareMes", style={
                    "backgroundColor":"#ffffff",
                    "padding":"15px",
                    "borderRadius":"5px",
                    "boxShadow":"0 2px 5px rgba(0,0,0,0.1)",
                    "fontWeight":"500",
                    "fontSize":"16px",
                    "textAlign":"center",
                    "color":"#8e44ad"
                })
            ]),
        ]),
        html.H4("Tabla de Datos (Editable)", style={"marginTop":"20px","color":"#2c3e50","fontWeight":"500"}),
        dash_table.DataTable(
            id="tablaEditableGeneral",
            columns=[],
            data=[],
            editable=True,
            page_size=20,
            style_table={
                'overflowX': 'auto', 
                'backgroundColor':'#ffffff',
                'border':'1px solid #ccc',
                'borderRadius':'5px',
                'boxShadow':'0 2px 5px rgba(0,0,0,0.1)'
            },
            style_cell={'fontFamily':'Roboto','fontSize':'14px','textAlign':'left','padding':'5px'},
            style_header={'fontWeight':'500','backgroundColor':'#ecf0f1','borderBottom':'2px solid #ccc'},
        ),
        html.H4("Gráfico de Líneas (Histórico Filtrado por Canal/Programa)", style={"marginTop":"20px","color":"#2c3e50","fontWeight":"500"}),
        dcc.Graph(id="graficoLineas", figure={}, style={"backgroundColor":"#ffffff","border":"1px solid #ccc","borderRadius":"5px","padding":"10px","boxShadow":"0 2px 5px rgba(0,0,0,0.1)"}),
        html.H4("Histograma Global de SHARE (Filtrado y sin outliers)", style={"marginTop":"20px","color":"#2c3e50","fontWeight":"500"}),
        dcc.Graph(id="histogramaGlobal", figure={}, style={"backgroundColor":"#ffffff","border":"1px solid #ccc","borderRadius":"5px","padding":"10px","boxShadow":"0 2px 5px rgba(0,0,0,0.1)"}),
        html.Div(id="original_predicho", style={"display":"none"})
    ]),
    # Mensajes de Feedback para cada escenario
    html.Div(style={"marginTop":"20px"}, children=[
        html.Div(id=f"mensajeFeedback_{n}", style={"marginBottom":"10px", "textAlign":"center", "fontWeight":"500"}) for n in range(1, NUM_ESCENARIOS + 1)
    ])
])

# Callback para actualizar las opciones del dropdown 'programa' basado en 'canal'
@app.callback(
    Output("programa", "options"),
    Input("canal", "value")
)
def actualizar_programas(canal):
    if datos_iniciales.empty:
        return [{"label":"Todos","value":"Todos"}]
    df = datos_iniciales.copy()
    if canal != "Todos":
        df = df[df["channel_name"] == canal]
    programas = df["program_name"].unique().tolist()
    return [{"label":"Todos","value":"Todos"}] + [{"label":p,"value":p} for p in sorted(programas)]

# Callback para actualizar las tablas de los escenarios cuando cambian los filtros globales
@app.callback(
    [
        Output(f"tablaEditable_{n}", "data") for n in range(1, NUM_ESCENARIOS +1)
    ] + [
        Output(f"tablaEditable_{n}", "columns") for n in range(1, NUM_ESCENARIOS +1)
    ],
    [
        Input("fecha", "date"),
        Input("canal", "value"),
        Input("programa", "value"),
    ]
)
def actualizar_tablas_escenarios(fecha, canal, programa):
    df_filtrado = filtrar_datos(fecha, canal, programa)
    if df_filtrado.empty:
        return [[], []] * NUM_ESCENARIOS
    else:
        columns = [
            {"name": "Program Name", "id": "program_name", "type":"text"},
            {"name": "Start Time", "id": "start_time", "type":"text"},
            {"name": "End Time", "id": "end_time", "type":"text"},
            {"name": "Descripción", "id": "descripcion", "type":"text"},
            {"name": "Escala Likert", "id": "escala_likert", "type":"numeric"},
            {"name": "Palabras Clave", "id": "palabras_clave", "type":"text"},
            {"name": "Share_predicho", "id": "share_predicho", "type":"numeric", "format":dict(specifier=".2f")},
        ]
        data = df_filtrado.to_dict('records')
        return [data for _ in range(NUM_ESCENARIOS)] + [columns for _ in range(NUM_ESCENARIOS)]

# Callback para actualizar la tabla general, gráficos y otros componentes
@app.callback(
    [
        Output("shareDia", "children"),
        Output("shareMes", "children"),
        Output("tablaEditableGeneral", "data"),
        Output("tablaEditableGeneral", "columns"),
        Output("tablaEditableGeneral", "style_data_conditional"),
        Output("graficoLineas", "figure"),
        Output("histogramaGlobal", "figure"),
        Output("descargarDatos", "href"),
        Output("original_predicho", "children")
    ],
    [
        Input("fecha", "date"),
        Input("canal", "value"),
        Input("programa", "value"),
        Input("tablaEditableGeneral", "data_timestamp")
    ],
    [
        State("tablaEditableGeneral", "data"),
        State("original_predicho", "children")
    ]
)
def actualizar_contenido(fecha, canal, programa, ts, tabla_data, original_predicho_json):
    df_filtrado = filtrar_datos(fecha, canal, programa)
    if not df_filtrado.empty:
        # Obtener la lista original de predicciones la primera vez
        if not original_predicho_json:
            original_predicho = df_filtrado["share_predicho"].tolist()
        else:
            try:
                original_predicho = json.loads(original_predicho_json)
            except json.JSONDecodeError:
                original_predicho = df_filtrado["share_predicho"].tolist()

        # Si el usuario ha editado la tabla
        if tabla_data and len(tabla_data) == len(df_filtrado):
            df_temp = df_filtrado.copy().reset_index(drop=True)
            df_tabla = pd.DataFrame(tabla_data)

            # Ajustar share_predicho si se cambia la descripción o palabras_clave
            for i in range(len(df_tabla)):
                old_desc = df_temp.loc[i, "descripcion"]
                new_desc = df_tabla.loc[i, "descripcion"]
                old_palabras = df_temp.loc[i, "palabras_clave"]
                new_palabras = df_tabla.loc[i, "palabras_clave"]
                if old_desc != new_desc or old_palabras != new_palabras:
                    original = original_predicho[i]
                    nuevo_valor = original * np.random.uniform(0.98, 1.02)  # Ajuste ±2%
                    df_temp.loc[i, "share_predicho"] = nuevo_valor

            df_temp["descripcion"] = df_tabla["descripcion"]
            df_temp["palabras_clave"] = df_tabla["palabras_clave"]
            df_filtrado = df_temp
    else:
        original_predicho = []

    if df_filtrado.empty:
        share_dia_text = "No hay datos"
        share_mes_text = "No hay datos"
    else:
        val_dia = df_filtrado["share_predicho"].mean(skipna=True)
        share_dia_text = f"{val_dia:.4f}" if pd.notnull(val_dia) else "No hay datos"

        if not datos_iniciales.empty and fecha:
            fecha_dt = pd.to_datetime(fecha)
            df_mes = datos_iniciales[
                (datos_iniciales["date"].dt.year == fecha_dt.year) &
                (datos_iniciales["date"].dt.month == fecha_dt.month)
            ]
            if df_mes.empty:
                share_mes_text = "No hay datos"
            else:
                val_mes = df_mes["share_predicho"].mean(skipna=True)
                share_mes_text = f"{val_mes:.4f}" if pd.notnull(val_mes) else "No hay datos"
        else:
            share_mes_text = "No hay datos"

    if df_filtrado.empty:
        data = []
        columns = []
        style_data_conditional = []
    else:
        data = df_filtrado.to_dict('records')

        for row in data:
            if isinstance(row["date"], pd.Timestamp) or isinstance(row["date"], datetime.date):
                row["date"] = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")

            if isinstance(row["fecha_inicio_del_programa"], pd.Timestamp) or isinstance(row["fecha_inicio_del_programa"], datetime.date):
                row["fecha_inicio_del_programa"] = pd.to_datetime(row["fecha_inicio_del_programa"]).strftime("%Y-%m-%d")

        columns = [
            {"name": "Program Name", "id": "program_name", "type":"text"},
            {"name": "Start Time", "id": "start_time", "type":"text"},
            {"name": "End Time", "id": "end_time", "type":"text"},
            {"name": "Descripción", "id": "descripcion", "type":"text"},
            {"name": "Escala Likert", "id": "escala_likert", "type":"numeric"},
            {"name": "Palabras Clave", "id": "palabras_clave", "type":"text"},
            {"name": "Share_predicho", "id": "share_predicho", "type":"numeric", "format":dict(specifier=".2f")},
        ]

        style_data_conditional = []
        if share_dia_text != "No hay datos":
            try:
                share_dia_num = float(share_dia_text)
                style_data_conditional = [
                    {
                        'if': {
                            'filter_query': f'{{share_predicho}} > {share_dia_num}'
                        },
                        'backgroundColor': 'rgba(46, 204, 113,0.2)'
                    },
                    {
                        'if': {
                            'filter_query': f'{{share_predicho}} <= {share_dia_num}'
                        },
                        'backgroundColor': 'rgba(231, 76, 60,0.2)'
                    }
                ]
            except ValueError:
                pass  # En caso de que share_dia_text no sea convertible a float

    # Gráfico de líneas
    df_lineas = filtrar_para_graficos(canal, programa)
    if df_lineas.empty:
        fig_lineas = go.Figure()
        fig_lineas.update_layout(title="No hay datos", template="plotly_white")
    else:
        df_lineas_grouped = df_lineas.groupby("date", as_index=False).agg({
            "share":"mean",
            "share_predicho":"mean"
        }).dropna(subset=["date"])
        df_lineas_grouped = df_lineas_grouped.sort_values("date")

        fig_lineas = go.Figure()
        fig_lineas.add_trace(go.Scatter(x=df_lineas_grouped["date"], y=df_lineas_grouped["share"], 
                                        mode='lines+markers', name='SHARE', line=dict(color='blue')))
        fig_lineas.add_trace(go.Scatter(x=df_lineas_grouped["date"], y=df_lineas_grouped["share_predicho"], 
                                        mode='lines+markers', name='SHARE_predicho', line=dict(color='red')))
        fig_lineas.update_layout(title="Evolución Histórica Filtrada de SHARE y SHARE_predicho", 
                                 xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")

    # Histograma Global Filtrado
    df_hist = filtrar_para_graficos(canal, programa)
    if df_hist.empty:
        fig_hist = go.Figure()
        fig_hist.update_layout(title="No hay datos globales", template="plotly_white")
    else:
        share_clean = remove_outliers(df_hist["share"].dropna())
        if share_clean.empty:
            fig_hist = go.Figure()
            fig_hist.update_layout(title="No hay datos globales después de remover outliers", template="plotly_white")
        else:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=share_clean, nbinsx=7, histnorm='percent',
                                            marker=dict(color='rgba(100,149,237,0.6)', line=dict(color='black', width=1))))
            fig_hist.update_layout(title="Histograma Global Filtrado de SHARE (7 discretizaciones, sin outliers)", 
                                   xaxis_title='SHARE', yaxis_title='% Frecuencias', template="plotly_white")

    # Generar enlace de descarga
    if not data:
        href = "#"
    else:
        df_original = df_filtrado.copy()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_original.to_excel(writer, index=False)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        filename = f"datos_filtrados_{datetime.date.today()}.xlsx"
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{encoded}"
        # El atributo 'download' se maneja en el componente 'html.A', así que solo actualizamos 'href'

    # Serializar la lista original de predicciones
    if not df_filtrado.empty:
        original_predicho_serializado = json.dumps(original_predicho)
    else:
        original_predicho_serializado = json.dumps([])

    return share_dia_text, share_mes_text, data, columns, style_data_conditional, fig_lineas, fig_hist, href, original_predicho_serializado

# Callbacks separados para cada escenario
for n in range(1, NUM_ESCENARIOS + 1):
    @app.callback(
        Output(f"mensajeFeedback_{n}", "children"),
        Input(f"btnEjecutar_{n}", "n_clicks"),
        [
            State(f"tablaEditable_{n}", "data"),
            State("emailRecipient", "value"),
            State("escenario", "value"),
        ],
        prevent_initial_call=True
    )
    def ejecutar_prediccion(n_clicks, data, email_recipient, escenario, n=n):
        if n_clicks > 0:
            # Asegurar que email_recipient sea una cadena de texto
            if isinstance(email_recipient, list):
                email_recipient = email_recipient[0] if email_recipient else ""
            
            if data and email_recipient and escenario:
                email_regex = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
                if re.match(email_regex, email_recipient):
                    try:
                        df_para_enviar = pd.DataFrame(data)
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df_para_enviar.to_excel(writer, index=False)
                        buffer.seek(0)
                        file_data = buffer.read()

                        # Configuración del correo
                        SMTP_SERVER = "smtp.sendgrid.net"
                        SMTP_PORT_SSL = 465  # Puerto para SSL
                        SMTP_PORT_TLS = 587  # Puerto para STARTTLS
                        SMTP_USER = "apikey"
                        SMTP_PASSWORD = os.getenv("SENDGRID_API_KEY")  # Asegúrate de definir esta variable de entorno
                        FROM_EMAIL = "gecaenvios@geca.es"  # Debe ser un remitente verificado en SendGrid

                        msg = EmailMessage()
                        msg["From"] = FROM_EMAIL
                        msg["To"] = email_recipient
                        msg["Subject"] = f"Datos filtrados con predicción - Escenario: {escenario}"
                        msg.set_content("Se adjuntan los datos filtrados con predicción.")

                        filename = f"datos_filtrados_{escenario}_{n}.xlsx"
                        msg.add_attachment(file_data, maintype='application',
                                           subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                           filename=filename)

                        context = ssl.create_default_context()

                        # Intentar conexión SSL
                        try:
                            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT_SSL, context=context) as server:
                                server.login(SMTP_USER, SMTP_PASSWORD)
                                server.send_message(msg)
                            return f"Correo enviado exitosamente para Escenario {n} (SSL)."
                        except (smtplib.SMTPException, ConnectionRefusedError) as e_ssl:
                            # Si falla SSL, intentar con STARTTLS
                            try:
                                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT_TLS) as server:
                                    server.ehlo()
                                    server.starttls(context=context)
                                    server.ehlo()
                                    server.login(SMTP_USER, SMTP_PASSWORD)
                                    server.send_message(msg)
                                return f"Correo enviado exitosamente para Escenario {n} (STARTTLS)."
                            except Exception as e_tls:
                                return f"Error al enviar el correo para Escenario {n}: {e_tls}"
                    except Exception as e_general:
                        return f"Error al procesar el correo para Escenario {n}: {e_general}"
                else:
                    return "Dirección de correo no válida."
            else:
                return "No se cumplen las condiciones para enviar el correo."
        return ""

if __name__ == "__main__":
    app.run_server(debug=True)