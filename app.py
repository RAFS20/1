import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL
import pandas as pd
import numpy as np
import datetime
import re, json

#############################
# UTILIDADES
#############################

def clean_column_names(cols):
    """Normaliza las cabeceras del DataFrame."""
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
    """Carga datos CSV y valida columnas requeridas."""
    try:
        data = pd.read_csv(file_path, dtype=str)
        data.columns = clean_column_names(data.columns)
        required_cols = [
            "date", "fecha_inicio_del_programa", "channel_name", "program_name",
            "start_time", "end_time", "share", "macrogenero", "genero",
            "subgenero", "share_predicho", "palabras_clave", "escala_likert"
        ]
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError("Faltan columnas: " + ", ".join(missing))
        if "descripcion" not in data.columns:
            data["descripcion"] = ""
        # Convertir fechas
        data["date"] = pd.to_datetime(data["date"], errors='coerce')
        data["fecha_inicio_del_programa"] = pd.to_datetime(data["fecha_inicio_del_programa"], errors='coerce')
        # Convertir numéricos
        data["share"] = pd.to_numeric(data["share"], errors='coerce')
        data["share_predicho"] = pd.to_numeric(data["share_predicho"], errors='coerce')
        data["escala_likert"] = pd.to_numeric(data["escala_likert"], errors='coerce')
        # Normalizar canal a mayúsculas
        data["channel_name"] = data["channel_name"].str.upper()
        data = data.drop_duplicates(subset=["date", "start_time", "end_time", "channel_name", "program_name"])
        data = data[(~data["date"].isna()) & (~data["fecha_inicio_del_programa"].isna())]
        return data
    except Exception as e:
        print("Error al leer CSV:", e)
        hoy = datetime.date.today()
        data = pd.DataFrame({
            "date": [hoy] * 4,
            "fecha_inicio_del_programa": [hoy] * 4,
            "channel_name": ["LA 1", "LA 1", "LA 1", "LA 1"],
            "program_name": ["Programa A", "Programa B", "Programa C", "Programa D"],
            "start_time": ["10:00", "11:00", "12:00", "13:00"],
            "end_time": ["10:30", "11:30", "12:30", "13:30"],
            "descripcion": ["", "", "", ""],
            "escala_likert": [1, 2, 3, 4],
            "palabras_clave": ["", "", "", ""],
            "share": [0.1, 0.2, 0.3, 0.4],
            "share_predicho": [0.15, 0.25, 0.35, 0.45],
            "macrogenero": ["", "", "", ""],
            "genero": ["", "", "", ""],
            "subgenero": ["", "", "", ""]
        })
        return data

def filtrar_datos(fecha, canal, programa, df_base):
    """Filtra el DataFrame según fecha, canal y programa."""
    df = df_base.copy()
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

#############################
# FUNCIONES DE MANEJO DE TIEMPO
#############################

def adjust_time_str(time_str, base_date):
    """
    Si la hora es >= 24, se le resta 24 y se suma un día a base_date.
    Retorna (nueva_fecha, nuevo_time_str) en formato "HH:MM".
    """
    try:
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
    except:
        # En caso de error, se devuelve la hora sin cambio.
        return base_date, time_str
    if hour >= 24:
        hour -= 24
        new_date = base_date + datetime.timedelta(days=1)
    else:
        new_date = base_date
    new_time_str = f"{hour:02d}:{minute:02d}"
    return new_date, new_time_str

def time_str_to_minutes(time_str):
    """Convierte un string 'HH:MM' a minutos enteros."""
    try:
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return 0

def minutes_to_time_str(total_minutes):
    """Convierte minutos enteros a string 'HH:MM'. Nota: total_minutes puede superar 1440."""
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"

def add_minutes_to_time_str(time_str, base_date, minutes_to_add):
    """
    Suma un número de minutos a time_str, ajustando la fecha (base_date) en caso de cruzar medianoche.
    Retorna (nueva_fecha, nuevo_time_str).
    """
    total = time_str_to_minutes(time_str) + minutes_to_add
    day_offset = total // 1440  # 1440 minutos por día
    new_minutes = total % 1440
    new_time_str = minutes_to_time_str(new_minutes)
    new_date = base_date + datetime.timedelta(days=day_offset)
    return new_date, new_time_str

def process_scenario_data(data):
    """
    Recorre la data de un escenario:
      - Si se detecta que el end_time de una fila fue modificado (comparado con su valor base),
        se calcula la diferencia (en minutos) y se propaga a las filas siguientes (solamente
        aquellas que originalmente pertenecían al mismo día de emisión).
      - Se actualiza también el share_predicho aplicándole un factor fijo (dependiendo del programa)
        y un ajuste aleatorio de ±5%.
    """
    new_data = []
    modified_index = None
    cumulative_diff = 0
    modified_original_date = None  # se extrae del campo "original_broadcast_date"
    for i, row in enumerate(data):
        # Aseguramos que existan las variables base; si no, se inicializan
        if "start_time_base" not in row:
            row["start_time_base"] = row["start_time"]
        if "end_time_base" not in row:
            row["end_time_base"] = row["end_time"]
        # Comparamos el end_time actual con el base para detectar modificación
        new_end = time_str_to_minutes(row["end_time"])
        base_end = time_str_to_minutes(row["end_time_base"])
        diff = new_end - base_end
        if diff != 0 and modified_index is None:
            modified_index = i
            cumulative_diff = diff
            # Guardamos el día de emisión original (en formato ISO) de la fila modificada
            modified_original_date = row.get("original_broadcast_date", row.get("broadcast_date"))
            # Actualizamos el valor base de esta fila para que futuras modificaciones sean relativas
            row["end_time_base"] = row["end_time"]
        # Si ya hubo una modificación y esta fila es posterior
        if modified_index is not None and i > modified_index:
            # Propagar solo a filas cuyo día original coincide con el de la fila modificada
            if row.get("original_broadcast_date") == modified_original_date:
                try:
                    base_day = datetime.datetime.strptime(row["original_broadcast_date"], "%Y-%m-%d").date()
                except:
                    base_day = datetime.date.today()
                # Se actualiza start_time y end_time a partir de sus valores base sumando la diferencia
                new_bd, new_start = add_minutes_to_time_str(row["start_time_base"], base_day, cumulative_diff)
                _, new_end = add_minutes_to_time_str(row["end_time_base"], base_day, cumulative_diff)
                row["start_time"] = new_start
                row["end_time"] = new_end
                row["broadcast_date"] = new_bd.isoformat()
                # También se actualizan los valores base para futuras modificaciones
                row["start_time_base"] = new_start
                row["end_time_base"] = new_end
        # Actualizamos share_predicho: se utiliza el valor base multiplicado por el factor fijo
        # (1.178979 para ESTRENO, 0.94646546 para los demás) y se le aplica un ±5% aleatorio.
        prog = row.get("program_name", "").upper()
        factor = 1.178979 if prog == "ESTRENO" else 0.94646546
        random_factor = 1 + np.random.uniform(-0.05, 0.05)
        try:
            base_share = float(row.get("share_predicho_base", row.get("share_predicho", 0)))
        except:
            base_share = 0
        row["share_predicho"] = round(base_share * factor * random_factor, 2)
        new_data.append(row)
    return new_data

#############################
# DATOS INICIALES
#############################

datos_iniciales = load_data("datos.csv")
if not datos_iniciales.empty:
    min_date = datos_iniciales["date"].min()
    if pd.isnull(min_date):
        min_date = datetime.date(2000, 1, 1)
    else:
        min_date = min_date.date()
    max_date = datetime.date.today() + datetime.timedelta(days=365)
else:
    min_date = datetime.date(2000, 1, 1)
    max_date = datetime.date.today() + datetime.timedelta(days=365)

canales_disponibles = datos_iniciales["channel_name"].unique().tolist() if not datos_iniciales.empty else []
canales_disponibles_sorted = sorted(canales_disponibles)
canal_default = "LA 1" if "LA 1" in canales_disponibles else (canales_disponibles_sorted[0] if canales_disponibles else "Todos")

#############################
# CONFIGURACIÓN DE DASH
#############################

external_stylesheets = [
    'https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css',
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server
NUM_ESCENARIOS = 4

#############################
# LAYOUT AUXILIAR: Creación de Escenario
#############################
def crear_escenario(n):
    """
    Crea el layout para un escenario usando dash_table.DataTable
    configurada para ser editable y con selección de celdas.
    Se agrega una columna extra "Acción" que muestra el ícono "🔍".
    """
    return html.Div(
        style={
            "border": "1px solid #ccc",
            "padding": "10px",
            "borderRadius": "5px",
            "backgroundColor": "#ffffff",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "display": "flex",
            "flexDirection": "column",
            "height": "100%"
        },
        children=[
            html.H4(f"Escenario {n}", style={"color": "#2c3e50", "fontWeight": "500", "margin": "5px 0"}),
            html.Div(
                style={"marginBottom": "10px"},
                children=[
                    html.Label("Share Promedio Diario (share_predicho):", style={"color": "#34495e", "marginBottom": "5px"}),
                    html.Pre(
                        id={"type": "sharePromedio", "index": n},
                        children="0.00",
                        style={
                            "backgroundColor": "#ecf0f1",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "color": "#16a085",
                            "fontWeight": "500",
                            "margin": "0"
                        }
                    )
                ]
            ),
            dash_table.DataTable(
                id=f"tablaEditable_{n}",
                columns=[
                    {"name": "Program_Name", "id": "program_name", "type": "text"},
                    {"name": "Start_Time", "id": "start_time", "type": "text"},
                    {"name": "End_Time", "id": "end_time", "type": "text"},
                    {"name": "Descripción", "id": "descripcion", "type": "text"},
                    {"name": "Escala Likert", "id": "escala_likert", "type": "numeric"},
                    {"name": "Palabras Clave", "id": "palabras_clave", "type": "text"},
                    {"name": "Share_predicho", "id": "share_predicho", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "Acción", "id": "accion", "presentation": "markdown"}
                ],
                data=[],  # Se llenará vía callback
                editable=True,
                cell_selectable=True,
                row_selectable="single",
                active_cell=None,
                style_table={
                    "overflowY": "auto",
                    "height": "300px",
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #ccc",
                    "borderRadius": "5px",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
                },
                style_cell={
                    "fontFamily": "Roboto",
                    "fontSize": "14px",
                    "textAlign": "left",
                    "padding": "5px"
                },
                style_header={
                    "fontWeight": "500",
                    "backgroundColor": "#ecf0f1",
                    "borderBottom": "2px solid #ccc"
                },
            ),
            html.Div(
                "Haga clic en el ícono de lupita para abrir el buscador (pasado o ESTRENO).",
                style={"fontSize": "12px", "color": "#7f8c8d", "marginTop": "5px"}
            )
        ]
    )

#############################
# LAYOUT PRINCIPAL
#############################
app.layout = html.Div(
    style={"margin": "0", "padding": "10px", "backgroundColor": "#f7f7f7", "fontFamily": "Roboto"},
    children=[
        html.H1("Panel Analista", style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px", "fontWeight": "700"}),
        # Filtros globales
        html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "justifyContent": "center"},
                    children=[
                        html.Div(
                            style={"width": "240px"},
                            children=[
                                html.Label("Seleccionar Fecha:", style={"color": "#34495e", "marginTop": "5px"}),
                                dcc.DatePickerSingle(
                                    id="fecha",
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    date=datetime.date.today(),
                                    display_format="DD/MM/YYYY",
                                    style={"marginBottom": "10px", "marginTop": "5px"}
                                ),
                            ]
                        ),
                        html.Div(
                            style={"width": "240px"},
                            children=[
                                html.Label("Seleccionar Canal:", style={"color": "#34495e"}),
                                dcc.Dropdown(
                                    id="canal",
                                    options=[{"label": "Todos", "value": "Todos"}] +
                                            [{"label": c, "value": c} for c in canales_disponibles_sorted],
                                    value=canal_default,
                                    style={"marginBottom": "10px", "marginTop": "5px"}
                                ),
                            ]
                        ),
                        html.Div(
                            style={"width": "240px"},
                            children=[
                                html.Label("Seleccionar Programa:", style={"color": "#34495e"}),
                                dcc.Dropdown(
                                    id="programa",
                                    options=[{"label": "Todos", "value": "Todos"}],
                                    value="Todos",
                                    style={"marginBottom": "10px", "marginTop": "5px"}
                                ),
                            ]
                        ),
                        html.Div(
                            style={"width": "240px"},
                            children=[
                                html.Label("Correo del Destinatario:", style={"color": "#34495e"}),
                                dcc.Input(
                                    id="emailRecipient",
                                    type="email",
                                    placeholder="ejemplo@dominio.com",
                                    value="rafernandezsalguero@mediapro.tv",
                                    style={
                                        "width": "100%", "marginBottom": "10px", "marginTop": "5px",
                                        "padding": "5px", "border": "1px solid #ccc", "borderRadius": "3px"
                                    }
                                ),
                            ]
                        ),
                        html.Div(
                            style={"width": "240px"},
                            children=[
                                html.Label("Escenario:", style={"color": "#34495e"}),
                                dcc.Input(
                                    id="escenario",
                                    type="text",
                                    placeholder="Nombre del escenario",
                                    value="resultados",
                                    style={
                                        "width": "100%", "marginBottom": "10px", "marginTop": "5px",
                                        "padding": "5px", "border": "1px solid #ccc", "borderRadius": "3px"
                                    }
                                ),
                            ]
                        ),
                        html.Div(
                            style={"width": "240px"},
                            children=[
                                html.Label("Descargar Datos:", style={"color": "#34495e"}),
                                html.A(
                                    "Descargar Datos",
                                    id="descargarDatos",
                                    download="",
                                    href="#",
                                    target="_blank",
                                    style={
                                        "display": "inline-block", "marginTop": "5px",
                                        "color": "white", "backgroundColor": "#27ae60",
                                        "padding": "10px", "textDecoration": "none",
                                        "borderRadius": "3px", "textAlign": "center", "fontWeight": "500"
                                    }
                                )
                            ]
                        ),
                    ]
                ),
            ]
        ),
        # Malla de escenarios (2x2)
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"},
            children=[crear_escenario(n) for n in range(1, NUM_ESCENARIOS + 1)]
        ),
        # Botón global para refrescar
        html.Div(
            style={"marginTop": "10px", "textAlign": "center"},
            children=[
                html.Button(
                    "Refrescar/Afectar Escenarios",
                    id="btnRefrescarEscenarios",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#f39c12", "color": "white", "border": "none",
                        "padding": "10px 20px", "cursor": "pointer",
                        "borderRadius": "3px", "fontWeight": "500"
                    }
                )
            ]
        ),
        # Mensaje de feedback (opcional)
        html.Div(id="mensajeFeedback", style={"marginTop": "10px", "textAlign": "center", "fontWeight": "500"}),
        # Modal emergente para el buscador (pasado o ESTRENO)
        html.Div(
            id="modalContextMenu",
            style={
                "display": "none",
                "position": "fixed",
                "zIndex": 9999,
                "left": 0,
                "top": 0,
                "width": "100%",
                "height": "100%",
                "overflow": "auto",
                "backgroundColor": "rgba(0,0,0,0.5)"
            },
            children=[
                html.Div(
                    style={
                        "position": "fixed",
                        "top": "50%",
                        "left": "50%",
                        "transform": "translate(-50%, -50%)",
                        "backgroundColor": "#fff",
                        "padding": "20px",
                        "zIndex": 10000,
                        "borderRadius": "10px",
                        "width": "400px",
                        "boxShadow": "0 2px 10px rgba(0,0,0,0.3)"
                    },
                    children=[
                        html.H3("Seleccionar Programa"),
                        dcc.Input(
                            id="inputBuscarProgramas",
                            type="text",
                            placeholder="Buscar programa...",
                            style={"width": "100%", "marginBottom": "10px"}
                        ),
                        html.Div(
                            id="divResultados",
                            style={
                                "maxHeight": "200px", "overflowY": "auto",
                                "border": "1px solid #ccc", "padding": "5px", "marginBottom": "10px"
                            }
                        ),
                        html.Button(
                            "Seleccionar Estreno",
                            id="btnSeleccionarEstreno",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#8e44ad", "color": "white", "border": "none",
                                "padding": "10px", "cursor": "pointer", "borderRadius": "3px",
                                "marginRight": "10px"
                            }
                        ),
                        html.Button(
                            "Cancelar",
                            id="btnCancelarContextMenu",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#c0392b", "color": "white", "border": "none",
                                "padding": "10px", "cursor": "pointer", "borderRadius": "3px"
                            }
                        )
                    ]
                )
            ]
        ),
        # Stores
        dcc.Store(id="storeAction"),
        dcc.Store(id="storeRightClickSelection"),
        dcc.Store(id="storeScenarioData")
    ]
)

#############################
# CALLBACKS
#############################

# Callback B: Actualiza los resultados de búsqueda en el modal según lo escrito.
@app.callback(
    Output("divResultados", "children"),
    Input("inputBuscarProgramas", "value")
)
def update_search_results(search_value):
    if not search_value:
        return []
    programas = datos_iniciales["program_name"].dropna().unique().tolist()
    filtered = [p for p in programas if search_value.lower() in p.lower()]
    return [
        html.Button(
            prog,
            id={"type": "opcionPrograma", "program": prog},
            n_clicks=0,
            style={"width": "100%", "textAlign": "left", "marginBottom": "5px"}
        )
        for prog in sorted(filtered)
    ]

# Callback C unificado: Maneja la edición de tablas, clics en la columna "Acción", refresco y selección en el modal.
@app.callback(
    [
        Output("storeScenarioData", "data"),
        Output("tablaEditable_1", "data"),
        Output("tablaEditable_2", "data"),
        Output("tablaEditable_3", "data"),
        Output("tablaEditable_4", "data"),
        Output({"type": "sharePromedio", "index": 1}, "children"),
        Output({"type": "sharePromedio", "index": 2}, "children"),
        Output({"type": "sharePromedio", "index": 3}, "children"),
        Output({"type": "sharePromedio", "index": 4}, "children"),
        Output("storeRightClickSelection", "data"),
        Output("storeAction", "data")
    ],
    [
        # Inputs para detectar clic en la celda (columna "accion")
        Input("tablaEditable_1", "active_cell"),
        Input("tablaEditable_2", "active_cell"),
        Input("tablaEditable_3", "active_cell"),
        Input("tablaEditable_4", "active_cell"),
        # Botones y opciones del modal/refresco
        Input("btnRefrescarEscenarios", "n_clicks"),
        Input("btnSeleccionarEstreno", "n_clicks"),
        Input("btnCancelarContextMenu", "n_clicks"),
        Input({"type": "opcionPrograma", "program": ALL}, "n_clicks"),
        # Entradas para capturar ediciones en las tablas
        Input("tablaEditable_1", "data"),
        Input("tablaEditable_2", "data"),
        Input("tablaEditable_3", "data"),
        Input("tablaEditable_4", "data")
    ],
    [
        State("storeScenarioData", "data"),
        State("storeAction", "data"),
        State("fecha", "date"),
        State("canal", "value"),
        State("programa", "value")
    ]
)
def handle_scenarios(
    ac1, ac2, ac3, ac4,
    n_refrescar, n_estreno, n_cancel, n_opciones,
    data1, data2, data3, data4,
    current_store, action_data,
    fecha, canal, programa
):
    """
    Este callback unifica la lógica de:
      - Inicialización (filtrado de datos y ajuste de tiempos al cargar, incluyendo:
          • Convertir horas mayores a 24 al día siguiente.
          • Almacenar campos base y el día de emisión ("broadcast_date" y "original_broadcast_date")
      - Propagar cambios de _end_time_: si se detecta una modificación en una fila,
        se calcula la diferencia en minutos y se suma o resta a los programas posteriores (dentro
        del mismo día original).
      - Aplicar un ajuste aleatorio del ±5% al share_predicho en cada actualización.
      - Manejar la apertura y selección del modal.
    """
    if current_store is None:
        current_store = {}

    new_storeAction = action_data
    new_storeRightClick = None  # Se limpia al usar el modal

    ctx = callback_context
    trigger_ids = [t["prop_id"] for t in ctx.triggered] if ctx.triggered else []

    # -----------------------
    # Inicialización (primer carga)
    if not ctx.triggered:
        if not current_store:
            df_filtrado = filtrar_datos(fecha, canal, programa, datos_iniciales)
            # Se incluye "date" para tener la fecha original (aunque no se muestra en la tabla)
            columnas = ["date", "program_name", "start_time", "end_time", "descripcion",
                        "escala_likert", "palabras_clave", "share_predicho"]
            try:
                data_records = df_filtrado[columnas].to_dict("records")
            except:
                data_records = []
            # Ajuste de tiempos: si start_time o end_time tienen hora>=24, se corrige y se almacena el día de emisión.
            for rec in data_records:
                # Convertir la columna "date" a objeto date
                if isinstance(rec["date"], str):
                    try:
                        base_date = datetime.datetime.strptime(rec["date"], "%Y-%m-%d").date()
                    except:
                        base_date = datetime.date.today()
                else:
                    base_date = rec["date"] if isinstance(rec["date"], datetime.date) else rec["date"].date()
                new_date_start, new_start = adjust_time_str(rec["start_time"], base_date)
                new_date_end, new_end = adjust_time_str(rec["end_time"], base_date)
                rec["start_time"] = new_start
                rec["end_time"] = new_end
                rec["broadcast_date"] = new_date_start.isoformat()  # Se usa el día de inicio como día de emisión
                rec["original_broadcast_date"] = new_date_start.isoformat()
                # Se almacenan versiones "base" para detectar ediciones posteriores
                rec["start_time_base"] = new_start
                rec["end_time_base"] = new_end
                rec["accion"] = "🔍"
                # Se conserva share_predicho_base para posteriores cálculos
                rec["share_predicho_base"] = rec["share_predicho"]
            for i in range(1, NUM_ESCENARIOS + 1):
                # Se copia la data filtrada (ya con ajustes de tiempo) a cada escenario
                current_store[f"scenario_{i}"] = data_records.copy()

        # Preparar datos para las tablas y calcular promedios
        tablas = []
        shares = []
        for i in range(1, NUM_ESCENARIOS + 1):
            tabla = current_store.get(f"scenario_{i}", [])
            tablas.append(tabla)
            vals = [float(r.get("share_predicho", 0)) for r in tabla if r.get("share_predicho") is not None]
            avg = np.mean(vals) if vals else 0.0
            shares.append(f"{avg:.2f}")
        return (
            current_store,
            tablas[0], tablas[1], tablas[2], tablas[3],
            shares[0], shares[1], shares[2], shares[3],
            None,
            new_storeAction
        )

    # -----------------------
    # Si se presiona el botón de refrescar
    if any("btnRefrescarEscenarios" in tid for tid in trigger_ids) and n_refrescar:
        df_filtrado = filtrar_datos(fecha, canal, programa, datos_iniciales)
        columnas = ["date", "program_name", "start_time", "end_time", "descripcion",
                    "escala_likert", "palabras_clave", "share_predicho"]
        try:
            data_records = df_filtrado[columnas].to_dict("records")
        except:
            data_records = []
        for rec in data_records:
            if isinstance(rec["date"], str):
                try:
                    base_date = datetime.datetime.strptime(rec["date"], "%Y-%m-%d").date()
                except:
                    base_date = datetime.date.today()
            else:
                base_date = rec["date"] if isinstance(rec["date"], datetime.date) else rec["date"].date()
            new_date_start, new_start = adjust_time_str(rec["start_time"], base_date)
            new_date_end, new_end = adjust_time_str(rec["end_time"], base_date)
            rec["start_time"] = new_start
            rec["end_time"] = new_end
            rec["broadcast_date"] = new_date_start.isoformat()
            rec["original_broadcast_date"] = new_date_start.isoformat()
            rec["start_time_base"] = new_start
            rec["end_time_base"] = new_end
            rec["accion"] = "🔍"
            rec["share_predicho_base"] = rec["share_predicho"]
        for i in range(1, NUM_ESCENARIOS + 1):
            current_store[f"scenario_{i}"] = data_records.copy()

    # -----------------------
    # Si se modificó la data de alguna tabla (capturando ediciones en cualquier celda)
    # Se aplica la función process_scenario_data para cada escenario
    if any("tablaEditable_1.data" in tid for tid in trigger_ids) and data1 is not None:
        current_store["scenario_1"] = process_scenario_data(data1)
    if any("tablaEditable_2.data" in tid for tid in trigger_ids) and data2 is not None:
        current_store["scenario_2"] = process_scenario_data(data2)
    if any("tablaEditable_3.data" in tid for tid in trigger_ids) and data3 is not None:
        current_store["scenario_3"] = process_scenario_data(data3)
    if any("tablaEditable_4.data" in tid for tid in trigger_ids) and data4 is not None:
        current_store["scenario_4"] = process_scenario_data(data4)

    # -----------------------
    # Si se hace clic en la columna "Acción" (para abrir el modal)
    clicked_cells = [ac1, ac2, ac3, ac4]
    for i, cell in enumerate(clicked_cells, start=1):
        if cell and cell.get("column_id") == "accion":
            new_storeAction = {"scenario": i, "row": cell["row"]}
            break

    # -----------------------
    # Si se presiona el botón de Cancelar en el modal
    if any("btnCancelarContextMenu" in tid for tid in trigger_ids) and n_cancel:
        new_storeAction = None
        new_storeRightClick = None

    # -----------------------
    # Si se selecciona "ESTRENO" o un programa desde el modal
    if new_storeAction and "scenario" in new_storeAction and "row" in new_storeAction:
        scenario_idx = new_storeAction["scenario"]
        row_idx = new_storeAction["row"]
        key_scenario = f"scenario_{scenario_idx}"
        current_store.setdefault(key_scenario, [])
        if any("btnSeleccionarEstreno" in tid for tid in trigger_ids) and n_estreno:
            nuevo_programa = "ESTRENO"
        else:
            nuevo_programa = None
            for tid in trigger_ids:
                if "opcionPrograma" in tid:
                    try:
                        dict_str = tid.split(".")[0].replace("'", '"')
                        prog_id = json.loads(dict_str)
                        if prog_id.get("program"):
                            nuevo_programa = prog_id["program"]
                            break
                    except:
                        pass
        if nuevo_programa:
            if row_idx < len(current_store[key_scenario]):
                current_store[key_scenario][row_idx]["program_name"] = nuevo_programa
                try:
                    base_val = float(current_store[key_scenario][row_idx].get("share_predicho_base", current_store[key_scenario][row_idx]["share_predicho"]))
                except:
                    base_val = 0
                factor = 1.178979 if nuevo_programa.upper() == "ESTRENO" else 0.94646546
                # Se actualiza share_predicho también con el ±5% aleatorio
                random_factor = 1 + np.random.uniform(-0.05, 0.05)
                current_store[key_scenario][row_idx]["share_predicho"] = round(base_val * factor * random_factor, 2)
            new_storeAction = None
            new_storeRightClick = None

    # -----------------------
    # Preparar los datos finales para las tablas y calcular el promedio de share_predicho
    tablas = []
    shares = []
    for i in range(1, NUM_ESCENARIOS + 1):
        tabla = current_store.get(f"scenario_{i}", [])
        tablas.append(tabla)
        vals = []
        for row in tabla:
            try:
                vals.append(float(row.get("share_predicho", 0)))
            except:
                pass
        avg = np.mean(vals) if vals else 0.0
        shares.append(f"{avg:.2f}")

    return (
        current_store,          # storeScenarioData
        tablas[0],
        tablas[1],
        tablas[2],
        tablas[3],
        shares[0],
        shares[1],
        shares[2],
        shares[3],
        new_storeRightClick,    # storeRightClickSelection
        new_storeAction         # storeAction
    )

# Callback D: Controla la visibilidad del modal según storeAction y storeRightClickSelection.
@app.callback(
    Output("modalContextMenu", "style"),
    [Input("storeAction", "data"), Input("storeRightClickSelection", "data")]
)
def update_modal_style(action_data, selection):
    """
    Muestra el modal si storeAction no es None (se hizo clic en la lupita)
    y storeRightClickSelection es None (acción sin completar).
    """
    if action_data is not None and selection is None:
        return {"display": "block"}
    return {"display": "none"}

#############################
# MAIN
#############################
if __name__ == "__main__":
    app.run_server(debug=True)
