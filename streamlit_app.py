import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="SuperDespacho – Proyección mensual", layout="wide")
ART_DIR = Path(__file__).parent / "artifacts"

PROD_PATH = ART_DIR / "panel_producto_mes.parquet"
CLI_PATH  = ART_DIR / "panel_cliente_mes.parquet"
CAT_PATH  = ART_DIR / "panel_categoria_mes.parquet"

# -----------------------------
# Helpers
# -----------------------------
def month_start_from_yyyymm(s: str) -> pd.Timestamp:
    """Parsea YYYY-MM y retorna Timestamp al primer día del mes."""
    return pd.to_datetime(f"{s}-01").to_period("M").to_timestamp()

def normalize_month(ts) -> pd.Timestamp:
    return pd.to_datetime(ts).to_period("M").to_timestamp()

def safe_filter_options(options: list[str], query: str, max_show: int = 250) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return options[:max_show]
    out = [x for x in options if q in str(x).lower()]
    return out[:max_show]

def seasonal_baseline(series_m: pd.Series, target_month: pd.Timestamp,
                     k_years: int = 3, fallback_last_n: int = 6) -> float:
    """
    Baseline estacional:
    - toma el mismo mes del año (ej: febrero) en hasta k_years ocurrencias recientes
    - promedia
    - si no existe, cae a promedio de últimos fallback_last_n meses
    """
    s = series_m.dropna().astype(float).sort_index()
    if s.empty:
        return 0.0

    target_month = normalize_month(target_month)
    m = target_month.month

    same_month = s[s.index.month == m]
    if len(same_month) >= 1:
        yhat = float(same_month.tail(k_years).mean())
        return max(0.0, yhat)

    yhat = float(s.tail(fallback_last_n).mean())
    return max(0.0, yhat)

def fmt_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_parquet(path: Path, _mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normalizar mes
    df["anio_mes"] = pd.to_datetime(df["anio_mes"]).dt.to_period("M").dt.to_timestamp()
    return df

def must_exist(path: Path):
    if not path.exists():
        st.error(f"Falta archivo: {path}")
        st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("SuperDespacho – Proyección mensual (MVP baseline)")
st.caption("MVP sin ML: proyección estacional simple (mismo mes en años previos) + históricos")

must_exist(PROD_PATH)
must_exist(CLI_PATH)
must_exist(CAT_PATH)

prod = load_parquet(PROD_PATH, os.path.getmtime(PROD_PATH))
cli  = load_parquet(CLI_PATH,  os.path.getmtime(CLI_PATH))
cat  = load_parquet(CAT_PATH,  os.path.getmtime(CAT_PATH))

# Último mes global (máximo entre paneles)
last_month = max(prod["anio_mes"].max(), cli["anio_mes"].max(), cat["anio_mes"].max())
default_next = (last_month + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Productos", f"{prod['producto_normalizado'].nunique():,}".replace(",", "."))
c2.metric("Clientes", f"{cli['cliente'].nunique():,}".replace(",", "."))
c3.metric("Categorías", f"{cat['categoria_final'].nunique():,}".replace(",", "."))
c4.metric("Último mes", f"{last_month:%Y-%m}")

st.divider()

# -----------------------------
# Mes objetivo
# -----------------------------
st.subheader("Mes objetivo")
mes_str = st.text_input("Mes a proyectar (YYYY-MM)", value=default_next.strftime("%Y-%m"))

try:
    mes_obj = month_start_from_yyyymm(mes_str)
except Exception:
    st.error("Formato inválido. Usa YYYY-MM (ej: 2026-02).")
    st.stop()

horizon = (mes_obj.to_period("M") - last_month.to_period("M")).n
st.caption(f"Mes objetivo: {mes_obj:%Y-%m} | Horizonte vs último dato: {horizon} mes(es).")

st.divider()

tab1, tab2, tab3 = st.tabs(
    ["Producto: unidades y margen", "Cliente: compra y margen", "Empresa: ventas y margen"]
)

# -----------------------------
# TAB 1: Producto
# -----------------------------
with tab1:
    st.subheader("Producto – Proyección mensual")

    productos = sorted(prod["producto_normalizado"].dropna().astype(str).unique().tolist())
    q_prod = st.text_input("Buscar producto", value="")
    prod_fil = safe_filter_options(productos, q_prod)

    if not prod_fil:
        st.warning("Sin coincidencias.")
        st.stop()

    producto_sel = st.selectbox("Selecciona producto", prod_fil)

    dfp = prod[prod["producto_normalizado"].astype(str) == str(producto_sel)].copy().sort_values("anio_mes")
    last_p = dfp["anio_mes"].max()
    st.caption(f"Último mes disponible para este producto: {last_p:%Y-%m}")

    # Series
    s_q = dfp.set_index("anio_mes")["q_total_mes"]
    s_v = dfp.set_index("anio_mes")["ventas_total_mes"]
    s_m = dfp.set_index("anio_mes")["margen_total_mes"]

    # Baseline estacional
    yhat_q = seasonal_baseline(s_q, mes_obj)
    yhat_v = seasonal_baseline(s_v, mes_obj)
    yhat_m = seasonal_baseline(s_m, mes_obj)

    a, b, c = st.columns(3)
    a.metric("Unidades esperadas", fmt_int(yhat_q))
    b.metric("Ventas esperadas (CLP)", fmt_int(yhat_v))
    c.metric("Margen esperado (CLP)", fmt_int(yhat_m))

    with st.expander("Histórico últimos 24 meses"):
        out = dfp[["anio_mes","q_total_mes","ventas_total_mes","margen_total_mes","clientes_distintos"]].tail(24)
        st.dataframe(out)



# -----------------------------
# TAB 2: Cliente
# -----------------------------
with tab2:
    st.subheader("Cliente – Proyección mensual")

    clientes = sorted(cli["cliente"].dropna().astype(str).unique().tolist())
    q_cli = st.text_input("Buscar cliente", value="", key="q_cli")
    cli_fil = safe_filter_options(clientes, q_cli)

    if not cli_fil:
        st.warning("Sin coincidencias.")
        st.stop()

    cliente_sel = st.selectbox("Selecciona cliente", cli_fil)

    dfc = cli[cli["cliente"].astype(str) == str(cliente_sel)].copy().sort_values("anio_mes")
    last_c = dfc["anio_mes"].max()
    st.caption(f"Último mes disponible para este cliente: {last_c:%Y-%m}")

    s_q = dfc.set_index("anio_mes")["q_total_mes"]
    s_v = dfc.set_index("anio_mes")["ventas_total_mes"]
    s_m = dfc.set_index("anio_mes")["margen_total_mes"]

    yhat_q = seasonal_baseline(s_q, mes_obj)
    yhat_v = seasonal_baseline(s_v, mes_obj)
    yhat_m = seasonal_baseline(s_m, mes_obj)

    a, b, c = st.columns(3)
    a.metric("Unidades esperadas", fmt_int(yhat_q))
    b.metric("Compra esperada (CLP)", fmt_int(yhat_v))
    c.metric("Margen esperado (CLP)", fmt_int(yhat_m))

    with st.expander("Histórico últimos 24 meses"):
        out = dfc[["anio_mes","q_total_mes","ventas_total_mes","margen_total_mes","productos_distintos","categorias_distintas"]].tail(24)
        st.dataframe(out)

# -----------------------------
# TAB 3: Empresa (agregado)
# -----------------------------
with tab3:
    st.subheader("Empresa – Proyección mensual (agregado)")

    # Agregado mensual empresa: sumando categorías (una fila por mes por categoría)
    dfE = (
        cat.groupby("anio_mes", as_index=False)[["q_total_mes","ventas_total_mes","margen_total_mes"]]
        .sum()
        .sort_values("anio_mes")
    )

    last_e = dfE["anio_mes"].max()
    st.caption(f"Último mes disponible (empresa): {last_e:%Y-%m}")

    s_q = dfE.set_index("anio_mes")["q_total_mes"]
    s_v = dfE.set_index("anio_mes")["ventas_total_mes"]
    s_m = dfE.set_index("anio_mes")["margen_total_mes"]

    yhat_q = seasonal_baseline(s_q, mes_obj)
    yhat_v = seasonal_baseline(s_v, mes_obj)
    yhat_m = seasonal_baseline(s_m, mes_obj)

    a, b, c = st.columns(3)
    a.metric("Unidades esperadas (empresa)", fmt_int(yhat_q))
    b.metric("Ventas esperadas (empresa, CLP)", fmt_int(yhat_v))
    c.metric("Margen esperado (empresa, CLP)", fmt_int(yhat_m))

    with st.expander("Histórico últimos 24 meses"):
        st.dataframe(dfE.tail(24))


        