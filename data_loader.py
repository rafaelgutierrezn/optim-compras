import json
from pathlib import Path
import pandas as pd


def load_instance(folder):
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text())
    periodos = list(range(1, int(cfg["periodos"]) + 1))

    df_prod  = pd.read_csv(folder / "productos.csv")
    df_prov  = pd.read_csv(folder / "proveedores.csv")
    df_comp  = pd.read_csv(folder / "compatibilidad.csv")
    df_dem   = pd.read_csv(folder / "demanda.csv")
    df_stock = pd.read_csv(folder / "stock_inicial.csv")

    return {
        "productos":    df_prod["id"].astype(str).tolist(),
        "proveedores":  df_prov["id"].astype(str).tolist(),
        "periodos":     periodos,
        "compatibilidad": {
            (str(r.id_producto), str(r.id_proveedor))
            for r in df_comp.itertuples()
        },
        "vida_util":   dict(zip(df_prod["id"].astype(str), df_prod["vida_util"].astype(int))),
        "costo_mant":  dict(zip(df_prod["id"].astype(str), df_prod["costo_mant"])),
        "costo_merma": dict(zip(df_prod["id"].astype(str), df_prod["costo_merma"])),
        "moq":         dict(zip(df_prod["id"].astype(str), df_prod["moq"].astype(int))),
        "volumen":     dict(zip(df_prod["id"].astype(str), df_prod["volumen"])),
        "costo_fijo":  dict(zip(df_prov["id"].astype(str), df_prov["costo_fijo"])),
        "min_pedido":  dict(zip(df_prov["id"].astype(str), df_prov["min_pedido_valor"])),
        "precio": {
            (str(r.id_producto), str(r.id_proveedor)): float(r.precio)
            for r in df_comp.itertuples()
        },
        "demanda": {
            (str(r.id_producto), int(r.periodo)): float(r.cantidad)
            for r in df_dem.itertuples()
        },
        "stock_inicial": {
            (str(r.id_producto), int(r.edad)): float(r.cantidad)
            for r in df_stock.itertuples()
        },
        "capacidad": float(cfg["capacidad"]),
    }
