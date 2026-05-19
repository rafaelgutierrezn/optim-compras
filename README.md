# Optimización de Compras — MILP + HAPPO
desarrollado por RAfael Gutiérrez
## Descripción
Challenge para el cargo AI/ML Engineer
## Estructura

```
data_loader.py              # Carga Datos de empleo
model.py                    # Modelo MILP (PuLP)
main.py                     # ejecuta y muestra las salidas
happo.py                    # Entrenamiento multi-agente con HAPPO demo muy demo
data/instances/demo_small/  #  datos de ejemplo
requirements.txt           
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# Solución óptima (MILP)
python main.py --instance data/instances/demo_small

# Entrenamiento multi-agente (HAPPO)
python happo.py
```

## Resultados

| Método | Coste |
|--------|-------|
| MILP (óptimo) | 128.25 |
| HAPPO | 184.75 |
