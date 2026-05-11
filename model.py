import pulp


def build_and_solve(data, verbose=False):
    prob = pulp.LpProblem("optim_compras", pulp.LpMinimize)

    productos = data["productos"]
    proveedores = data["proveedores"]
    periodos = data["periodos"]
    compat = list(data["compatibilidad"])
    edades = {i: list(range(data["vida_util"][i] + 1)) for i in productos}
    M = sum(data["demanda"].values()) + sum(data["stock_inicial"].values()) + 1

    # --- Variables ---
    p  = {(i,j,t): pulp.LpVariable(f"p_{i}_{j}_{t}",  lowBound=0)           for (i,j) in compat   for t in periodos}
    I  = {(i,a,t): pulp.LpVariable(f"I_{i}_{a}_{t}",  lowBound=0)           for i in productos for a in edades[i] for t in periodos}
    x  = {(i,a,t): pulp.LpVariable(f"x_{i}_{a}_{t}",  lowBound=0)           for i in productos for a in edades[i] for t in periodos}
    y  = {(j,t):   pulp.LpVariable(f"y_{j}_{t}",       cat="Binary")         for j in proveedores   for t in periodos}
    yp = {(i,j,t): pulp.LpVariable(f"yp_{i}_{j}_{t}", cat="Binary")         for (i,j) in compat   for t in periodos}
    w  = {(i,t):   pulp.LpVariable(f"w_{i}_{t}",       lowBound=0)           for i in productos     for t in periodos}

    # --- Función objetivo ---
    prob += (
        pulp.lpSum(data["precio"][i,j]    * p[i,j,t]  for (i,j) in compat   for t in periodos)
      + pulp.lpSum(data["costo_fijo"][j]  * y[j,t]    for j in proveedores   for t in periodos)
      + pulp.lpSum(data["costo_mant"][i]  * I[i,a,t]  for i in productos for a in edades[i] for t in periodos)
      + pulp.lpSum(data["costo_merma"][i] * w[i,t]    for i in productos     for t in periodos)
    )

    # --- R1: las compras entran con edad 0 ---
    for i in productos:
        sj = [j for j in proveedores if (i, j) in data["compatibilidad"]]
        for t in periodos:
            prob += I[i,0,t] == pulp.lpSum(p[i,j,t] for j in sj)

    # --- R2: el stock no consumido envejece un periodo ---
    for i in productos:
        Li = data["vida_util"][i]
        for a in range(1, Li + 1):
            for idx, t in enumerate(periodos):
                if idx == 0:
                    prob += I[i,a,t] == data["stock_inicial"].get((i, a - 1), 0.0)
                else:
                    tp = periodos[idx - 1]
                    prob += I[i,a,t] == I[i,a-1,tp] - x[i,a-1,tp]

    # --- R3: satisfacción exacta de la demanda ---
    for i in productos:
        Li = data["vida_util"][i]
        for t in periodos:
            prob += pulp.lpSum(x[i,a,t] for a in range(Li + 1)) == data["demanda"].get((i, t), 0.0)

    # --- R4: no consumir más del inventario disponible (edad >= 1) ---
    for i in productos:
        for a in range(1, data["vida_util"][i] + 1):
            for t in periodos:
                prob += x[i,a,t] <= I[i,a,t]

    # --- R5: no consumir más de las compras del periodo (edad 0) ---
    for i in productos:
        sj = [j for j in proveedores if (i, j) in data["compatibilidad"]]
        for t in periodos:
            prob += x[i,0,t] <= pulp.lpSum(p[i,j,t] for j in sj)

    # --- R6: merma = inventario de edad máxima no consumido ---
    for i in productos:
        Li = data["vida_util"][i]
        for t in periodos:
            prob += w[i,t] == I[i,Li,t] - x[i,Li,t]

    # --- R8: valor mínimo de pedido por proveedor activo ---
    for j in proveedores:
        pj = [i for i in productos if (i, j) in data["compatibilidad"]]
        for t in periodos:
            prob += pulp.lpSum(data["precio"][i,j] * p[i,j,t] for i in pj) >= data["min_pedido"][j] * y[j,t]

    # --- R10: MOQ y activación binaria producto-proveedor ---
    for (i, j) in compat:
        for t in periodos:
            prob += p[i,j,t] >= data["moq"][i] * yp[i,j,t]
            prob += p[i,j,t] <= M * yp[i,j,t]
    for j in proveedores:
        pj = [i for i in productos if (i, j) in data["compatibilidad"]]
        for t in periodos:
            for i in pj:
                prob += y[j,t] >= yp[i,j,t]
            prob += y[j,t] <= pulp.lpSum(yp[i,j,t] for i in pj)

    # --- R11: capacidad total del almacén ---
    for t in periodos:
        prob += pulp.lpSum(I[i,a,t] * data["volumen"][i] for i in productos for a in edades[i]) <= data["capacidad"]

    # --- Resolver ---
    prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))
    status = pulp.LpStatus[prob.status]

    if prob.status != 1:
        return {"status": status}

    def v(var):
        return max(0.0, var.varValue or 0.0)

    compras  = {k: v(var) for k, var in p.items()}
    inv      = {k: v(var) for k, var in I.items()}
    merma    = {k: v(var) for k, var in w.items()}
    activacion = {k: round(v(var)) for k, var in y.items()}

    return {
        "status":       status,
        "costo_total":  pulp.value(prob.objective),
        "costo_compra": sum(data["precio"][i,j] * compras[i,j,t]   for (i,j) in compat   for t in periodos),
        "costo_fijo":   sum(data["costo_fijo"][j] * activacion[j,t] for j in proveedores   for t in periodos),
        "costo_mant":   sum(data["costo_mant"][i] * inv[i,a,t]      for i in productos for a in edades[i] for t in periodos),
        "costo_merma":  sum(data["costo_merma"][i] * merma[i,t]     for i in productos     for t in periodos),
        "compras":      compras,
        "inventario":   inv,
        "merma":        merma,
        "activacion":   activacion,
    }
