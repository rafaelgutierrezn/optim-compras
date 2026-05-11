import argparse
import sys
from tabulate import tabulate
from data_loader import load_instance
from model import build_and_solve


def print_report(result, data):
    print("\n=== RESUMEN DE COSTES ===")
    print(tabulate([
        ["Estado",             result["status"]],
        ["Coste total",        f"{result['costo_total']:.2f}"],
        ["  Compra",           f"{result['costo_compra']:.2f}"],
        ["  Fijo proveedores", f"{result['costo_fijo']:.2f}"],
        ["  Mantenimiento",    f"{result['costo_mant']:.2f}"],
        ["  Merma/caducidad",  f"{result['costo_merma']:.2f}"],
    ], tablefmt="simple"))

    filas = [
        {"Periodo": t, "Producto": i, "Proveedor": j, "Cantidad": round(qty, 2),
         "Coste": round(data["precio"][i, j] * qty, 2)}
        for (i, j, t), qty in result["compras"].items() if qty > 1e-4
    ]
    if filas:
        print("\n=== PLAN DE COMPRAS ===")
        print(tabulate(sorted(filas, key=lambda r: (r["Periodo"], r["Producto"])),
                       headers="keys", tablefmt="simple", showindex=False))

    mermas = [
        {"Periodo": t, "Producto": i, "Merma": round(qty, 2)}
        for (i, t), qty in result["merma"].items() if qty > 1e-4
    ]
    if mermas:
        print("\n=== MERMA / CADUCIDAD ===")
        print(tabulate(sorted(mermas, key=lambda r: (r["Periodo"], r["Producto"])),
                       headers="keys", tablefmt="simple", showindex=False))
    else:
        print("\nSin merma en ningún periodo.")


def main():
    parser = argparse.ArgumentParser(description="Optimización de compras — MILP")
    parser.add_argument("--instance", default="data/instances/demo_small")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data   = load_instance(args.instance)
    result = build_and_solve(data, verbose=args.verbose)
    print_report(result, data)

    if result["status"] != "Optimal":
        sys.exit(1)


if __name__ == "__main__":
    main()
