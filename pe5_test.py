import os, io, math, time, contextlib, tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

XLSX = os.path.join(HERE, "pe5", "PROSPER_DATA_5_ANGLE_0_LIQ_5_WCT_0_GOR_25.xlsx")
if not os.path.exists(XLSX):
    alt = os.path.join(HERE, "combined_shuffled_150k.xlsx")
    if os.path.exists(alt):
        XLSX = alt

INVALID = 3.4e38

ANGLE_IS_FROM_HORIZON = False
SIGMA_IS_mN_PER_m = True

from pe5_direct import run_pe5

def area_from_D_m(D_m: float) -> float:
    return math.pi * (D_m ** 2) / 4.0

def main():
    with contextlib.redirect_stdout(io.StringIO()):
        from pe5_direct import run_pe5

    need_cols = [
        "OIL_MASS_FLOW_RATE", "WATER_MASS_FLOW_RATE", "GAS_MASS_FLOW_RATE",
        "OIL_DENSITY", "WATER_DENSITY",
        "LIQUID_DENSITY", "LIQ_VISCOSITY",
        "GAS_DENSITY", "GAS_VISCOSITY",
        "DIAMETER", "ROUGHNESS",
        "ANGLE_FROM_VERTICAL", "GAS_LIQ_INTERFACIAL_TENSION",
        "PRESSURE",
        "FRICTION_GRADIENT", "GRAVITY_GRADIENT", "TOTAL_GRADIENT",
    ]

    if not os.path.exists(XLSX):
        raise SystemExit(f"Файл не найден: {XLSX}")

    df = pd.read_excel(XLSX)
    print(f"Источник данных: {XLSX}")
    print(f"Загружено строк: {len(df)}")
    df = df[~(df == INVALID).any(axis=1)].reset_index(drop=True)
    if df is None:
        print("После фильтра не осталось строк — проверь xlsx."); return


    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"В xlsx отсутствуют колонки: {miss}")

    calc_fric, calc_grav, orig_fric, orig_grav, calc_total, orig_total = [], [], [], [], [], []
    rows = []

    tracemalloc.start()
    t0 = time.perf_counter()

    for _, r in df.iterrows():
        def vol_from_mass(mdot, rho):
            md = float(mdot); rh = float(rho)
            return md / rh if (rh > 0 and md > 0) else 0.0

        q_oil_m3d = vol_from_mass(r["OIL_MASS_FLOW_RATE"],   r["OIL_DENSITY"])
        q_wat_m3d = vol_from_mass(r["WATER_MASS_FLOW_RATE"], r["WATER_DENSITY"])
        q_gas_m3d = vol_from_mass(r["GAS_MASS_FLOW_RATE"],   r["GAS_DENSITY"])

        rhoL = float(r["LIQUID_DENSITY"])
        muL  = float(r["LIQ_VISCOSITY"])
        rhoG = float(r["GAS_DENSITY"])
        muG  = float(r["GAS_VISCOSITY"])

        D_m     = float(r["DIAMETER"])
        eps_abs = float(r["ROUGHNESS"])
        eps_rel = (eps_abs / D_m) if D_m > 0 else 0.0

        raw_deg        = float(r["ANGLE_FROM_VERTICAL"]) 
        angle_used_deg = raw_deg if ANGLE_IS_FROM_HORIZON else (90.0 - raw_deg)

        sigma = float(r["GAS_LIQ_INTERFACIAL_TENSION"])
        sigma_mN_m = sigma if SIGMA_IS_mN_PER_m else (sigma * 1000.0)

        p_bar = float(r["PRESSURE"])

        with contextlib.redirect_stdout(io.StringIO()):
            res = run_pe5(
                q_oil_m3d=q_oil_m3d, q_wat_m3d=q_wat_m3d, q_gas_m3d=q_gas_m3d,
                rhoL_kgm3=rhoL, rhoG_kgm3=rhoG,
                muL_cP=muL, muG_cP=muG, sigma_mN_m=sigma_mN_m,
                diam_m=D_m, eps_rel=eps_rel, angle_deg=angle_used_deg,
                p_bar=p_bar,
                fric_on=1, grav_on=1, accel_on=1, flag2_value=0
            )

        A = area_from_D_m(D_m)
        Vsl = ((q_oil_m3d + q_wat_m3d) / 86400.0) / A if A > 0 else np.nan
        Vsg = (q_gas_m3d / 86400.0) / A if A > 0 else np.nan
        Ql  = (q_oil_m3d + q_wat_m3d)
        Qg_over_Ql = (q_gas_m3d / Ql) if Ql > 0 else np.nan

        true_fric = float(r["FRICTION_GRADIENT"]) 
        true_grav = float(r["GRAVITY_GRADIENT"])  
        true_total = float(r["TOTAL_GRADIENT"])


        fric_calc = float(res["psi_per_ft"]["fric"]) 
        grav_calc = float(res["psi_per_ft"]["grav"]) 
        total_calc = float(res["psi_per_ft"]["total"])


        calc_fric.append(fric_calc)
        calc_grav.append(grav_calc)
        orig_fric.append(true_fric)
        orig_grav.append(true_grav)
        calc_total.append(total_calc)
        orig_total.append(true_total)


        code   = int(res.get("code", -1))
        holdup = float(res.get("holdup", float("nan")))
        br = res.get("bubble_raw", {}) if isinstance(res, dict) else {}
        g_pipe   = float(br.get("g_pipe",   float("nan")))
        non_grav = float(br.get("non_grav", float("nan")))
        accel_ax = float(br.get("accel_ax", float("nan")))
        total_ax = float(br.get("total_ax", float("nan")))
        balance_eps = total_ax - (non_grav + accel_ax) if np.isfinite(total_ax) and np.isfinite(non_grav) and np.isfinite(accel_ax) else np.nan

        rows.append({
            "true_fric": true_fric, "calc_fric": fric_calc,
            "true_grav": true_grav, "calc_grav": grav_calc,
            "err_fric": fric_calc - true_fric, "err_grav": grav_calc - true_grav,
            "abs_err_fric": abs(fric_calc - true_fric),
            "code": code, "holdup": holdup,
            "Vsg_m_s": Vsg, "Vsl_m_s": Vsl, "Qg_over_Ql": Qg_over_Ql,
            "GAS_DENSITY": rhoG, "GAS_VISCOSITY": muG,
            "LIQUID_DENSITY": rhoL, "LIQ_VISCOSITY": muL,
            "PRESSURE": p_bar, "DIAMETER": D_m, "ROUGHNESS": eps_abs,
            "angle_used_deg": angle_used_deg, "ANGLE_FROM_VERTICAL": raw_deg,
            "g_pipe": g_pipe, "non_grav": non_grav, "accel_ax": accel_ax,
            "total_ax": total_ax, "balance_eps": balance_eps, "true_total": true_total, "calc_total": total_calc, "err_total": total_calc - true_total,
            "abs_err_total": abs(total_calc - true_total),

        })

    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        print("Пусто после фильтра INVALID.")
        return

    def mae(a, b):  return float(np.mean(np.abs(np.array(a) - np.array(b))))
    def mape(a, b):
        vals = []
        for x, y in zip(a, b):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            denom = max(abs(x) + abs(y), 1e-2)
            vals.append(200.0 * abs(x - y) / denom)
        return (sum(vals) / len(vals)) if vals else float("nan")


    print("\nПроверка точности (psi/ft):")
    print(f"  Gravity MAE : {mae(res_df['calc_grav'], res_df['true_grav']):.8f}")
    print(f"  Gravity MAPE: {mape(res_df['calc_grav'], res_df['true_grav']):.2f}%")
    print(f"  Friction MAE: {mae(res_df['calc_fric'], res_df['true_fric']):.8f}")
    print(f"  Friction MAPE:{mape(res_df['calc_fric'], res_df['true_fric']):.2f}%")
    print(f"  Total MAE   : {mae(res_df['calc_total'], res_df['true_total']):.8f}")
    print(f"  Total MAPE  : {mape(res_df['calc_total'], res_df['true_total']):.2f}%")
    print(f"\nPerformance: time {t1 - t0:.4f}s, peak mem {peak/1024:.2f} KB")

    if calc_grav and calc_fric and len(calc_grav) == len(orig_grav) == len(calc_fric) == len(orig_fric):
        plt.figure(figsize=(6,5))
        plt.scatter(orig_grav, calc_grav, s=20)
        lims = [min(orig_grav+calc_grav), max(orig_grav+calc_grav)]
        plt.plot(lims, lims, '--')
        plt.xlabel('True gravity (psi/ft)')
        plt.ylabel('Calc gravity (psi/ft)')
        plt.title('Gravity')
        plt.tight_layout()
        plt.show()

        errors_grav = [c - o for c, o in zip(calc_grav, orig_grav)]
        plt.figure(figsize=(6,5))
        plt.hist(errors_grav, bins=15)
        plt.axvline(0, linestyle='--')
        plt.title('Gravity error (psi/ft)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,5))
        plt.scatter(orig_fric, calc_fric, s=20)
        lims = [min(orig_fric+calc_fric), max(orig_fric+calc_fric)]
        plt.plot(lims, lims, '--')
        plt.xlabel('True friction (psi/ft)')
        plt.ylabel('Calc friction (psi/ft)')
        plt.title('Friction')
        plt.tight_layout()
        plt.show()

        errors_fric = [c - o for c, o in zip(calc_fric, orig_fric)]
        plt.figure(figsize=(6,5))
        plt.hist(errors_fric, bins=15)
        plt.axvline(0, linestyle='--')
        plt.title('Friction error (psi/ft)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
