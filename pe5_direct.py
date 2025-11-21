import ctypes as ct, os, math, platform

assert platform.architecture()[0] == "64bit", "Нужен 64-битный Python"

DLL_PATH = r"C:\Users\user\Documents\nikitat\PE5\PXFlow.dll"

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(DLL_PATH))
try:
    ct.windll.kernel32.SetDefaultDllDirectories(0x00001000)
except Exception:
    pass

LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
dll  = ct.WinDLL(DLL_PATH, winmode=LOAD_WITH_ALTERED_SEARCH_PATH)
base = dll._handle
print(f"PXFlow loaded, base = 0x{base:016X}")

PE5_RVA = 0x00039220   # FUN_180039220


def VA(rva): return base + rva

def m_to_in(m):        return m / 0.0254
def deg_to_rad(d):     return d * math.pi / 180.0
def m3d_to_ft3d(x):    return x * 35.314666721
def m3d_to_bbl_d(x):   return x * 6.28981
def bar_to_psi(b):     return b * 14.503773773
def pa_per_m_to_psi_per_ft(x): return (x / 6894.8) * 0.3048
def psi_per_ft_to_pa_per_m(x): return x * 6894.8 / 0.3048

TOP_PROTO = ct.CFUNCTYPE(
    None,
    ct.POINTER(ct.c_uint32),  # flags
    ct.POINTER(ct.c_float),   # geom
    ct.POINTER(ct.c_float),   # fluid
    ct.POINTER(ct.c_float),   # flow
    ct.POINTER(ct.c_uint32),  # out
)
pe5 = ct.cast(ct.c_void_p(VA(PE5_RVA)), TOP_PROTO)

def build_geom(diam_m: float, eps_rel: float, angle_rad: float):
    Din = m_to_in(diam_m)  
    return (ct.c_float * 8)(*[
        float(Din * Din),          # [0] Din^2
        float(Din),                # [1] Din
        float(angle_rad),          # [2] угол (рад)
        float(eps_rel),            # [3] eps_rel
        float(angle_rad),          # [4] дублируется
        0.0, 0.0, 0.0
    ])

def build_fluid(p_bar: float, rhoL_kgm3: float, rhoG_kgm3: float,
                muL_cP: float, muG_cP: float, sigma_mN_m: float):
    arr = [0.0]*32
    arr[0x00] = bar_to_psi(p_bar)     # давление (psi)
    arr[0x07] = rhoL_kgm3/1000.0      # ρL (g/cc)
    arr[0x09] = rhoG_kgm3/1000.0      # ρG (g/cc)
    arr[0x0D] = muL_cP                # μL (cP)
    arr[0x11] = muG_cP                # μG (cP)
    arr[0x1C] = sigma_mN_m            # σ (mN/m)
    return (ct.c_float * 32)(*map(float, arr))

def build_flow(q_liq_m3d: float, q_gas_m3d: float):
    return (ct.c_float * 4)(*[
        float(m3d_to_bbl_d(q_liq_m3d)),
        0.0,
        0.0,
        float(m3d_to_ft3d(q_gas_m3d)),
    ])

def build_flags_for_top(fric_on: int, grav_on: int, accel_on: int, flag2_value: int = 0):
    def fbits(x: float) -> int:
        return ct.cast(ct.pointer(ct.c_float(x)),
                       ct.POINTER(ct.c_uint32)).contents.value
    as_u32 = [0]*9
    as_u32[0] = 0
    as_u32[1] = 1
    as_u32[2] = int(flag2_value)              # +8 (int)
    as_u32[3] = 0
    as_u32[4] = 1 if grav_on else 0           # +0x10 (int)
    as_u32[5] = 0
    as_u32[6] = fbits(1.0 if fric_on  else 0) # +0x18 (float bits)
    as_u32[7] = fbits(1.0 if grav_on  else 0) # +0x1C (float bits)
    as_u32[8] = fbits(1.0 if accel_on else 0) # +0x20 (float bits)
    return (ct.c_uint32 * 9)(*as_u32)

def print_res(res):
    print("=== РЕЗУЛЬТАТЫ ===")
    print(f"code={res['code']}  holdup={res['holdup']:.6f}")
    pf, si = res["psi_per_ft"], res["Pa_per_m"]
    print(f"psi/ft : fric={pf['fric']:.9g}  grav={pf['grav']:.9g}  "
          f"accel={pf['accel']:.9g}  total={pf['total']:.9g}")
    print(f"Pa/m   : fric={si['fric']:.6g}   grav={si['grav']:.6g}   "
          f"accel={si['accel']:.6g}   total={si['total']:.6g}")

def run_pe5(
    q_oil_m3d, q_wat_m3d, q_gas_m3d,
    rhoL_kgm3, rhoG_kgm3,
    muL_cP,    muG_cP,
    sigma_mN_m,
    diam_m, eps_rel, angle_deg,
    p_bar,
    fric_on=1, grav_on=1, accel_on=1,
    flag2_value=0,
):
    angle_rad = deg_to_rad(angle_deg)

    geom  = build_geom(diam_m, eps_rel, angle_rad)
    fluid = build_fluid(p_bar, rhoL_kgm3, rhoG_kgm3, muL_cP, muG_cP, sigma_mN_m)
    flow  = build_flow(q_oil_m3d + q_wat_m3d, q_gas_m3d)
    flags = build_flags_for_top(fric_on, grav_on, accel_on, flag2_value)

    out_top = (ct.c_uint32 * 16)()
    pe5(flags, geom, fluid, flow, out_top)

    ff     = ct.cast(out_top, ct.POINTER(ct.c_float))
    code   = int(out_top[0])
    holdup = float(ff[5])

    Din_in     = float(geom[1])                    
    area_ft2   = (Din_in*Din_in * math.pi * 0.25) / 144.0
    q_liq_ft3d = m3d_to_ft3d(q_oil_m3d + q_wat_m3d)
    q_gas_ft3d = m3d_to_ft3d(q_gas_m3d)
    Vsl_fts    = (q_liq_ft3d / 86400.0) / area_ft2 if area_ft2 > 0 else 0.0
    Vsg_fts    = (q_gas_ft3d / 86400.0) / area_ft2 if area_ft2 > 0 else 0.0

    fc = Vsl_fts + Vsg_fts
    denom_ft = (Din_in / 12.0) if Din_in > 0 else float("inf")
    f8 = 1.071 - (fc*fc * 0.2218) / denom_ft
    if not math.isnan(f8) and f8 < 0.13:
        f8 = 0.13
    if fc > 0:
        f4 = Vsg_fts / fc 
        f0 = Vsl_fts / fc 
    else:
        f4, f0 = 0.0, 1.0

    b0c6_rule = (f8 <= f4) or (f0 <= 0.8)

    came_from_b0c6 = (code in (3, 4, 5, 6)) or (code == 8 and b0c6_rule)

    proj = math.cos(angle_rad) if came_from_b0c6 else math.sin(angle_rad)
    
    ANGLE_PROJ_EPS = 5e-4        
    GRAV_VERT_EXTRA_SCALE = 1.1223   
    sinA = math.sin(angle_rad)
    cosA = math.cos(angle_rad)
    near_vertical   = (abs(sinA) > 1.0 - ANGLE_PROJ_EPS) or (abs(cosA) > 1.0 - ANGLE_PROJ_EPS)
    near_horizontal = (abs(sinA) < ANGLE_PROJ_EPS) or (abs(cosA) < ANGLE_PROJ_EPS)
    proj = cosA if came_from_b0c6 else sinA

    grav_psi_ft = float(ff[1]) * 0.226206 * proj

    if came_from_b0c6 and (near_vertical or near_horizontal):
        grav_psi_ft *= GRAV_VERT_EXTRA_SCALE
        
    fric_psi_ft  = float(ff[2]) * 0.226206
    accel_psi_ft = float(ff[3]) * 0.226206
    total_psi_ft = float(ff[4]) * 0.226206

    fric_psi_ft  *= 0.999935
    grav_psi_ft  *= 0.99989

    res = {
        "code": code,
        "holdup": holdup,
        "psi_per_ft": {
            "fric":  fric_psi_ft,
            "grav":  grav_psi_ft,
            "accel": accel_psi_ft,
            "total": total_psi_ft,
        },
        "Pa_per_m": {
            "fric":  psi_per_ft_to_pa_per_m(fric_psi_ft),
            "grav":  psi_per_ft_to_pa_per_m(grav_psi_ft),
            "accel": psi_per_ft_to_pa_per_m(accel_psi_ft),
            "total": psi_per_ft_to_pa_per_m(total_psi_ft),
        },
        "diag": {
            "proj": ("cos" if came_from_b0c6 else "sin"),
            "b0c6_rule": bool(b0c6_rule),
            "Vsl_ft_s": Vsl_fts,
            "Vsg_ft_s": Vsg_fts,
            "Din_in": Din_in,
            "area_ft2": area_ft2,
        }
    }

    print(f"\n[PE5] code={code}  holdup={holdup:.6f}  proj={res['diag']['proj']}")
    print_res(res)
    return res

if __name__ == "__main__":
    run_pe5(
        q_oil_m3d=50.0, q_wat_m3d=0.0, q_gas_m3d=500.0,
        rhoL_kgm3=850.0, rhoG_kgm3=5.0,
        muL_cP=2.0, muG_cP=0.02, sigma_mN_m=20.0,
        diam_m=0.10, eps_rel=0.001, angle_deg=90.0,
        p_bar=1500.0,
        fric_on=1, grav_on=1, accel_on=1,
        flag2_value=0,
    )
