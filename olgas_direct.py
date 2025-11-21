# olgas_direct.py
#
# OLGAS2000_SSH из libolgas2000.dll
# Интерфейс и структура результата такие же, как у run_pe5 из pe5_direct.py

import ctypes as ct
import os
import math
import platform
import struct

# ===== единицы, 1-в-1 как в pe5_direct.py =====

def m_to_in(m):        return m / 0.0254
def deg_to_rad(d):     return d * math.pi / 180.0
def m3d_to_ft3d(x):    return x * 35.314666721
def m3d_to_bbl_d(x):   return x * 6.28981
def bar_to_psi(b):     return b * 14.503773773
def psi_per_ft_to_pa_per_m(x): return x * 6894.8 / 0.3048

STUB_MODE = os.environ.get("OLGAS_STUB") == "1"


# ===== загрузка DLL OLGAS =====

def _resolve_dll_path():
    """Возвращает путь к DLL и проверяет, что он существует."""
    dll_path = os.environ.get(
        "OLGAS_DLL_PATH",
        os.path.join(os.path.dirname(__file__), "libolgas2000.dll"),
    )

    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"Не найден libolgas2000.dll по пути {dll_path}. "
            "Положи DLL рядом со скриптом или укажи OLGAS_DLL_PATH"
        )
    return dll_path


def _detect_pe_bits(dll_path):
    """Определяет разрядность PE-файла по полю OptionalHeader.Magic (0x10b/0x20b)."""

    with open(dll_path, "rb") as f:
        data = f.read(0x200)

    if data[:2] != b"MZ":
        return None

    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_off + 0x18 + 2 > len(data):
        return None

    magic = struct.unpack_from("<H", data, pe_off + 0x18)[0]
    if magic == 0x20B:
        return 64
    if magic == 0x10B:
        return 32
    return None


def _ensure_platform_matches(dll_path):
    """Сообщает реальную разрядность DLL и проверяет, что среда ей соответствует.

    Если выставить переменную окружения ``OLGAS_FORCE_LOAD=1``, проверка пропускается —
    это может понадобиться на хостах без нужной платформы, чтобы хотя бы выполнить
    отладочную компиляцию. При этом загрузка 32-битной DLL в 64-битном интерпретаторе
    всё равно, скорее всего, завершится ошибкой.
    """

    if os.environ.get("OLGAS_FORCE_LOAD") == "1":
        return

    dll_bits = _detect_pe_bits(dll_path)
    py_bits = struct.calcsize("P") * 8

    if os.name != "nt":
        raise OSError(
            f"libolgas2000.dll — {dll_bits or '?'}-битный PE, нужен Windows; "
            f"текущая платформа: {platform.system()} (Python {py_bits}-бит). "
            "Запусти скрипт под Windows с Python той же разрядности, что и DLL"
        )

    if dll_bits and dll_bits != py_bits:
        raise OSError(
            f"libolgas2000.dll — {dll_bits}-битная, Python сейчас {py_bits}-бит; "
            "нужно совпадение разрядностей. Установи Python соответствующей битности"
        )


def _load_olgas_dll():
    if STUB_MODE:
        return None

    dll_path = _resolve_dll_path()
    _ensure_platform_matches(dll_path)

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.dirname(dll_path))
    try:
        ct.windll.kernel32.SetDefaultDllDirectories(0x00001000)
    except Exception:
        # Для старых версий Windows это нормально
        pass

    return ct.CDLL(dll_path)


LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
olgas_dll = _load_olgas_dll()

# ===== типы и прототип =====

OLGAS_MSG = ct.c_ubyte * 16
DoublePtr = ct.POINTER(ct.c_double)
IntPtr    = ct.POINTER(ct.c_int)
UIntPtr   = ct.POINTER(ct.c_uint)

if not STUB_MODE:
    OLGAS2000_SSH = olgas_dll.OLGAS2000_SSH
    OLGAS2000_SSH.restype = None

    OLGAS2000_SSH.argtypes = (
        ct.POINTER(OLGAS_MSG),  # param_1 (msg[16])

        # param_2..param_6: double*
        DoublePtr, DoublePtr, DoublePtr, DoublePtr, DoublePtr,

        # param_7: undefined8 / opaque (8 байт!)
        ct.c_double,


        # param_8..param_14: double*
        DoublePtr, DoublePtr, DoublePtr, DoublePtr, DoublePtr, DoublePtr, DoublePtr,

        # param_15..param_20: int*/uint*/int*/...
        IntPtr, UIntPtr, IntPtr, IntPtr, IntPtr, IntPtr,

        # param_21..param_42: 22 x double*
        *([DoublePtr] * 22),

        # param_43..param_44: int*
        IntPtr, IntPtr,

        # param_45: double*
        DoublePtr,

        # param_46: int* (код)
        IntPtr,

        # param_47..param_48: double*
        DoublePtr, DoublePtr,
    )

# ===== выделение аргументов =====

def _alloc_olgas_all():
    msg = OLGAS_MSG()
    msg_p = ct.pointer(msg)

    # входы
    p2 = ct.c_double(0.0)
    p3 = ct.c_double(0.0)
    p4 = ct.c_double(0.0)
    p5 = ct.c_double(0.0)
    p6 = ct.c_double(0.0)
    p7 = ct.c_double(0.0)

    p8  = ct.c_double(0.0)
    p9  = ct.c_double(0.0)
    p10 = ct.c_double(0.0)
    p11 = ct.c_double(0.0)
    p12 = ct.c_double(0.0)
    p13 = ct.c_double(0.0)
    p14 = ct.c_double(0.0)

    p15 = ct.c_int(0)
    p16 = ct.c_uint(0)
    p17 = ct.c_int(0)
    p18 = ct.c_int(0)
    p19 = ct.c_int(0)
    p20 = ct.c_int(0)

    # выходы/вспомогательные
    p21  = ct.c_double(0.0)
    p22  = ct.c_double(0.0)
    p23  = ct.c_double(0.0)
    p24  = ct.c_double(0.0)
    p25  = ct.c_double(0.0)
    p26  = ct.c_double(0.0)
    p27  = ct.c_double(0.0)
    p28  = ct.c_double(0.0)
    p29  = ct.c_double(0.0)
    p30  = ct.c_double(0.0)
    p31  = ct.c_double(0.0)
    p32  = ct.c_double(0.0)
    p33  = ct.c_double(0.0)
    p34  = ct.c_double(0.0)
    p35  = ct.c_double(0.0)
    p36  = ct.c_double(0.0)
    p37  = ct.c_double(0.0)
    p38  = ct.c_double(0.0)
    p39  = ct.c_double(0.0)
    p40  = ct.c_double(0.0)
    p41  = ct.c_double(0.0)
    p42  = ct.c_double(0.0)

    p43 = ct.c_int(0)
    p44 = ct.c_int(0)

    p45 = ct.c_double(0.0)
    p46 = ct.c_int(0)

    # param_47 – массив из 7 даблов (как видно из кода)
    arr47 = (ct.c_double * 7)()
    p48   = ct.c_double(0.0)   # последний double*

    args = dict(
        param_1 = msg_p,
        param_2 = ct.byref(p2),
        param_3 = ct.byref(p3),
        param_4 = ct.byref(p4),
        param_5 = ct.byref(p5),
        param_6 = ct.byref(p6),
        param_7 = p7,
        param_8  = ct.byref(p8),
        param_9  = ct.byref(p9),
        param_10 = ct.byref(p10),
        param_11 = ct.byref(p11),
        param_12 = ct.byref(p12),
        param_13 = ct.byref(p13),
        param_14 = ct.byref(p14),
        param_15 = ct.byref(p15),
        param_16 = ct.byref(p16),
        param_17 = ct.byref(p17),
        param_18 = ct.byref(p18),
        param_19 = ct.byref(p19),
        param_20 = ct.byref(p20),

        param_21 = ct.byref(p21),
        param_22 = ct.byref(p22),
        param_23 = ct.byref(p23),
        param_24 = ct.byref(p24),
        param_25 = ct.byref(p25),
        param_26 = ct.byref(p26),
        param_27 = ct.byref(p27),
        param_28 = ct.byref(p28),
        param_29 = ct.byref(p29),
        param_30 = ct.byref(p30),
        param_31 = ct.byref(p31),
        param_32 = ct.byref(p32),
        param_33 = ct.byref(p33),
        param_34 = ct.byref(p34),
        param_35 = ct.byref(p35),
        param_36 = ct.byref(p36),
        param_37 = ct.byref(p37),
        param_38 = ct.byref(p38),
        param_39 = ct.byref(p39),
        param_40 = ct.byref(p40),
        param_41 = ct.byref(p41),
        param_42 = ct.byref(p42),

        param_43 = ct.byref(p43),
        param_44 = ct.byref(p44),
        param_45 = ct.byref(p45),
        param_46 = ct.byref(p46),

        param_47 = arr47,     # как double*
        param_48 = ct.byref(p48),
    )

    scalars = dict(
        msg=msg,
        p2=p2, p3=p3, p4=p4, p5=p5, p6=p6,
        p8=p8, p9=p9, p10=p10, p11=p11, p12=p12, p13=p13, p14=p14,
        p15=p15, p16=p16, p17=p17, p18=p18, p19=p19, p20=p20,
        p21=p21, p22=p22, p23=p23, p24=p24, p25=p25, p26=p26,
        p27=p27, p28=p28, p29=p29, p30=p30, p31=p31, p32=p32,
        p33=p33, p34=p34, p35=p35, p36=p36, p37=p37, p38=p38,
        p39=p39, p40=p40, p41=p41, p42=p42,
        p43=p43, p44=p44,
        p45=p45, p46=p46,
        arr47=arr47,
        p48=p48,
    )

    return args, scalars

def _stub_olgas(args, scalars):
    """Простейшая заглушка, позволяющая получить осмысленный вывод без DLL.

    Это не настоящая корреляция OLGAS. Значения подобраны, чтобы быть сравнимыми
    по порядку величины с примером PE5, и масштабируются от расхода.
    """

    # Извлекаем то, что положили перед вызовом
    q_liq = scalars["p11"].value
    q_gas = scalars["p12"].value
    angle = scalars["p4"].value

    total_flow = max(q_liq + q_gas, 1e-9)
    holdup = q_liq / total_flow

    # базовый градиент в psi/ft, слегка масштабируем по расходу и углу
    base_grad = 0.02 * (total_flow / 500.0)
    grav_factor = abs(math.sin(angle))

    total_psi_ft = base_grad * (0.4 + 0.6 * grav_factor)
    fric_psi_ft = total_psi_ft * 0.2

    scalars["p23"].value = holdup
    scalars["p36"].value = total_psi_ft
    scalars["p37"].value = fric_psi_ft

    # arr47 – просто нули для совместимости
    for i in range(len(scalars["arr47"])):
        scalars["arr47"][i] = 0.0

    # код завершения — 6 как в примере PE5
    scalars["p46"].value = 6


def _call_olgas(args, scalars):
    if STUB_MODE:
        _stub_olgas(args, scalars)
        return

    OLGAS2000_SSH(
        args["param_1"],
        args["param_2"], args["param_3"], args["param_4"], args["param_5"], args["param_6"],
        args["param_7"],
        args["param_8"], args["param_9"], args["param_10"], args["param_11"], args["param_12"],
        args["param_13"], args["param_14"],
        args["param_15"], args["param_16"], args["param_17"], args["param_18"],
        args["param_19"], args["param_20"],
        args["param_21"], args["param_22"], args["param_23"], args["param_24"], args["param_25"],
        args["param_26"], args["param_27"], args["param_28"], args["param_29"], args["param_30"],
        args["param_31"], args["param_32"], args["param_33"], args["param_34"], args["param_35"],
        args["param_36"], args["param_37"], args["param_38"], args["param_39"], args["param_40"],
        args["param_41"], args["param_42"],
        args["param_43"], args["param_44"],
        args["param_45"], args["param_46"],
        args["param_47"], args["param_48"],
    )

# ===== верхнеуровневая функция, как run_pe5 =====

def run_olgas(
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

    # 1) посчитаем те же гидродинамические штуки, что и в run_pe5
    Din_in     = m_to_in(diam_m)
    area_ft2   = (Din_in*Din_in * math.pi * 0.25) / 144.0
    q_liq_m3d  = q_oil_m3d + q_wat_m3d
    q_liq_ft3d = m3d_to_ft3d(q_liq_m3d)
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

    ANGLE_PROJ_EPS = 5e-4        
    GRAV_VERT_EXTRA_SCALE = 1.1223   
    sinA = math.sin(angle_rad)
    cosA = math.cos(angle_rad)
    near_vertical   = (abs(sinA) > 1.0 - ANGLE_PROJ_EPS) or (abs(cosA) > 1.0 - ANGLE_PROJ_EPS)
    near_horizontal = (abs(sinA) < ANGLE_PROJ_EPS) or (abs(cosA) < 1.0 - ANGLE_PROJ_EPS)

    # 2) подготовим аргументы OLGAS

    args, S = _alloc_olgas_all()

    # Сюда можно зашить тот же смысл, что и в PE5:
    # p2 – давление (psi), p3 – диаметр (ft), p4 – угол (рад),
    # p5/p6 – плотности, p8/p9 – вязкости, p10 – σ, p11/p12 – расходы.
    S["p2"].value = bar_to_psi(p_bar)
    S["p3"].value = Din_in / 12.0       # ft
    S["p4"].value = angle_rad

    S["p5"].value = rhoL_kgm3
    S["p6"].value = rhoG_kgm3

    S["p8"].value  = muL_cP
    S["p9"].value  = muG_cP
    S["p10"].value = sigma_mN_m
    S["p11"].value = q_liq_m3d
    S["p12"].value = q_gas_m3d
    S["p13"].value = eps_rel
    S["p14"].value = 0.0

    # флаги / режимы
    S["p17"].value = int(fric_on)
    S["p18"].value = int(grav_on)
    S["p19"].value = int(accel_on)
    S["p20"].value = int(flag2_value)

    _call_olgas(args, S)

    code = S["p46"].value  # это *param_48 в FUN_18005c2b0

    came_from_b0c6 = (code in (3, 4, 5, 6)) or (code == 8 and b0c6_rule)
    proj = cosA if came_from_b0c6 else sinA

    # ===== holdup =====
    # Предположение: p23 = volume fraction «основной» фазы (например, жидкости).
    holdup = S["p23"].value

    # ===== градиенты =====
    # p36 – полный градиент; p37 – без гравитации.
    total_psi_ft = S["p36"].value       # знак/масштаб зависят от DLL; возможно потребуется калибровка
    fric_accel   = S["p37"].value

    # Восстановим гравитационную часть как разность:
    grav_psi_ft  = total_psi_ft - fric_accel

    # Пока акселерацию считаем 0, а всё fric+accel вешаем на fric.
    # Когда поймёшь, что лежит в arr47[..], легко разделишь.
    fric_psi_ft  = fric_accel
    accel_psi_ft = 0.0

    # Коррекция как в PE5 (мультипликаторы) – по желанию:
    fric_psi_ft  *= 0.999935
    grav_psi_ft  *= 0.99989

    fric_Pa_m   = psi_per_ft_to_pa_per_m(fric_psi_ft)
    grav_Pa_m   = psi_per_ft_to_pa_per_m(grav_psi_ft)
    accel_Pa_m  = psi_per_ft_to_pa_per_m(accel_psi_ft)
    total_Pa_m  = psi_per_ft_to_pa_per_m(total_psi_ft)

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
            "fric":  fric_Pa_m,
            "grav":  grav_Pa_m,
            "accel": accel_Pa_m,
            "total": total_Pa_m,
        },
        "diag": {
            "proj": ("cos" if came_from_b0c6 else "sin"),
            "b0c6_rule": bool(b0c6_rule),
            "Vsl_ft_s": Vsl_fts,
            "Vsg_ft_s": Vsg_fts,
            "Din_in": Din_in,
            "area_ft2": area_ft2,
            # сырые данные, чтобы можно было самому разметить
            "raw_olgas": {
                "p23": S["p23"].value,
                "p24": S["p24"].value,
                "p25": S["p25"].value,
                "p26": S["p26"].value,
                "p27": S["p27"].value,
                "p28": S["p28"].value,
                "p29": S["p29"].value,
                "p30": S["p30"].value,
                "p31": S["p31"].value,
                "p32": S["p32"].value,
                "p33": S["p33"].value,
                "p34": S["p34"].value,
                "p35": S["p35"].value,
                "p36": S["p36"].value,
                "p37": S["p37"].value,
                "p38": S["p38"].value,
                "p39": S["p39"].value,
                "p40": S["p40"].value,
                "p41": S["p41"].value,
                "p42": S["p42"].value,
                "arr47": [float(x) for x in S["arr47"]],
            },
        }
    }

    return res


if __name__ == "__main__":
    r = run_olgas(
        q_oil_m3d=50.0, q_wat_m3d=0.0, q_gas_m3d=500.0,
        rhoL_kgm3=850.0, rhoG_kgm3=5.0,
        muL_cP=2.0, muG_cP=0.02, sigma_mN_m=20.0,
        diam_m=0.10, eps_rel=0.001, angle_deg=90.0,
        p_bar=1500.0,
        fric_on=1, grav_on=1, accel_on=1,
        flag2_value=0,
    )
    print("code:", r["code"], "holdup:", r["holdup"])
    print("psi/ft:", r["psi_per_ft"])
