"""
Модуль математичного моделювання одноступеневої та двоступеневої
зворотноосмотичної установки (ЗОУ / RO).
Розроблено на основі коду RO-2022.ipynb.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def zmishyvach(Gr: float, Ck: float, Ci: float, Gv: float) -> Tuple[float, float]:
    """
    Змішувач вхідного потоку вихідної води з потоком рециркуляту концентрату.
    
    Параметри:
        Gr: витрата рециркуляту, м3/год
        Ck: концентрація солей у рециркуляті, мг/л
        Ci: концентрація солей у вихідній воді, мг/л
        Gv: сумарна витрата води, що подається на мембрану, м3/год
        
    Повертає:
        Gi: витрата вихідної води (забір зі свердловини/джерела), м3/год
        Cv: концентрація солей у потоці на вході в мембрану, мг/л
    """
    Gi = Gv - Gr
    Cv = (Gr * Ck + Gi * Ci) / Gv if Gv > 0 else 0.0
    return Gi, Cv


def rozdiluvach(Gk: float, Xr: float) -> Tuple[float, float]:
    """
    Розділювач потоку концентрату на рециркулят і скид (дренаж).
    
    Параметри:
        Gk: витрата концентрату з мембрани, м3/год
        Xr: коефіцієнт рециркуляції (частка рециркуляту від концентрату, 0 <= Xr <= 1)
        
    Повертає:
        Gr: витрата рециркуляту, м3/год
        Gs: витрата скиду (концентрат у дренаж), м3/год
    """
    Gr = Gk * Xr
    Gs = Gk - Gr
    return Gr, Gs


def membrana(Gv: float, Cv: float, Xz: float, Xp: float) -> Tuple[float, float, float, float]:
    """
    Мембранний елемент / модуль зворотного осмосу.
    
    Параметри:
        Gv: витрата вхідної води на мембрану, м3/год
        Cv: концентрація солей на вході в мембрану, мг/л
        Xz: селективність мембрани / коефіцієнт затримання солей (0 < Xz < 1)
        Xp: вихід пермеату мембранного модуля (коефіцієнт конверсії, 0 < Xp < 1)
        
    Повертає:
        Gp: витрата пермеату (знесоленої води), м3/год
        Cp: концентрація солей у пермеаті, мг/л
        Gk: витрата концентрату, м3/год
        Ck: концентрація солей у концентраті, мг/л
    """
    Gp = Gv * Xp
    Gk = Gv - Gp
    Cp = Cv * (1.0 - Xz)
    Ck = (Gv * Cv - Gp * Cp) / Gk if Gk > 0 else 0.0
    return Gp, Cp, Gk, Ck


def zmish2(Gr1: float, Ck1: float, Gr2: float, Ck2: float) -> Tuple[float, float]:
    """
    Змішувач двох потоків рециркуляту (з 1-го та 2-го ступенів ЗОУ).
    
    Параметри:
        Gr1: витрата рециркуляту з 1-го ступеня, м3/год
        Ck1: концентрація солей у концентраті 1-го ступеня, мг/л
        Gr2: витрата рециркуляту з 2-го ступеня, м3/год
        Ck2: концентрація солей у концентраті 2-го ступеня, мг/л
        
    Повертає:
        Gr: сумарна витрата об'єднаного рециркуляту, м3/год
        Ck: середня концентрація солей в об'єднаному рециркуляті, мг/л
    """
    Gr = Gr1 + Gr2
    if Gr <= 0:
        return 0.0, 0.0
    Ck = (Gr1 * Ck1 + Gr2 * Ck2) / Gr
    return Gr, Ck


def calculate_single_stage(
    Ci: float = 1000.0,
    Gv: float = 100.0,
    Xz: float = 0.99,
    Xp: float = 0.20,
    Xr: float = 0.77,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    """
    Ітераційний розрахунок одноступеневої схеми зворотного осмосу (ЗО-1).
    """
    Cv = 1.2 * Ci  # Початкове наближення
    history = []
    converged = False

    Gp = Cp = Gk = Ck = Gr = Gs = Gi = 0.0

    for it in range(1, max_iter + 1):
        prev_Cv = Cv
        Gp, Cp, Gk, Ck = membrana(Gv, Cv, Xz, Xp)
        Gr, Gs = rozdiluvach(Gk, Xr)
        Gi, Cv = zmishyvach(Gr, Ck, Ci, Gv)
        
        diff = abs(Cv - prev_Cv)
        history.append({
            "iter": it,
            "Cv": Cv,
            "Cp": Cp,
            "Ck": Ck,
            "diff": diff
        })
        
        if diff < tol and it > 3:
            converged = True
            break

    # Перевірка матеріального балансу
    # 1) Баланс по воді: Вхід (Gi) = Вихід (Gp + Gs)
    water_in = Gi
    water_out = Gp + Gs
    water_delta = water_in - water_out

    # 2) Баланс по розчиненій солі: LP = Gi * Ci; RP = Gp * Cp + Gs * Ck
    salt_in = Gi * Ci
    salt_out = Gp * Cp + Gs * Ck
    salt_delta = salt_in - salt_out
    is_balance_ok = abs(salt_delta) <= 1e-2

    # Загальний коефіцієнт виходу пермеату установки (відносно вихідної води Gi)
    recovery_system = (Gp / Gi * 100.0) if Gi > 0 else 0.0
    rejection_system = ((Ci - Cp) / Ci * 100.0) if Ci > 0 else 0.0

    return {
        "converged": converged,
        "iterations": len(history),
        "history": history,
        "inputs": {
            "Ci": Ci,
            "Gv": Gv,
            "Xz": Xz,
            "Xp": Xp,
            "Xr": Xr,
        },
        "flows": {
            "Gi": Gi,
            "Gv": Gv,
            "Gp": Gp,
            "Gk": Gk,
            "Gr": Gr,
            "Gs": Gs,
        },
        "concentrations": {
            "Ci": Ci,
            "Cv": Cv,
            "Cp": Cp,
            "Ck": Ck,
        },
        "balance": {
            "water_in": water_in,
            "water_out": water_out,
            "water_delta": water_delta,
            "salt_in": salt_in,
            "salt_out": salt_out,
            "salt_delta": salt_delta,
            "is_balance_ok": is_balance_ok,
        },
        "kpi": {
            "recovery_system": recovery_system,
            "rejection_system": rejection_system,
            "concentration_factor": (Ck / Ci) if Ci > 0 else 0.0,
        }
    }


def calculate_two_stage(
    Ci: float = 1000.0,
    Gv: float = 100.0,
    Xz1: float = 0.99,
    Xp1: float = 0.20,
    Xr1: float = 0.20,
    Xz2: float = 0.99,
    Xp2: float = 0.20,
    Xr2: float = 0.25,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> Dict[str, Any]:
    """
    Ітераційний розрахунок двоступеневої схеми зворотного осмосу (ЗОУ-2)
    зі знесоленням за пермеатом (пермеат 1-го ступеня живить 2-й ступінь,
    концентрати обох ступенів рециркулюють та змішуються з вихідною водою).
    """
    Cv1 = 1.1 * Ci  # Початкове наближення
    history = []
    converged = False

    Gp1 = Cp1 = Gk1 = Ck1 = Gr1 = Gs1 = 0.0
    Gp2 = Cp2 = Gk2 = Ck2 = Gr2 = Gs2 = 0.0
    Grr = Ckk = Gii = 0.0

    for it in range(1, max_iter + 1):
        prev_Cv1 = Cv1
        # Ступінь 1
        Gp1, Cp1, Gk1, Ck1 = membrana(Gv, Cv1, Xz1, Xp1)
        # Ступінь 2 (живлення пермеатом 1-го ступеня)
        Gp2, Cp2, Gk2, Ck2 = membrana(Gp1, Cp1, Xz2, Xp2)
        
        # Розділювачі концентратів
        Gr1, Gs1 = rozdiluvach(Gk1, Xr1)
        Gr2, Gs2 = rozdiluvach(Gk2, Xr2)
        
        # Змішування рециркулятів обох ступенів
        Grr, Ckk = zmish2(Gr1, Ck1, Gr2, Ck2)
        
        # Змішування рециркуляту з вихідною водою
        Gii, Cv1 = zmishyvach(Grr, Ckk, Ci, Gv)

        diff = abs(Cv1 - prev_Cv1)
        history.append({
            "iter": it,
            "Cv1": Cv1,
            "Cp1": Cp1,
            "Ck1": Ck1,
            "Cp2": Cp2,
            "Ck2": Ck2,
            "diff": diff
        })

        if diff < tol and it > 3:
            converged = True
            break

    # Баланси
    # 1) Баланс по воді: Вхід (Gii) = Вихід (Gp2 + Gs1 + Gs2)
    water_in = Gii
    water_out = Gp2 + Gs1 + Gs2
    water_delta = water_in - water_out

    # 2) Баланс по солі: LP = Gii * Ci; RP = Gp2 * Cp2 + Gs1 * Ck1 + Gs2 * Ck2
    salt_in = Gii * Ci
    salt_out = Gp2 * Cp2 + Gs1 * Ck1 + Gs2 * Ck2
    salt_delta = salt_in - salt_out
    is_balance_ok = abs(salt_delta) <= 1e-2

    # Відхилення Cv від Ci (умова з Colab: Cv == Ci)
    cv_ci_diff = Cv1 - Ci
    cv_ci_pct = (cv_ci_diff / Ci * 100.0) if Ci > 0 else 0.0

    recovery_system = (Gp2 / Gii * 100.0) if Gii > 0 else 0.0
    rejection_system = ((Ci - Cp2) / Ci * 100.0) if Ci > 0 else 0.0

    return {
        "converged": converged,
        "iterations": len(history),
        "history": history,
        "inputs": {
            "Ci": Ci,
            "Gv": Gv,
            "Xz1": Xz1,
            "Xp1": Xp1,
            "Xr1": Xr1,
            "Xz2": Xz2,
            "Xp2": Xp2,
            "Xr2": Xr2,
        },
        "flows": {
            "Gii": Gii,
            "Gv": Gv,
            "Gp1": Gp1,
            "Gk1": Gk1,
            "Gr1": Gr1,
            "Gs1": Gs1,
            "Gp2": Gp2,
            "Gk2": Gk2,
            "Gr2": Gr2,
            "Gs2": Gs2,
            "Grr": Grr,
        },
        "concentrations": {
            "Ci": Ci,
            "Cv1": Cv1,
            "Cp1": Cp1,
            "Ck1": Ck1,
            "Cp2": Cp2,
            "Ck2": Ck2,
            "Ckk": Ckk,
        },
        "balance": {
            "water_in": water_in,
            "water_out": water_out,
            "water_delta": water_delta,
            "salt_in": salt_in,
            "salt_out": salt_out,
            "salt_delta": salt_delta,
            "is_balance_ok": is_balance_ok,
            "cv_ci_diff": cv_ci_diff,
            "cv_ci_pct": cv_ci_pct,
        },
        "kpi": {
            "recovery_system": recovery_system,
            "rejection_system": rejection_system,
            "stage1_recovery": (Gp1 / Gv * 100.0),
            "stage2_recovery": (Gp2 / Gp1 * 100.0) if Gp1 > 0 else 0.0,
        }
    }


def solve_xr2_for_target_cv(
    Ci: float = 1000.0,
    Gv: float = 100.0,
    Xz1: float = 0.99,
    Xp1: float = 0.20,
    Xr1: float = 0.20,
    Xz2: float = 0.99,
    Xp2: float = 0.20,
    target_cv: float = 1000.0,
    xr2_bounds: Tuple[float, float] = (0.01, 0.99)
) -> float:
    """
    Підбирає коефіцієнт рециркуляції 2-го ступеня Xr2 методом бісекції,
    щоб задовольнити умову Cv1 == target_cv (як зазначено в завданні Colab).
    """
    low, high = xr2_bounds
    best_xr2 = 0.25
    
    for _ in range(30):
        mid = (low + high) / 2.0
        res = calculate_two_stage(
            Ci=Ci, Gv=Gv,
            Xz1=Xz1, Xp1=Xp1, Xr1=Xr1,
            Xz2=Xz2, Xp2=Xp2, Xr2=mid
        )
        val = res["concentrations"]["Cv1"]
        diff = val - target_cv
        
        if abs(diff) < 1e-4:
            return mid
            
        # Якщо збільшення Xr2 збільшує або зменшує Cv1
        res_high = calculate_two_stage(
            Ci=Ci, Gv=Gv,
            Xz1=Xz1, Xp1=Xp1, Xr1=Xr1,
            Xz2=Xz2, Xp2=Xp2, Xr2=high
        )
        if (res_high["concentrations"]["Cv1"] - target_cv) * diff > 0:
            high = mid
        else:
            low = mid
        best_xr2 = mid
        
    return best_xr2
