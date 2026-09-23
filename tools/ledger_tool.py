# -*- coding: utf-8 -*-
"""
车间台账查询工具
读取ledger.json，根据区域名查询完整业务信息
"""
import json
import os
from config import LEDGER_PATH, LOG_ENABLED


def _log(msg: str):
    if LOG_ENABLED:
        print("[LEDGER] " + msg)


def query_workshop_ledger(area_name: str) -> dict:
    """
    查询车间台账
    :param area_name: 区域名称
    :return: {
        "success": bool, "area": str,
        "person_in_charge": str, "contact": str,
        "risk_base_level": str, "forbidden_objects": list,
        "hazard_mapping": dict, "safety_rule": str,
        "risk_rule": dict, "error": str
    }
    """
    result = {
        "success": False,
        "area": area_name,
        "person_in_charge": "",
        "contact": "",
        "risk_base_level": "",
        "forbidden_objects": [],
        "hazard_mapping": {},
        "safety_rule": "",
        "risk_rule": {},
        "error": ""
    }

    if not os.path.exists(LEDGER_PATH):
        result["error"] = "台账文件不存在: %s" % LEDGER_PATH
        _log(result["error"])
        return result

    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)

        # 精确匹配
        area_data = ledger_data.get(area_name)

        # 模糊匹配
        if area_data is None:
            for key in ledger_data.keys():
                if area_name in key or key in area_name:
                    area_data = ledger_data[key]
                    result["area"] = key
                    break

        if area_data is None:
            result["error"] = "区域[%s]不在台账中" % area_name
            _log(result["error"])
            return result

        result["success"] = True
        result["person_in_charge"] = area_data.get("person_in_charge", "")
        result["contact"] = area_data.get("contact", "")
        result["risk_base_level"] = area_data.get("risk_base_level", "")
        result["forbidden_objects"] = area_data.get("forbidden_objects", [])
        result["hazard_mapping"] = area_data.get("hazard_mapping", {})
        result["safety_rule"] = area_data.get("safety_rule", "")
        result["risk_rule"] = area_data.get("risk_rule", {})

        _log("查询成功: %s -> %s" % (area_name, result["person_in_charge"]))
        return result

    except json.JSONDecodeError as e:
        result["error"] = "台账JSON解析失败: %s" % str(e)
        _log(result["error"])
        return result
    except Exception as e:
        result["error"] = "台账查询异常: %s: %s" % (type(e).__name__, str(e))
        _log(result["error"])
        return result
