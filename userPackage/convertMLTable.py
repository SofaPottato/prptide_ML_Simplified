
import re

import pandas as pd


def loadMLTable(path, labelCol="trueLabel"):

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("檔案 {} 欄位太少，至少要有 ID + label。".format(path))
    idCol = df.columns[0]
    df = df.set_index(idCol)
    df.index.name = idCol
    if labelCol not in df.columns:
        raise ValueError(
            "在 {} 找不到 label 欄 '{}'，實際欄位有：{}".format(path, labelCol, list(df.columns)[:5])
        )
    ySer = df[labelCol].astype(int)
    featDf = df.drop(columns=[labelCol])

    nameMap = {}
    used = {}
    for raw in featDf.columns:
        name = str(raw).strip()
        if name.endswith("__pred"):
            name = name[:-len("__pred")]
        if "|" in name:
            name = name.rsplit("|", 1)[1]
        name = re.sub(r"[^0-9A-Za-z]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if name == "":
            name = "feat"
        clean = name
        if clean in used:
            used[clean] += 1
            clean = "{}_{}".format(clean, used[clean])
        else:
            used[clean] = 1
        nameMap[raw] = clean

    # ---- 套用欄名對照、確保特徵是數值、把 y 放到最後一欄 ----
    out = featDf.rename(columns=nameMap).copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out["y"] = ySer.values
    return out
