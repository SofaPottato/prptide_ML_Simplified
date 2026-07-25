import re

import pandas as pd

PRED_SUFFIX = "__pred"
INVALID_RAW = -1    # mlTable 中代表「該組合沒給有效答案」(呼叫失敗) 的值
INVALID_FILL = 0.5  # 視為棄權票，落在 no(0) 與 yes(1) 正中間


def loadMLTable(path, labelCol="trueLabel", invalidFill=INVALID_FILL):
    """讀取 mlTable CSV，回傳「特徵 + y」的 DataFrame。

    - 第一欄視為 ID，設為 index。
    - 只取欄名以 `__pred` 結尾的模型預測欄當特徵，其餘 metadata 忽略。
    - 欄名保留「模型 + prompt 組合」，不可只留 prompt 組合 (不同模型的同名組合會撞名)。
    - 無效答案 (-1) 補成 invalidFill，代表該組合棄權；傳 None 則保留原值。
    - label 欄轉為整數並命名為 y，放在最後一欄。
    """

    # ---- 1. 讀檔，第一欄當 ID 設為 index ----
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("檔案 {} 欄位太少，至少要有 ID + label。".format(path))
    df = df.set_index(df.columns[0])

    # ---- 2. 取出 label 欄，轉為整數 ----
    if labelCol not in df.columns:
        raise ValueError(
            "在 {} 找不到 label 欄 '{}'，實際欄位有：{}".format(
                path, labelCol, list(df.columns)[:5]
            )
        )
    ySer = df[labelCol].astype(int)

    # ---- 3. 只留模型預測欄 (__pred 結尾)，排除 originalAns / passage 等 metadata ----
    predCols = [c for c in df.columns if str(c).endswith(PRED_SUFFIX)]
    featDf = df[predCols]

    # ---- 4. 整理欄名：去掉 __pred、只留英數與底線，重複則加 _2, _3 ... ----
    # 注意：不可用 rsplit("|") 只取 prompt 組合，否則 Qwen3:8b|EMO01 與 llama3.1:8b|EMO01
    #      會撞名成 EMO01 / EMO01_2，後續完全看不出哪一欄屬於哪個模型。
    nameMap = {}
    counts = {}
    for raw in featDf.columns:
        name = str(raw).strip()
        if name.endswith(PRED_SUFFIX):
            name = name[: -len(PRED_SUFFIX)]
        name = re.sub(r"[^0-9A-Za-z]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_") or "feat"

        if name in counts:
            counts[name] += 1
            name = "{}_{}".format(name, counts[name])
        else:
            counts[name] = 1
        nameMap[raw] = name

    # ---- 5. 套用欄名、確保特徵為數值 ----
    out = featDf.rename(columns=nameMap)
    out = out.apply(pd.to_numeric, errors="coerce")

    # ---- 6. 無效答案 (-1) 視為棄權票 ----
    # -1 的來源是 LLM 呼叫失敗 (connection failed)，不是模型判斷結果。
    # 若保留 -1，會讓模型以為「無效 < no < yes」有大小關係；且 minMax 正規化時
    # 一個 -1 就會把整欄壓成 {0, 0.5, 1}，使該欄的 no 變成 0.5、與其他純二元欄不一致。
    if invalidFill is not None:
        invalidNum = int((out == INVALID_RAW).sum().sum())
        if invalidNum:
            out = out.replace(INVALID_RAW, invalidFill)
            print("[loadMLTable] {}：{} 個無效答案 (-1) 已補為 {}".format(
                path, invalidNum, invalidFill))

    # ---- 7. 把 y 放到最後一欄 ----
    out["y"] = ySer.values
    return out
