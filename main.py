import re
import os
import json
import requests
import numpy as np
import xarray as xr
import pandas as pd
from herbie import Herbie
from urllib.parse import urljoin
from datetime import datetime, timedelta, timezone

# =============================================================================
# 全域設定
# =============================================================================
LAND_MASK_PATH  = "environment-2down/land_mask.npy"
OUTPUT_NPZ      = "V4_model_data.npz"
STATE_FILE      = "forecast_list.json"   # 記錄上次成功的週期
DOWNSAMPLE_FACTOR = 2
GRAVITY           = 9.80665

TARGET_VAR_ORDER = [
    't2m', 'msl', 'u10', 'v10',
    'sst',
    'z_850', 'u_850', 'v_850', 't_850',
    'z_500', 'u_500', 'v_500', 'q_500',
    'u_200', 'v_200',
]

FIXED_STATS = {
    't2m':   {'mean': 280.0,    'std': 25.0},
    'msl':   {'mean': 101000.0, 'std': 1400.0},
    'u10':   {'mean': 0.0,      'std': 10.0},
    'v10':   {'mean': 0.0,      'std': 10.0},
    'sst':   {'mean': 287.0,    'std': 12.0},
    'z_850': {'mean': 14000.0,  'std': 1600.0},
    'u_850': {'mean': 1.0,      'std': 10.0},
    'v_850': {'mean': 0.0,      'std': 7.0},
    't_850': {'mean': 275.0,    'std': 16.0},
    'z_500': {'mean': 55000.0,  'std': 3500.0},
    'u_500': {'mean': 7.0,      'std': 15.0},
    'v_500': {'mean': 0.0,      'std': 15.0},
    'q_500': {'mean': 0.0009,   'std': 0.0012},
    'u_200': {'mean': 12.0,     'std': 20.0},
    'v_200': {'mean': 0.0,      'std': 15.0},
}

# cfgrib 篩選鍵對照表
GRIB_MAP = {
    't2m': {'shortName': '2t'},
    'msl': {'shortName': 'prmsl'},
    'u10': {'shortName': '10u'},
    'v10': {'shortName': '10v'},
    'sst': {'shortName': 't', 'typeOfLevel': 'surface'},
    't':   {'shortName': 't'},
    'z':   {'shortName': 'gh'},
    'u':   {'shortName': 'u'},
    'v':   {'shortName': 'v'},
    'q':   {'shortName': 'q'},
}

# GRIB2 下載篩選語法
SEARCH_PATTERN = (
    ":("
    "TMP:2 m above ground|"
    "PRMSL:mean sea level|"
    "UGRD:10 m above ground|"
    "VGRD:10 m above ground|"
    "TMP:surface|"
    "(HGT|UGRD|VGRD|TMP):850 mb|"
    "(HGT|UGRD|VGRD|SPFH):500 mb|"
    "(UGRD|VGRD):200 mb"
    ")"
)

# ── 模型設定：新增模型在此擴充即可 ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "FNV3": {
        "json_path":   "active_typhoon/cyclone_data_fnv3.json",
        "csv_path":    "active_typhoon/FNV3.csv",
        "url_template": (
            "https://deepmind.google.com/science/weatherlab/download/cyclones/"
            "FNV3/ensemble/paired/csv/FNV3_{time}_paired.csv"
        ),
        "time_offset_h": -6,   # 相對 forecast_list.json 的時間偏移
    },
    "GENC": {
        "json_path":   "active_typhoon/cyclone_data_genc.json",
        "csv_path":    "active_typhoon/GENC.csv",
        "url_template": (
            "https://deepmind.google.com/science/weatherlab/download/cyclones/"
            "GENC/ensemble/paired/csv/GENC_{time}_paired.csv"
        ),
        "time_offset_h": -6,
    },
}


# =============================================================================
# 工具函式
# =============================================================================

def get_latest_available_cycle():
    """計算目前 UTC 時間下最新可用的 GFS 初始場週期。"""
    buffer = timedelta(hours=4)          # GFS 發布延遲約 3–4 小時
    now    = datetime.now(timezone.utc) - buffer
    for cycle_h in [18, 12, 6, 0]:
        if now.hour >= cycle_h:
            return now.strftime("%Y%m%d"), f"{cycle_h:02d}"
    prev = now - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "18"


def get_previous_cycle(date_str, cycle_str):
    """取得前一個 GFS 週期（往前 6 小時）。"""
    t = datetime.strptime(f"{date_str}{cycle_str}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    p = t - timedelta(hours=6)
    return p.strftime("%Y%m%d"), p.strftime("%H")


def cycle_id(date_str, cycle_str):
    return f"{date_str}_{cycle_str}Z"


def get_forecast_list(json_path="forecast_list.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            v4_list = data.get("EGTY_V4", [])
            folders = []
            for item in v4_list:
                folders.append(item["folder"].replace("-", "").replace("T", "_") + "Z")
            return folders
    except json.JSONDecodeError:
        print("讀取json錯誤")
        return []
    
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

_FMTS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",    "%Y-%m-%dT%H", "%Y%m%d%H",
)

def parse_dt(raw: str):
    for fmt in _FMTS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None

# =============================================================================
# 資料下載
# =============================================================================

def download_gfs(date, cycle, hour="000"):
    """使用 Herbie 下載指定 GFS 週期的篩選 GRIB 檔。"""
    tag = f"{date} {cycle}Z F{hour}"
    print(f"[下載] {tag}")
    H = Herbie(
        date=f"{date} {cycle}:00",
        model='gfs',
        product='pgrb2.0p25',
        fxx=int(hour),
        save_dir="./herbie_cache",
    )
    try:
        path = H.download(search=SEARCH_PATTERN)
    except Exception as e:
        print(f"[錯誤] Herbie 下載失敗: {e}")
        return None

    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[錯誤] 下載的檔案無效: {path}")
        if path and os.path.exists(path):
            os.remove(path)
        return None

    print(f"[完成] {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path

def download_model_data(model_name: str) -> str | None:
    cfg = MODEL_CONFIGS[model_name]

    try:
        if not os.path.exists("forecast_list.json"):
            print(f"❌ [{model_name}] 找不到 forecast_list.json，跳過下載。")
            return None

        file_time = load_json("forecast_list.json").get("EGTY_V4")[0].get("folder")
        dt_obj    = parse_dt(file_time)
        if not dt_obj:
            print(f"❌ [{model_name}] 無法解析時間字串: {file_time}")
            return None

        dt_obj      += timedelta(hours=cfg["time_offset_h"])
        url_time_str = dt_obj.strftime("%Y_%m_%dT%H_00")
        url          = cfg["url_template"].format(time=url_time_str)
        save_path    = cfg["csv_path"]

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path

    except Exception as e:
        print(f"❌ [{model_name}] 下載失敗: {e}")
        return None

def convert_csv_to_json(csv_path: str, output_path: str, model_name: str = ""):
    tag = f"[{model_name}] " if model_name else ""

    df = pd.read_csv(csv_path, skiprows=6)
    df = df.where(pd.notnull(df), None)

    result = {}
    for track_id, track_group in df.groupby("track_id"):
        result[track_id] = []
        for sample_id, sample_group in track_group.groupby("sample"):
            sample_data = {"sample_id": float(sample_id), "data_points": []}
            for _, row in sample_group.iterrows():
                sample_data["data_points"].append({
                    "valid_time": row["valid_time"],
                    "coordinates": {
                        "lat": row["lat"],
                        "lon": row["lon"],
                    },
                    "intensity": {
                        "mslp_hpa":       row["minimum_sea_level_pressure_hpa"],
                        "max_wind_knots": row["maximum_sustained_wind_speed_knots"],
                    },
                })
            result[track_id].append(sample_data)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ {tag}JSON 已儲存至：{output_path}")
    print(f"{'='*60}")

# =============================================================================
# GRIB 解析
# =============================================================================

def parse_grib(filepath):
    """從 GRIB 檔逐一讀取 TARGET_VAR_ORDER 中的變數並堆疊為陣列。"""
    arrays, names = [], []
    lats = lons = time_val = None

    for var_name in TARGET_VAR_ORDER:
        base = var_name.split('_')[0]
        fk   = GRIB_MAP[base].copy()
        if '_' in var_name:
            fk['typeOfLevel'] = 'isobaricInhPa'
            fk['level']       = int(var_name.split('_')[1])

        ds = None
        try:
            ds  = xr.open_dataset(filepath, engine='cfgrib',
                                  backend_kwargs={'indexpath': '',
                                                  'filter_by_keys': fk})
            key = list(ds.data_vars)[0]
            arr = ds[key].squeeze().values

            if var_name.startswith('z_'):       # 位勢高度 → 位能 (m²/s²)
                arr = arr * GRAVITY

            arrays.append(arr)
            names.append(var_name)

            if lats is None:
                lats     = ds['latitude'].values
                lons     = ds['longitude'].values
                time_val = ds['time'].values

            print(f"  ✓ {var_name}")
        except Exception as e:
            print(f"  ✗ {var_name}: {e}")
        finally:
            if ds is not None:
                ds.close()

    if not arrays:
        return None, None, None, None, None

    return np.stack(arrays, axis=0), lats, lons, time_val, names

def parse_atcf_to_json(file_content, typhoon_id):
    if not file_content:
        return None
        
    lines = [line.strip() for line in file_content.decode('utf-8').split('\n') if line.strip()]
    
    for line in reversed(lines):
        # 使用 strip 清除每個欄位的頭尾空白
        columns = [item.strip() for item in line.split(',')]
        
        if len(columns) < 10:
            continue
            
        try:
            # 關鍵修正：先 strip 再檢查是否為數字
            wind_raw = columns[8]
            mslp_raw = columns[9]
            
            if not (wind_raw.isdigit() and mslp_raw.isdigit()):
                continue

            # 處理緯度 (例如 "265N")
            raw_lat = columns[6] # 修正：根據你貼的數據，緯度在索引 6
            lat_val = "".join(filter(str.isdigit, raw_lat))
            lat = float(lat_val) / 10.0
            if 'S' in raw_lat: lat = -lat
            
            # 處理經度 (例如 "1495E")
            raw_lon = columns[7] # 修正：根據你貼的數據，經度在索引 7
            lon_val = "".join(filter(str.isdigit, raw_lon))
            lon = float(lon_val) / 10.0
            if 'W' in raw_lon: lon = -lon
            
            wind = float(wind_raw)
            mslp = float(mslp_raw)
            
            # 氣旋名稱處理
            name = "INVEST"
            if len(columns) > 27 and columns[27].strip():
                name = columns[27].strip()

            t = columns[2]
            formatted_time = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:00:00"

            return {
                "name": name,
                "valid_time": formatted_time,
                "coordinates": {"lat": lat, "lon": lon},
                "intensity": {
                    "mslp_hpa": mslp,
                    "max_wind_knots": wind
                }
            }
        except Exception:
            continue
            
    return None
# =============================================================================
# 主流程
# =============================================================================

def gfs_main():
    # ── 步驟 0：決定週期，判斷是否需要重新下載 ────────────────────────────────
    latest_date,  latest_cycle  = get_latest_available_cycle()
    prev_date,    prev_cycle    = get_previous_cycle(latest_date, latest_cycle)
    current_cid = cycle_id(latest_date, latest_cycle)

    print(f"最新週期: {current_cid}")
    print(f"前一週期: {cycle_id(prev_date, prev_cycle)}")
    print(f"輸出檔案: {OUTPUT_NPZ}")
    if current_cid in get_forecast_list():
        print(f"[跳過] {current_cid} 已於上次執行時處理完畢，無需重複下載。")
        return

    # ── 步驟 1：載入陸地遮罩 ──────────────────────────────────────────────────
    try:
        land_mask = np.load(LAND_MASK_PATH)
        print(f"陸地遮罩載入完成，形狀: {land_mask.shape}")
    except FileNotFoundError:
        print(f"[錯誤] 找不到陸地遮罩: {LAND_MASK_PATH}")
        return

    # ── 步驟 2：下載兩個時間步 ────────────────────────────────────────────────
    tasks = [
        {'date': prev_date,   'cycle': prev_cycle,   'hour': '000'},
        {'date': latest_date, 'cycle': latest_cycle, 'hour': '000'},
    ]
    paths = []
    for t in tasks:
        p = download_gfs(t['date'], t['cycle'], t['hour'])
        if p is None:
            print("[錯誤] 下載失敗，中止執行。")
            return
        paths.append(p)

    # ── 步驟 3：解析 GRIB ─────────────────────────────────────────────────────
    all_data, all_times = [], []
    channel_names = None

    for fp in paths:
        print(f"\n解析: {fp}")
        data, lats, lons, tv, names = parse_grib(fp)
        if data is None:
            print("[錯誤] 解析失敗，中止執行。")
            return
        if channel_names is None:
            channel_names = names
        all_data.append(data)
        all_times.append(tv)

    data = np.stack(all_data, axis=0)       # (T, C, H, W)
    print(f"\n堆疊完成: {data.shape}")

    # ── 步驟 4：預處理 ────────────────────────────────────────────────────────
    # 移除 90°S 多餘列
    if data.shape[2] == 721:
        data = data[:, :, :-1, :]
        lats = lats[:-1]

    # 2× 下採樣
    T, C, H, W  = data.shape
    nH = H // DOWNSAMPLE_FACTOR
    nW = W // DOWNSAMPLE_FACTOR
    data = (data
            .reshape(T, C, nH, DOWNSAMPLE_FACTOR, nW, DOWNSAMPLE_FACTOR)
            .mean(axis=(3, 5)))
    lats_ds = lats[DOWNSAMPLE_FACTOR // 2 :: DOWNSAMPLE_FACTOR]
    lons_ds = lons[DOWNSAMPLE_FACTOR // 2 :: DOWNSAMPLE_FACTOR]
    print(f"下採樣後: {data.shape}")

    # 標準化
    for i, name in enumerate(channel_names):
        if name in FIXED_STATS:
            m, s = FIXED_STATS[name]['mean'], FIXED_STATS[name]['std']
            data[:, i] = (data[:, i] - m) / s
        else:
            print(f"[警告] {name} 未設定 FIXED_STATS，跳過標準化。")

    # SST 陸地遮罩
    if 'sst' in channel_names:
        idx = channel_names.index('sst')
        if land_mask.shape == data.shape[2:]:
            for t in range(data.shape[0]):
                data[t, idx] = np.where(land_mask == 1, 0.0, data[t, idx])
            print("SST 陸地遮罩已套用。")
        else:
            print(f"[警告] 陸地遮罩形狀 {land_mask.shape} ≠ 資料 {data.shape[2:]}，跳過。")

    # ── 步驟 5：儲存 ──────────────────────────────────────────────────────────
    time_strs = np.array([
        np.datetime_as_string(tv, unit='h') for tv in all_times
    ])

    np.savez_compressed(
        OUTPUT_NPZ,
        data          = data.astype(np.float32),
        times         = time_strs,
        lats          = lats_ds.astype(np.float32),
        lons          = lons_ds.astype(np.float32),
        channel_names = np.array(channel_names),
    )
    print(f"\n✅ 已儲存 {OUTPUT_NPZ}  →  形狀 {data.shape}")

def google_main():
    os.makedirs("active_typhoon", exist_ok=True)
    try:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_tracks")
        os.makedirs(output_dir, exist_ok=True)

        for model_name in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[model_name]
            csv_path = download_model_data(model_name)
            convert_csv_to_json(csv_path, cfg["json_path"], model_name)
            os.remove(csv_path)
    except Exception as e:
        print(f"沒獲取到有效資料: {e}")

def active_main():
    base_url = "https://www.natyphoon.top/atcf/temp/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0...)"
    }
    
    response = requests.get(base_url, headers=headers, timeout=15)
    response.raise_for_status()
    html_text = response.text
    
    fn_match = re.search(r'const\s+FN\s*=\s*\[(.*?)\]\s*;', html_text, re.S)
    lm_match = re.search(r'const\s+LM\s*=\s*\[(.*?)\]\s*;', html_text, re.S)

    fn_list = [x.strip().strip('"').strip("'") for x in fn_match.group(1).split(",") if x.strip()]
    lm_list = [x.strip().strip('"').strip("'") for x in lm_match.group(1).split(",") if x.strip()]
    
    files = list(zip(fn_list, lm_list))
    cutoff = datetime.now() - timedelta(hours=24)
    
    # --- 關鍵修正：將字典放在迴圈外 ---
    active_list = {}

    print(f"🕓 目前時間: {datetime.now():%Y/%m/%d %H:%M}")
    print(f"📅 截止線: {cutoff:%Y/%m/%d %H:%M}\n")

    for typhoon_data_id, updata_time in files:
        try:
            dt = datetime.strptime(updata_time, "%Y/%m/%d %H:%M")
            
            if dt > cutoff:
                file_url = urljoin(base_url, typhoon_data_id)
                # 取得編號 (例如: bwp042026.dat -> WP042026)
                typhoon_id = str(typhoon_data_id)[1:-4].upper()  
                
                print(f"→ {typhoon_id:10} | 下載中...", end="\r")
                
                file_resp = requests.get(file_url, headers=headers, timeout=30)
                file_resp.raise_for_status()

                # 解析並加入字典
                data = parse_atcf_to_json(file_resp.content, typhoon_id)
                if data:
                    active_list[typhoon_id] = data
                
                # 存檔邏輯
                save_dir = os.path.join("active_typhoon", typhoon_id)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, typhoon_data_id), "wb") as f:
                    f.write(file_resp.content)
                
                output_path = "active_typhoon/active_list.json" # 你可以自定義路徑
                    
                # 確保資料夾存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if active_list:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(active_list, f, ensure_ascii=False, indent=4)
                    print(f"\n✅ 所有活動氣旋數據已儲存至：{output_path}")
                else:
                    print("\nℹ️ 沒有符合條件的活動氣旋數據，未產生 JSON。")


                print(f"✅ {typhoon_id:10} | 處理完成{' '*10}")
        
        except Exception as e:
            print(f"\n⚠️ 錯誤: {typhoon_data_id} -> {e}")
            continue
        

if __name__ == "__main__":
    gfs_main()
    google_main()
    active_main()
