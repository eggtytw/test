import json
import re
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from shapely.validation import make_valid
import os
import shutil
import glob
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cartopy.crs as ccrs

# ── Saffir-Simpson 色彩分級 ────────────────────────────────────────────────────
CATEGORIES = [
    {"name": "Category 5",     "min_kt": 137, "color": "#F700FF"},
    {"name": "Category 4",     "min_kt": 113, "color": "#C90000"},
    {"name": "Category 3",     "min_kt":  96, "color": "#FF5100"},
    {"name": "Category 2",     "min_kt":  83, "color": "#D29E00"},
    {"name": "Category 1",     "min_kt":  64, "color": "#CAC600"},
    {"name": "Tropical Storm", "min_kt":  34, "color": "#00A424"},
    {"name": "Depression",     "min_kt":  25, "color": "#0047A4"},
    {"name": "Disturbance",    "min_kt":   0, "color": "#7C7C7C"},
]

# ── 模型設定：新增模型在此擴充即可 ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "FNV3": {
        "json_path":   "active_typhoon/cyclone_data_fnv3.json",
    },
    "GENC": {
        "json_path":   "active_typhoon/cyclone_data_genc.json",
    },
    "EGTY_V4": {
        "json_path":   "active_typhoon/cyclone_data_egty4.json",
    },
}

ERROR_RADII = {  # 誤差圈半徑（公里）對應預報小時數
    0:   50,
    24:  100,
    48:  150,
    72:  200,
    96:  300,
    120: 500,
    192: 700,
}

# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def get_error_radius(hours: float) -> float:
    times = sorted(ERROR_RADII.keys())
    return np.interp(hours, times, [ERROR_RADII[t] for t in times])

def get_color(wind_kt: float) -> str:
    for cat in CATEGORIES:
        if wind_kt >= cat["min_kt"]:
            return cat["color"]
    return CATEGORIES[-1]["color"]

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

def unwrap_lons(lons) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.array(lons, dtype=float))))

# ══════════════════════════════════════════════════════════════════════════════
# 地圖範圍計算
# ══════════════════════════════════════════════════════════════════════════════

def compute_map_params(tracks: list, pad_deg: float = 5.0):
    all_lons = np.concatenate([t["lons"] for t in tracks])
    all_lats = np.concatenate([t["lats"] for t in tracks])

    lon_min_uw, lon_max_uw = all_lons.min(), all_lons.max()
    lat_min,    lat_max    = all_lats.min(), all_lats.max()

    lon_center = (lon_min_uw + lon_max_uw) / 2.0
    lat_center = (lat_min    + lat_max)    / 2.0

    raw_half_lon = (lon_max_uw - lon_min_uw) / 2.0 + pad_deg
    raw_half_lat = (lat_max    - lat_min)    / 2.0 + pad_deg

    target_aspect  = 4.0 / 3.0
    current_aspect = raw_half_lon / raw_half_lat

    if current_aspect > target_aspect:
        half_lon = raw_half_lon
        half_lat = raw_half_lon / target_aspect
    else:
        half_lat = raw_half_lat
        half_lon = raw_half_lat * target_aspect

    central_lon = ((lon_center + 180) % 360) - 180
    extent_proj = (
        lon_center - half_lon,
        lon_center + half_lon,
        lat_center - half_lat,
        lat_center + half_lat,
    )
    return central_lon, extent_proj, lon_center, lat_center

# ══════════════════════════════════════════════════════════════════════════════
# JSON 結構解析
# ══════════════════════════════════════════════════════════════════════════════

def parse_tracks(samples: list) -> list:
    tracks = []
    for sample in samples:
        pts = sample["data_points"]
        if not pts:
            continue
        times, lats, lons, winds, pressure = [], [], [], [], []
        for p in pts:
            times.append(p["valid_time"])
            lats.append(p["coordinates"]["lat"])
            lons.append(p["coordinates"]["lon"])
            winds.append(p["intensity"]["max_wind_knots"])
            pressure.append(p["intensity"]["mslp_hpa"])

        tracks.append({
            "sample_id":    sample["sample_id"],
            "times":        times,
            "lats":         np.array(lats,     dtype=float),
            "lons":         unwrap_lons(lons),
            "winds":        np.array(winds,    dtype=float),
            "pressure":     np.array(pressure, dtype=float),
            "max_wind":     float(np.nanmax(winds)),
            "min_pressure": float(np.nanmin(pressure)),
        })
    return tracks

def ensemble_mean(tracks: list, min_members: int = 20):
    if not tracks:
        return None
    
    # 找出所有成員中最長的預報步數
    max_n = max(len(t["lats"]) for t in tracks)
    
    mean_lats, mean_lons, mean_winds = [], [], []
    mean_times = []

    for i in range(max_n):
        # 篩選出在此時步 (i) 仍然存活（有數據）的成員
        lats_at_i = [t["lats"][i] for t in tracks if i < len(t["lats"])]
        lons_at_i = [t["lons"][i] for t in tracks if i < len(t["lons"])]
        winds_at_i = [t["winds"][i] for t in tracks if i < len(t["winds"])]
        
        # ── 核心修改：存活成員數必須大於指定門檻（預設 20）才計入平均 ──
        if len(lats_at_i) > min_members:
            mean_lats.append(np.nanmean(lats_at_i))
            mean_lons.append(np.nanmean(lons_at_i))
            mean_winds.append(np.nanmean(winds_at_i))
            
            # 取得該時步的時間戳記
            for t in tracks:
                if i < len(t["times"]):
                    mean_times.append(t["times"][i])
                    break
        else:
            # 一旦存活成員數不足，直接中斷後續時步的計算。
            # 這能確保平均路徑與誤差圈（Cone）在成員分歧過大、代表性不足時優雅地收尾。
            break

    # 如果連前幾個時步的成員數都不夠，則無法產生平均路徑
    if not mean_lats:
        return None

    return {
        "lats":         np.array(mean_lats),
        "lons":         np.array(mean_lons),
        "winds":        np.array(mean_winds),
        "times":        mean_times,
        "max_wind":     float(np.nanmean([t["max_wind"]  for t in tracks])),
        "min_pressure": float(np.nanmean([t["min_pressure"] for t in tracks])),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 繪圖輔助
# ══════════════════════════════════════════════════════════════════════════════

def get_geodesic_circle(lon, lat, radius_km, num_points=128):
    """以測地線建構圓形多邊形，num_points 越高邊緣越平滑。"""
    geod = Geod(ellps='WGS84')
    angles = np.linspace(0, 360, num_points, endpoint=False)
    lons, lats, _ = geod.fwd(
        [lon] * num_points, [lat] * num_points,
        angles, [radius_km * 1000] * num_points,
    )
    lons = np.array(lons)
    lons = ((lons - lon + 180) % 360) + lon - 180
    return Polygon(zip(lons, lats))


def build_cone_polygon(path_lons, path_lats, path_times, radius_fn,
                       interp_steps: int = 50, smooth_deg: float = 30):
    """
    優化後的誤差圈建構器：
    1. 將 interp_steps 提高，增加中間圓的密度。
    2. 加大 smooth_deg (0.8 ~ 1.2)，這能更強力地抹平凹凸。
    """
    t0 = path_times[0]

    # ── 步驟 1：收集各時步的 (lon, lat, radius) ───────────────────────────────
    waypoints = []
    for i, (lon, lat, t) in enumerate(zip(path_lons, path_lats, path_times)):
        if t is None:
            continue
        hours = (t - t0).total_seconds() / 3600
        waypoints.append((lon, lat, radius_fn(i, hours)))

    if len(waypoints) < 2:
        return None

    # ── 步驟 2：插值填滿間隙 ────────────────────────────────
    circles = []
    for j in range(len(waypoints) - 1):
        lon0, lat0, r0 = waypoints[j]
        lon1, lat1, r1 = waypoints[j + 1]
        
        # 根據兩點距離動態決定插值次數，或者直接使用較高的固定值
        steps = interp_steps 
        for k in range(steps): 
            alpha = k / steps
            circles.append(get_geodesic_circle(
                lon0 + alpha * (lon1 - lon0),
                lat0 + alpha * (lat1 - lat0),
                r0  + alpha * (r1  - r0),
            ))
            
    # 加入最後一個時步
    lon_f, lat_f, r_f = waypoints[-1]
    circles.append(get_geodesic_circle(lon_f, lat_f, r_f))

    # ── 步驟 3：聯集與平滑處理 ──────────────────────────────────────
    # 這裡使用 buffer(smooth_deg).buffer(-smooth_deg) 是一種閉合(Closing)運算
    # 它會填滿外部輪廓的凹角。
    full_cone = unary_union(circles)
    
    if not full_cone.is_empty:
        # 增加 buffer 量能讓線條更圓滑，但要注意不要加太大導致形狀失真
        full_cone = full_cone.buffer(smooth_deg, quad_segs=8).buffer(-smooth_deg, quad_segs=8)

    return full_cone if not full_cone.is_empty else None


def _add_cone_to_axes(ax, cone_polygon):
    ax.add_geometries(
        [cone_polygon],
        crs=ccrs.PlateCarree(),
        facecolor="#F3FF4A",
        edgecolor='#cc3300',
        alpha=0.4,
        zorder=2,
    )


def draw_forecast_cone(ax, track, data_crs):
    """EGTY_V4 / 單成員用：以固定 ERROR_RADII 繪製預報扇形。"""
    times = [parse_dt(t) for t in track["times"]]
    cone = build_cone_polygon(
        track["lons"], track["lats"], times,
        radius_fn=lambda i, hours: get_error_radius(hours),
    )
    if cone is not None:
        _add_cone_to_axes(ax, cone)


def compute_member_spread_radius(tracks: list, time_idx: int,
                                 fixed_fallback_km: float) -> float:
    """
    計算 time_idx 時步各成員相對於平均位置的最大測地距離（km）。
    若有效成員數 < 總數一半，回退至 fixed_fallback_km。
    最小值限制為 ERROR_RADII[0]（t=0 的固定半徑）。
    """
    n_total = len(tracks)
    lats, lons = [], []
    for tr in tracks:
        if time_idx < len(tr["lats"]):
            lats.append(tr["lats"][time_idx])
            lons.append(tr["lons"][time_idx])

    # ── 有效成員不足一半：回退 ────────────────────────────────────────────────
    if len(lats) < n_total / 2 or len(lats) < 2:
        return fixed_fallback_km

    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))

    geod = Geod(ellps='WGS84')
    max_dist_km = 0.0
    for la, lo in zip(lats, lons):
        _, _, dist_m = geod.inv(mean_lon, mean_lat, lo, la)
        max_dist_km = max(max_dist_km, dist_m / 1000.0)

    return max(max_dist_km, float(ERROR_RADII[0]))


def draw_member_spread_cone(ax, tracks: list, mean: dict):
    """
    多成員用：以各時步成員空間離散度為半徑，沿集合平均路徑繪製預報扇形。
    成員數 < 一半時，該時步改用固定 ERROR_RADII。
    """
    if not mean:
        return

    times = [parse_dt(t) for t in mean["times"]]

    def radius_fn(i, hours):
        fixed = get_error_radius(hours)
        return compute_member_spread_radius(tracks, i, fixed)

    cone = build_cone_polygon(mean["lons"], mean["lats"], times, radius_fn)
    if cone is not None:
        _add_cone_to_axes(ax, cone)


def draw_track(ax, lons, lats, winds, lw=0.9, alpha=0.70, zorder=2, data_crs=None):
    if data_crs is None:
        data_crs = ccrs.PlateCarree()
    for i in range(len(lons) - 1):
        c = get_color((winds[i] + winds[i + 1]) / 2)
        ax.plot(
            [lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
            color=c, linewidth=lw, alpha=alpha,
            transform=data_crs, zorder=zorder, solid_capstyle="round",
        )


def draw_mean_track(ax, mean: dict, data_crs, interval_h: int = 24, show_labels: bool = True):
    lons, lats, winds = mean["lons"], mean["lats"], mean["winds"]
    
    # 繪製起點 X 標記（保留起點，方便辨識颱風出發位置）
    ax.plot(lons[0], lats[0], "kx", markersize=11, markeredgewidth=2.5,
            transform=data_crs, zorder=7)
            
    parsed = [parse_dt(r) for r in mean["times"]]
    t0, done = parsed[0], set()
    for i, t in enumerate(parsed):
        if t is None or t0 is None:
            continue
        elh = (t - t0).total_seconds() / 3600
        lh  = round(elh / interval_h) * interval_h
        if lh > 0 and lh not in done:
            done.add(lh)
            
            # ── 當開啟標註時，才繪製黑點（黑色正方形）與文字標籤 ──
            if show_labels:
                # 繪製時間節點（黑點）
                ax.plot(lons[i], lats[i], "ks", markersize=7,
                        transform=data_crs, zorder=7)
                
                # 繪製文字標籤
                wind_val = winds[i]
                wind_str = f"{int(round(wind_val))}kt" if not np.isnan(wind_val) else ""
                label_text = f"+{int(lh)}h\n{wind_str}".strip()

                ax.text(
                    lons[i] + 0.6, lats[i], label_text,
                    fontsize=8.5, fontweight="bold", color="black",
                    transform=data_crs, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                )

def draw_member_list(fig, tracks: list, mean: dict):
    items = sorted(tracks, key=lambda t: (t["max_wind"], -t["min_pressure"]), reverse=True)
    ax2   = fig.add_axes([0.843, 0.12, 0.15, 0.8])
    ax2.axis("off")
    for i in range(len(items) + 1):
        y = 1.0 - i / 47
        if i < len(items):
            t     = items[i]
            color = get_color(t["max_wind"])
            ax2.text(0.05, y, f'#{int(t["sample_id"] + 1)}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.25, y, f'{t["max_wind"]:.1f}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.45, y,
                     f'kt  {t["min_pressure"]:.1f}hPa' if t["min_pressure"] < 1000
                     else f'kt  {int(t["min_pressure"])}hPa',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
        else:
            color = get_color(mean["max_wind"])
            ax2.text(0.00, y - 0.03, "mean",
                     fontsize=10, color="black", va="center", ha="left", fontfamily="monospace")
            ax2.text(0.25, y - 0.03, f'{mean["max_wind"]:.1f}',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")
            ax2.text(0.45, y - 0.03,
                     f'kt  {mean["min_pressure"]:.1f}hPa' if mean["min_pressure"] < 1000
                     else f'kt  {int(mean["min_pressure"])}hPa',
                     fontsize=10, color=color, va="center", ha="left", fontfamily="monospace")


# ══════════════════════════════════════════════════════════════════════════════
# 地圖底圖建構（共用）
# ══════════════════════════════════════════════════════════════════════════════

def _build_map_figure(tracks: list, wide: bool = False):
    """
    建立 Figure + GeoAxes，回傳 (fig, ax, data_crs, to_standard_fn)。
    wide=True：地圖佔滿全寬（用於誤差圈圖，無右欄）。
    wide=False：地圖留右欄給成員列表。
    """
    central_lon, extent_proj, _, _ = compute_map_params(tracks)
    proj     = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()
    lon_min_e, lon_max_e, lat_min_e, lat_max_e = extent_proj

    def to_standard(lon_uw):
        return ((lon_uw - central_lon + 180) % 360) - 180 + central_lon

    fig = plt.figure(figsize=(13.5, 9), facecolor="white")
    ax_rect = [0.01, 0.04, 0.99, 0.89] if wide else [0.01, 0.04, 0.85, 0.89]
    ax = fig.add_axes(ax_rect, projection=proj)
    ax.set_extent(
        [to_standard(lon_min_e), to_standard(lon_max_e), lat_min_e, lat_max_e],
        crs=data_crs,
    )

    fig.add_artist(matplotlib.lines.Line2D(
        [0.039, 0.99], [0.937, 0.937],
        transform=fig.transFigure, color="black", lw=1, zorder=10,
    ))

    ax.add_feature(cfeature.LAND,      facecolor="#C8C8C8", edgecolor="#888888", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor="#E8F4FA", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#777777", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#AAAAAA", zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor="#E8F4FA", edgecolor="#AAAAAA", linewidth=0.3, zorder=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5,
                      linestyle="--", crs=data_crs)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    handles = [mpatches.Patch(facecolor=c["color"], label=c["name"]) for c in CATEGORIES]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              framealpha=0.88, edgecolor="#AAAAAA", facecolor="white")

    return fig, ax, data_crs, to_standard


# ══════════════════════════════════════════════════════════════════════════════
# 主繪圖（支援多模型）
# ══════════════════════════════════════════════════════════════════════════════

def plot_one_track(track_id: str, samples: list, output_dir: str, model_name: str = "MODEL"):
    tracks = parse_tracks(samples)
    if not tracks:
        print(f"  [SKIP] {track_id}：無有效軌跡資料")
        return

    mean = ensemble_mean(tracks)

    # ── 判斷此次是否為誤差圈模式（EGTY_V4 或單成員） ────────────────────────
    is_cone_model = (model_name == "EGTY_V4" or len(tracks) == 1)
    # 誤差圈圖：全寬（wide=True），集合圖：保留右欄（wide=False）
    fig, ax, data_crs, _ = _build_map_figure(tracks, wide=is_cone_model)

    if is_cone_model:
        # 單成員：固定 ERROR_RADII 的誤差圈 + 加粗路徑；不顯示成員列表
        draw_forecast_cone(ax, tracks[0], data_crs)
        draw_track(ax, tracks[0]["lons"], tracks[0]["lats"], tracks[0]["winds"],
                lw=2.5, alpha=1.0, zorder=3, data_crs=data_crs)
                
        # ✨ 關鍵修復：單一成員算不出 mean，我們直接拿唯一的這條軌跡來畫時間標籤與起點 X
        draw_mean_track(ax, tracks[0], data_crs, show_labels=True)
        
    else:
        # 多成員：繪製所有成員路徑，並附上右欄成員列表
        for tr in tracks:
            draw_track(ax, tr["lons"], tr["lats"], tr["winds"],
                    lw=0.9, alpha=0.65, zorder=2, data_crs=data_crs)
        draw_member_list(fig, tracks, mean)
        
        # 多成員的平均路徑：關閉標籤保持畫面乾淨
        if mean:
            draw_mean_track(ax, mean, data_crs, show_labels=False)

    init_dt  = parse_dt(tracks[0]["times"][0])
    init_str = init_dt.strftime("%Y-%m-%dT%H")

    # 根據是否為誤差圈模式動態調整標題
    title_type = "Ensemble Cone" if is_cone_model else "Ensemble Forecast"

    ax.annotate(
        f"{model_name} {title_type} [{track_id}] Tropical Cyclone Track — {len(tracks)} Members\n"
        f"Maximum 1-minute Sustained Wind Speed and Minimum Central Pressure",
        xy=(0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )
    
    # 調整右上角文字的位置
    ax.annotate(
        f"Made By EGTY\n{init_str}:Forecast time",
        xy=(1.0 if is_cone_model else 1.2, 1.01),
        xycoords="axes fraction", textcoords="offset points",
        ha="right", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )

    safe_id  = track_id.replace("/", "_").replace("\\", "_")
    
    # ── 核心修改：判斷檔名後綴 ──
    out_filename = f"{init_str}_cone.png" if is_cone_model else f"{init_str}.png"
    out_path = os.path.join(output_dir, safe_id, model_name, out_filename)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)
    
    # 終端機輸出提示也動態調整
    file_type = "cone" if is_cone_model else "track"
    print(f"  ✅  [{model_name}] {track_id} {file_type}  →  {out_path}")

    # ── 多成員模型額外輸出以成員離散度為半徑的誤差圈圖 ──────────────────────
    if len(tracks) > 2:
        plot_member_spread_cone(track_id, tracks, mean, output_dir, model_name)

def plot_member_spread_cone(track_id: str, tracks: list, mean: dict,
                             output_dir: str, model_name: str):
    """
    多成員專用：額外輸出一張以成員空間離散度為半徑的誤差圈圖（全寬，無成員列表）。
    輸出檔名為 <init_str>_cone.png。
    """
    fig, ax, data_crs, _ = _build_map_figure(tracks, wide=True)

    # 誤差圈（以成員離散度為半徑）
    draw_member_spread_cone(ax, tracks, mean)

    # 集合平均路徑（加粗）
# ── 約在第 480 行附近的 plot_member_spread_cone 函式中 ──
    if mean:
        draw_track(ax, mean["lons"], mean["lats"], mean["winds"],
                   lw=2.5, alpha=1.0, zorder=3, data_crs=data_crs)
        # 保持 show_labels=True，在乾淨的誤差圈圖中標記時間與強度
        draw_mean_track(ax, mean, data_crs, show_labels=True)

    init_dt  = parse_dt(tracks[0]["times"][0])
    init_str = init_dt.strftime("%Y-%m-%dT%H")

    ax.annotate(
        f"{model_name} Ensemble Cone [{track_id}] — {len(tracks)} Members\n"
        f"Cone radius = member spread (fallback to fixed radii when < half members)",
        xy=(0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="left", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )
    ax.annotate(
        f"Made By EGTY\n{init_str}:Forecast time",
        xy=(1.0, 1.01), xycoords="axes fraction", textcoords="offset points",
        ha="right", va="bottom", fontsize=13, color="#444444", fontweight="bold",
    )

    safe_id  = track_id.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(output_dir, safe_id, model_name, f"{init_str}_cone.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"  ✅  [{model_name}] {track_id} cone →  {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 清理舊圖
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_old_tracks(output_dir: str):
    if not os.path.exists(output_dir):
        return

    now = datetime.utcnow()
    for track_id in os.listdir(output_dir):
        track_path = os.path.join(output_dir, track_id)
        if not os.path.isdir(track_path):
            continue

        for model_name in os.listdir(track_path):
            model_dir = os.path.join(track_path, model_name)
            if not os.path.isdir(model_dir):
                continue

            # 主圖與 _cone 圖分開管理
            for suffix, pattern in [("主圖", "*.png"), ]:
                # 取出所有 png（含 _cone）
                pass

            png_files = sorted(glob.glob(os.path.join(model_dir, "*.png")))
            if not png_files:
                continue

            # 規則 1：每個 stem 最多保留 4 版（主圖與 _cone 分開計算）
            base_files = sorted(f for f in png_files if not f.endswith("_cone.png"))
            cone_files = sorted(f for f in png_files if f.endswith("_cone.png"))

            for group in (base_files, cone_files):
                if len(group) > 4:
                    for f in group[:-4]:
                        try:
                            os.remove(f)
                            print(f"🗑️ [{model_name}] 刪除舊圖 (超過4張): {f}")
                        except Exception as e:
                            print(f"⚠️ 刪除失敗 {f}: {e}")

            # 規則 2：最新主圖超過 24 小時刪整個風暴資料夾
            remaining_base = sorted(f for f in glob.glob(os.path.join(model_dir, "*.png"))
                                    if not f.endswith("_cone.png"))
            if not remaining_base:
                continue
            newest_name = os.path.basename(remaining_base[-1]).replace(".png", "")
            try:
                newest_time = datetime.strptime(newest_name, "%Y-%m-%dT%H")
                if (now - newest_time).total_seconds() > 24 * 3600:
                    shutil.rmtree(track_path)
                    print(f"🧹 {track_id} 超過 24h 未更新，已刪除: {track_path}")
                    break
            except ValueError:
                print(f"⚠️ 無法從檔名解析時間: {newest_name}")


# ══════════════════════════════════════════════════════════════════════════════
# 單一模型的完整流程
# ══════════════════════════════════════════════════════════════════════════════

def run_model_pipeline(model_name: str, output_dir: str):
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'='*60}")
    print(f"  模型：{model_name}")
    print(f"{'='*60}")

    if not os.path.exists(cfg["json_path"]):
        print(f"❌ [{model_name}] 找不到 {cfg['json_path']}，跳過繪圖。")
        return

    data = load_json(cfg["json_path"])
    print(f"🌀 [{model_name}] 共 {len(data)} 個 track_id，開始繪圖…")
    for track_id, samples in data.items():
        plot_one_track(track_id, samples, output_dir, model_name=model_name)
    print(f"🎉 [{model_name}] 繪圖完成")
# ══════════════════════════════════════════════════════════════════════════════
# 相同風暴資料夾合併（繪圖完成後執行）
# ══════════════════════════════════════════════════════════════════════════════
def merge_same_storm_output_folders(output_dir: str, active_dir: str = "active_typhoon"):
    """
    掃描 active_dir 下的 .dat 檔案，找出已確認的 Invest→正式 升級過渡關係，
    再將對應的 output_dir PNG 資料夾合併（重疊檔名以 mtime 較新者為主）。
    只合併有明確過渡依據的配對，不會誤併不相關的擾動。
    """
    if not os.path.exists(active_dir) or not os.path.exists(output_dir):
        return

    # ── 1. 掃描 .dat 找出明確的過渡關係（與 merge_transitioned_typhoons 相同邏輯）──
    transitions: dict[str, str] = {}  # dest_id -> src_id

    for folder in os.listdir(active_dir):
        folder_path = os.path.join(active_dir, folder)
        if not os.path.isdir(folder_path) or folder == "model_tracks":
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".dat"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "rb") as f:
                    content = f.read()
                trans = find_transition_in_content(content)
                if trans:
                    src, dest = trans          # e.g. src=WP932026, dest=WP052026
                    transitions[dest] = src
                    print(f"\n🔗 偵測到氣旋升級過渡（PNG合併用）：{src} -> {dest}")

    if not transitions:
        print("  （無需合併的過渡關係）")
        return

    # ── 2. 依據過渡關係合併 output PNG 資料夾 ──────────────────────────────────
    for dest_id, src_id in transitions.items():
        dest_folder = os.path.join(output_dir, dest_id)
        src_folder  = os.path.join(output_dir, src_id)

        if not os.path.exists(dest_folder):
            print(f"  ⚠️  目標資料夾不存在，略過：{dest_folder}")
            continue
        if not os.path.exists(src_folder):
            print(f"  ⚠️  來源資料夾不存在，略過：{src_folder}")
            continue

        print(f"\n📂 合併 PNG：{src_id} → {dest_id}")
        _merge_track_folders(src_folder, dest_folder, src_id, dest_id)

        # 合併完成後刪除 Invest 的輸出資料夾
        try:
            shutil.rmtree(src_folder)
            print(f"  🧹 已刪除來源 PNG 資料夾：{src_folder}")
        except Exception as e:
            print(f"  ⚠️  無法刪除 {src_folder}：{e}")
def _merge_track_folders(src_folder: str, dest_folder: str,
                          src_id: str, dest_id: str):
    """
    將 src_folder/{model}/*.png 逐一合併至 dest_folder/{model}/。
    相同檔名時，以 mtime 較新的檔案為主（舊的直接覆蓋）。
    """
    if not os.path.isdir(src_folder):
        return

    for model_name in os.listdir(src_folder):
        src_model_dir  = os.path.join(src_folder,  model_name)
        dest_model_dir = os.path.join(dest_folder, model_name)

        if not os.path.isdir(src_model_dir):
            continue

        os.makedirs(dest_model_dir, exist_ok=True)
        png_files = glob.glob(os.path.join(src_model_dir, "*.png"))

        for src_file in png_files:
            fname     = os.path.basename(src_file)
            dest_file = os.path.join(dest_model_dir, fname)

            if os.path.exists(dest_file):
                src_mtime  = os.path.getmtime(src_file)
                dest_mtime = os.path.getmtime(dest_file)

                if src_mtime > dest_mtime:
                    # 來源較新 → 覆蓋目標
                    shutil.copy2(src_file, dest_file)
                    print(f"  ♻️  [{model_name}] {fname}：來源較新，已覆蓋")
                else:
                    # 目標已是最新 → 保留目標，略過
                    print(f"  ✔️  [{model_name}] {fname}：目標已是最新，略過")
            else:
                # 目標不存在 → 直接複製
                shutil.copy2(src_file, dest_file)
                print(f"  📂  [{model_name}] {fname}：已複製至 {dest_id}")

def find_transition_in_content(content_bytes: bytes):
    """從 .dat 內容中尋找升級過渡（TRANSITIONED）資訊。"""
    try:
        text = content_bytes.decode('utf-8')
        match = re.search(r'TRANSITIONED\s*,\s*([a-zA-Z0-9]+)\s+to\s+([a-zA-Z0-9]+)', text, re.I)
        if match:
            src, dest = match.groups()
            return standardise_typhoon_id(src), standardise_typhoon_id(dest)
    except Exception:
        pass
    return None

def standardise_typhoon_id(raw_id: str) -> str:
    """
    將 ATCF 的臨時編號（如 wpB32026, bwp932026）轉為標準的 WP932026 格式。
    JTWC Invest 編碼規則：A0-A9 代表 80-89，B0-B9 代表 90-99。
    """
    s = raw_id.strip().upper()
    if s.startswith('BWP'):
        s = s[1:]
    
    match = re.match(r'^([A-Z]{2})([0-9A-Z]{2})([0-9]{4})$', s)
    if match:
        basin, num_code, year = match.groups()
        if num_code.startswith('B') and num_code[1].isdigit():
            num_code = f"9{num_code[1]}"
        elif num_code.startswith('A') and num_code[1].isdigit():
            num_code = f"8{num_code[1]}"
        return f"{basin}{num_code}{year}"
    return s
# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "active_typhoon")
    os.makedirs(output_dir, exist_ok=True)

    for model_name in MODEL_CONFIGS:
        run_model_pipeline(model_name, output_dir)
    print(f"\n{'='*60}")
    print("🔗 開始合併相同風暴 PNG 資料夾…")
    merge_same_storm_output_folders(output_dir, active_dir="active_typhoon")  # ← 修改後
    print("✅ 合併完成！")

    # 3. 清理過期圖片
    print(f"\n{'='*60}")
    print("🧹 開始自動清理…")
    cleanup_old_tracks(output_dir)
    print("✅ 清理完成！")

if __name__ == "__main__":
    main()
