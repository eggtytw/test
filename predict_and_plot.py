import os
import numpy as np
import matplotlib.colors as mcolors
import json
import math
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import requests
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================================================================
# 1. 設定
# =============================================================================

MODEL_URL        = "https://huggingface.co/EGTY/Weather_model_V4.3/resolve/main/Weather_model_V4.3.pth"
MODEL_PATH       = "Weather_model_V4.3.pth"
TEST_NPZ_PATH    = "V4_model_data.npz"
STATIC_PATHS     = [
    "environment-2down/land_mask.npy",
    "environment-2down/soil_type.npy",
    "environment-2down/topography.npy",
]

VAR_NAMES = [
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

SEQ_LEN              = 2
BASE_CH              = 192
GROUP_NORM_GROUPS    = 8
AUTOREGRESSIVE_STEPS = 30
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEARCH_RADIUS_KM    = 150
VORTEX_VALIDATION_RADIUS_KM = 200
PATIENCE_RADIUS_INCREASE_KM = 50
PATIENCE_WINDOW_HOURS       = 24

VIS_VARS = ['t2m','msl','q_500','wind_speed']

REGIONS = {
    'global':        {'name': 'Global',       'extent': None},
    'west_pacific':  {'name': 'West Pacific',  'extent': [90,  180,   0,  45]},
    'south_pacific': {'name': 'South Pacific', 'extent': [90,  180, -45,   0]},
    'east_pacific':  {'name': 'East Pacific',  'extent': [-180, -90,   0,  45]},
    'east_atlantic': {'name': 'East Atlantic', 'extent': [-90,    0,   0,  45]},
}

# 色條配置
q_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", ["#613A00", '#FFFFFF', "#008F92", "#004792"])
wind_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", ["#FFFFFF", "#00C4B3", "#03C400", "#C4C400",
                         "#C45800", "#C40000", "#C40093"])

VAR_PLOT = {
    't2m': {
        'long_name': '2m Temperature',
        'unit':      '°C',
        'cmap':      'RdYlBu_r',
        'symmetric': False,
        'contour':   True,
        'n_levels':  15,
        'vmax':      40.0,
        'vmin':      -60.0,
    },
    'msl': {
        'long_name': 'Mean Sea Level Pressure',
        'unit':      'hPa',
        'cmap':      'Spectral_r',
        'symmetric': False,
        'contour':   True,
        'n_levels':  20,
        'vmax':      1072.0,
        'vmin':      952.0,
    },
    'q_500': {
        'long_name': '500 hPa Specific Humidity with Mean Sea Level Pressure Isobars',
        'unit':      'g/kg',
        'cmap':      q_cmap,
        'symmetric': False,
        'contour':   True,
        'n_levels':  -0.5,
        'vmax':      5,
        'vmin':      0,
    },
    'wind_speed': {
        'long_name': '10m Wind Speed with Mean Sea Level Pressure Isobars',
        'unit':      'm/s',
        'cmap':      wind_cmap,
        'symmetric': False,
        'contour':   True,
        'n_levels':  0,
        'vmax':      30.0,
        'vmin':      0,
    },
}

GIF          = True
GIF_DURATION = 0.5
SEED         = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# 2. 模型定義（與訓練版本保持完全一致）
# =============================================================================

def _custom_polar_padding(x, pad_y, vector_indices=None):
    if pad_y == 0:
        return x
    north_src = x[:, :, :, :pad_y, :]
    south_src = x[:, :, :, -pad_y:, :]
    if vector_indices:
        north_src = north_src.clone(); north_src[:, vector_indices] *= -1.0
        south_src = south_src.clone(); south_src[:, vector_indices] *= -1.0
    W = x.shape[-1]
    north = torch.flip(torch.roll(north_src, shifts=W // 2, dims=-1), dims=[-2])
    south = torch.flip(torch.roll(south_src, shifts=W // 2, dims=-1), dims=[-2])
    return torch.cat([north, x, south], dim=-2)


class CircularConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, vector_indices=None):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.kernel_size    = ks
        self.vector_indices = vector_indices
        self.conv = nn.Conv3d(in_channels, out_channels, ks, stride, padding=0, bias=bias)

    def forward(self, x):
        pd, py, px = [(k - 1) // 2 for k in self.kernel_size]
        x = _custom_polar_padding(x, py, self.vector_indices)
        if px > 0:
            x = torch.cat([x[..., -px:], x, x[..., :px]], dim=-1)
        if pd > 0:
            x = F.pad(x, (0, 0, 0, 0, pd, pd), 'reflect')
        return self.conv(x)


class CircularConvBlock3D_Optimized(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, is_first_layer=False,
                 vector_indices=None):
        super().__init__()
        fk = (2, 3, 3) if is_first_layer else (1, 3, 3)
        self.net = nn.Sequential(
            CircularConv3d(in_ch,  out_ch, fk,      vector_indices=vector_indices),
            nn.GroupNorm(groups, out_ch), nn.ReLU(True),
            CircularConv3d(out_ch, out_ch, (1,3,3), vector_indices=None),
            nn.GroupNorm(groups, out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class CircularUNet3D(nn.Module):
    def __init__(self, in_ch, base_ch, out_ch, groups, vector_channel_indices=None):
        super().__init__()
        B = base_ch
        self.enc1  = CircularConvBlock3D_Optimized(in_ch, B,    groups, is_first_layer=True,
                                                   vector_indices=vector_channel_indices)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2  = CircularConvBlock3D_Optimized(B,   B*2,   groups)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.enc3  = CircularConvBlock3D_Optimized(B*2, B*4,   groups)
        self.pool3 = nn.MaxPool3d((1, 2, 2))
        self.enc4  = CircularConvBlock3D_Optimized(B*4, B*8,   groups)
        self.b     = CircularConvBlock3D_Optimized(B*8, B*8,   groups)
        self.u3    = nn.ConvTranspose3d(B*8, B*4, (1,2,2), stride=(1,2,2))
        self.dec3  = CircularConvBlock3D_Optimized(B*8, B*4,   groups)
        self.u2    = nn.ConvTranspose3d(B*4, B*2, (1,2,2), stride=(1,2,2))
        self.dec2  = CircularConvBlock3D_Optimized(B*4, B*2,   groups)
        self.u1    = nn.ConvTranspose3d(B*2, B,   (1,2,2), stride=(1,2,2))
        self.dec1  = CircularConvBlock3D_Optimized(B*2, B,     groups)
        self.final = nn.Conv3d(B, out_ch, 1)

    def forward(self, x):
        c1 = self.enc1(x);  p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        b  = self.b(c4)
        d3 = self.dec3(torch.cat([c3, self.u3(b)],  dim=1))
        d2 = self.dec2(torch.cat([c2, self.u2(d3)], dim=1))
        d1 = self.dec1(torch.cat([c1, self.u1(d2)], dim=1))
        return self.final(d1).squeeze(2)

# =============================================================================
# 3. 輔助函式
# =============================================================================

def load_static(paths):
    tensors = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[警告] 靜態檔案不存在: {p}")
            continue
        arr = np.load(p).astype(np.float32)
        t   = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        tensors.append(t)
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def destandardize(pred_std, means, stds, var_names=None):
    """反標準化 (T, C, H, W)，並做單位轉換。"""
    m    = np.array(means, dtype=np.float32).reshape(1, -1, 1, 1)
    s    = np.array(stds,  dtype=np.float32).reshape(1, -1, 1, 1)
    data = pred_std * s + m
    if var_names is not None:
        for c, vname in enumerate(var_names):
            if vname == 't2m':
                data[:, c] -= 273.15       # K → °C
            elif vname == 'msl':
                data[:, c] /= 100.0        # Pa → hPa
            elif vname == 'q_500':
                data[:, c] *= 1000         # kg kg⁻¹ → g kg⁻¹
    return data


@torch.no_grad()
def autoregressive_predict(model, init_seq_std, static_tensor, steps, device):
    model.eval()
    T, C_w, H, W = init_seq_std.shape
    seq_np = init_seq_std.copy()
    preds  = []

    static_gpu = static_tensor.to(device) if static_tensor is not None else None
    if static_gpu is not None:
        static_exp = static_gpu.unsqueeze(0).expand(T, -1, -1, -1)

    for _ in tqdm(range(steps), desc="自迴歸預測"):
        wx = torch.from_numpy(seq_np).float().to(device)
        x  = torch.cat([wx, static_exp], dim=1) if static_gpu is not None else wx
        x  = x.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)

        out_np     = out.squeeze(0).cpu().numpy()
        preds.append(out_np)
        seq_np     = np.roll(seq_np, -1, axis=0)
        seq_np[-1] = out_np

    return np.stack(preds, axis=0)

# =============================================================================
# 4. 颱風追蹤專用的地理計算函式
# =============================================================================

def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.asin(math.sqrt(a))


def coords_to_grid_indices(lat, lon, H, W):
    lon_0_360 = lon if lon >= 0 else lon + 360
    i = int(round((90.0 - lat) / (180.0 / (H - 1))))
    j = int(round(lon_0_360 / (360.0 / W))) % W
    return min(max(i, 0), H - 1), j


def grid_indices_to_coords(i, j, H, W):
    lat       = 90.0 - i * (180.0 / (H - 1))
    lon_0_360 = j * (360.0 / W)
    lon       = lon_0_360 if lon_0_360 <= 180 else lon_0_360 - 360
    return lat, lon


def validate_cyclonic_circulation(field_data, center_lat, center_lon, var_indices):
    _, H, W    = field_data.shape
    u_idx, v_idx = var_indices['u10'], var_indices['v10']
    u_winds, v_winds, max_wind_speed = [], [], 0.0
    for i in range(H):
        for j in range(W):
            lat, lon = grid_indices_to_coords(i, j, H, W)
            if haversine(center_lon, center_lat, lon, lat) <= VORTEX_VALIDATION_RADIUS_KM:
                u, v = field_data[u_idx, i, j], field_data[v_idx, i, j]
                u_winds.append(u); v_winds.append(v)
                max_wind_speed = max(max_wind_speed, math.sqrt(u**2 + v**2))
    if not u_winds:
        return False
    u_range = max(u_winds) - min(u_winds)
    v_range = max(v_winds) - min(v_winds)
    return max_wind_speed > 6 and u_range > 5 and v_range > 5


def get_max_wind_speed(field_data, center_lat, center_lon, var_indices):
    """100 km 以內的最大 10m 風速 (m/s)。"""
    _, H, W            = field_data.shape
    max_wind_speed     = 0.0
    for i in range(H):
        for j in range(W):
            lat, lon = grid_indices_to_coords(i, j, H, W)
            if haversine(center_lon, center_lat, lon, lat) <= 100:
                u     = field_data[var_indices['u10'], i, j]
                v     = field_data[var_indices['v10'], i, j]
                speed = math.sqrt(u**2 + v**2)
                if speed > max_wind_speed:
                    max_wind_speed = speed
    return max_wind_speed


def find_pressure_minimum(field_data, start_lat, start_lon, search_radius, msl_idx):
    _, H, W      = field_data.shape
    min_pressure = float('inf')
    best_ij      = None
    for i in range(H):
        for j in range(W):
            lat, lon = grid_indices_to_coords(i, j, H, W)
            if haversine(start_lon, start_lat, lon, lat) <= search_radius:
                pressure = field_data[msl_idx, i, j]
                if pressure < min_pressure:
                    min_pressure = pressure
                    best_ij      = (i, j)
    return best_ij, min_pressure


def get_active_typhoon_initials(base_dir="active_typhoon"):
    typhoons = []
    if not os.path.exists(base_dir):
        print(f"[警告] 找不到資料夾: {base_dir}")
        return typhoons

    for file_path in glob.glob(os.path.join(base_dir, "*", "*.dat")):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue
            
            # 取得最後一行（最新觀測點）
            parts = [p.strip() for p in lines[-1].strip().split(',')]

            # 解析經緯度
            lat = float(parts[6][:-1]) / 10.0
            if parts[6].endswith('S'): lat = -lat
            lon = float(parts[7][:-1]) / 10.0
            if parts[7].endswith('W'): lon = -lon
            if lon > 180: lon -= 360

            # --- 新增：解析氣壓與風速 ---
            # ATCF 格式中：parts[8] 是最大持續風速 (knots), parts[9] 是中心最低氣壓 (hPa)
            wind_knots = float(parts[8]) if parts[8] else 0.0
            mslp_hpa = float(parts[9]) if parts[9] else 1013.0

            ty_name = parts[27] if len(parts) > 27 else "Unknown"
            ty_id   = f"{parts[0]}{parts[1]}{parts[2][:4]}"

            typhoons.append({
                "id": ty_id, 
                "name": ty_name,
                "latitude": lat, 
                "longitude": lon,
                "mslp": mslp_hpa,
                "wind": wind_knots
            })
            print(f"找到活躍颱風: {ty_name} ({ty_id}) | Lat:{lat}, Lon:{lon} | P:{mslp_hpa}hPa, V:{wind_knots}kt")
        except Exception as e:
            print(f"[錯誤] 無法讀取檔案 {file_path}: {e}")

    return typhoons

def convert_numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def track_typhoon(preds_phys, var_names, initial_track, typhoon_number,
                  member_id, base_time):
    var_indices = {name: i for i, name in enumerate(var_names)}
    steps, _, H, W = preds_phys.shape
    full_track     = list(initial_track)

    print(f"成員 {member_id} | 颱風 {typhoon_number} | 開始追蹤 (起始時間: {base_time})...")

    in_patience_mode    = False
    patience_search_radius = 0

    for t in range(steps):
        current_hour   = (t + 1) * 6
        valid_time_obj = base_time + timedelta(hours=current_hour)

        prev_lat = full_track[-1]['coordinates']['lat']
        prev_lon = full_track[-1]['coordinates']['lon']

        if in_patience_mode:
            current_search_radius  = patience_search_radius
            print(f"  [t={current_hour}h] 耐心模式啟用，半徑: {current_search_radius:.0f} km")
            patience_search_radius += PATIENCE_RADIUS_INCREASE_KM
        else:
            if len(full_track) < 2:
                current_search_radius = DEFAULT_SEARCH_RADIUS_KM
            else:
                l1 = full_track[-2]['coordinates']
                l2 = full_track[-1]['coordinates']
                last_move_dist        = haversine(l1['lon'], l1['lat'],
                                                  l2['lon'], l2['lat'])
                current_search_radius = max(100, last_move_dist * 1.5)

        best_ij, min_pressure = find_pressure_minimum(
            preds_phys[t], prev_lat, prev_lon,
            current_search_radius, var_indices['msl'])

        center_found_and_valid = False
        if best_ij is not None and min_pressure < 1010:
            potential_center_lat, potential_center_lon = grid_indices_to_coords(
                best_ij[0], best_ij[1], H, W)
            if validate_cyclonic_circulation(
                    preds_phys[t], potential_center_lat,
                    potential_center_lon, var_indices):
                center_found_and_valid = True

        if center_found_and_valid:
            if in_patience_mode:
                print(f"  [t={current_hour}h] 已重新鎖定中心，退出耐心模式。")
                in_patience_mode = False

            max_wind = get_max_wind_speed(
                preds_phys[t], potential_center_lat,
                potential_center_lon, var_indices)

            full_track.append({
                "valid_time":  valid_time_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "coordinates": {"lat": float(potential_center_lat),
                                 "lon": float(potential_center_lon)},
                "intensity":   {"mslp_hpa":       float(min_pressure),
                                 "max_wind_knots": float(max_wind * 1.94384)},
            })
        else:
            if current_hour <= PATIENCE_WINDOW_HOURS:
                if not in_patience_mode:
                    print(f"  [t={current_hour}h] 追蹤失敗，啟動耐心模式。")
                    in_patience_mode       = True
                    patience_search_radius = current_search_radius + PATIENCE_RADIUS_INCREASE_KM
                continue
            else:
                print(f"  [t={current_hour}h] 超過耐心窗口，追蹤中止。")
                break

    return full_track[len(initial_track):]

# =============================================================================
# 5. 淺色主題 & 繪圖核心
# =============================================================================

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'text.color':         '#111111',
    'axes.facecolor':     '#F5F8FA',
    'figure.facecolor':   '#FFFFFF',
    'axes.edgecolor':     '#D8DDE3',
    'xtick.color':        '#555555',
    'ytick.color':        '#555555',
    'figure.dpi':         100,
    'savefig.dpi':        100,
    'savefig.facecolor':  '#FFFFFF',
    'savefig.bbox':       None,
    'savefig.pad_inches': 0.0,
})

_OCEAN_L   = cfeature.NaturalEarthFeature(
    'physical', 'ocean', '50m', facecolor='#E8F0E4', edgecolor='none')
_LAND_L    = cfeature.NaturalEarthFeature(
    'physical', 'land',  '50m', facecolor='#E8F0E4', edgecolor='none')
_COAST_L   = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '50m',
    facecolor='none', edgecolor="#00830B", linewidth=0.75)
_BORDERS_L = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '50m',
    facecolor='none', edgecolor='#9BAAB5', linewidth=0.3, linestyle='--')


def _data_extent(lons, lats):
    dlon = abs(lons[1] - lons[0])
    dlat = abs(lats[0] - lats[1])
    return [lons[0]  - dlon/2, lons[-1] + dlon/2,
            lats[-1] - dlat/2, lats[0]  + dlat/2]


def _region_stats(data_2d, lons, lats, region_ext):
    if region_ext is None:
        arr = data_2d
    else:
        lon_min, lon_max, lat_min, lat_max = region_ext
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        if lon_mask.any() and lat_mask.any():
            arr = data_2d[np.ix_(lat_mask, lon_mask)]
        else:
            arr = data_2d
    return float(np.nanmax(arr)), float(np.nanmin(arr))


def _overlay_msl_isobars(ax, lons, lats, msl_data, region_ext, proj):
    lons_m, lats_m = np.meshgrid(lons, lats)

    if region_ext is not None:
        lon_min, lon_max, lat_min, lat_max = region_ext
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        sub = (msl_data[np.ix_(lat_mask, lon_mask)]
               if lon_mask.any() and lat_mask.any() else msl_data)
    else:
        sub = msl_data

    p_min  = np.floor(np.nanmin(sub) / 4) * 4
    p_max  = np.ceil(np.nanmax(sub)  / 4) * 4
    levels = np.arange(p_min, p_max + 4, 4)

    try:
        cs   = ax.contour(lons_m, lats_m, msl_data, levels=levels,
                          colors='#1A1A2E', linewidths=1.0, alpha=0.80,
                          transform=proj, zorder=8)
        lbls = ax.clabel(cs, inline=True, fontsize=6.0,
                         fmt='%.0f', use_clabeltext=True)
        for lbl in lbls:
            lbl.set_color('#1A1A2E')
            lbl.set_fontsize(6.0)
            lbl.set_fontweight('bold')
            lbl.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white')])
    except Exception as e:
        print(f"  [警告] 等壓線繪製失敗: {e}")


def _overlay_wind_barbs(ax, lons, lats, u_data, v_data, region_ext, proj):
    lons_m, lats_m = np.meshgrid(lons, lats)
    lon_idx = np.arange(0, len(lons), 5)
    lat_idx = np.arange(0, len(lats), 5)
    lo_s    = lons_m[np.ix_(lat_idx, lon_idx)]
    la_s    = lats_m[np.ix_(lat_idx, lon_idx)]
    u_s     = u_data[np.ix_(lat_idx, lon_idx)]
    v_s     = v_data[np.ix_(lat_idx, lon_idx)]
    spd_s   = np.sqrt(u_s**2 + v_s**2)

    color_map = [
        (spd_s < 5,                          "#3D6B8D", 0.65),
        ((spd_s >= 5) & (spd_s < 15),        "#10933C", 0.75),
        (spd_s >= 15,                         "#870000", 0.90),
    ]
    barb_kw = dict(
        transform=proj, zorder=9, length=5, linewidth=0.7, pivot='middle',
        sizes=dict(emptybarb=0.18, spacing=0.13, height=0.35),
        barb_increments=dict(half=2.5, full=5, flag=25),
    )
    for mask, color, alpha in color_map:
        if mask.any():
            ax.barbs(lo_s[mask], la_s[mask], u_s[mask], v_s[mask],
                     color=color, alpha=alpha, **barb_kw)


def plot_frame(data_2d, lons, lats, vmin, vmax, cfg, region_name,
               step_label, out_path,
               wind_u=None, wind_v=None, msl_data=None):
    extent_data = _data_extent(lons, lats)
    region_ext  = REGIONS[region_name]['extent']
    region_disp = REGIONS[region_name]['name']
    unit        = cfg['unit']
    data_max, data_min = _region_stats(data_2d, lons, lats, region_ext)

    fig = plt.figure(figsize=(16, 9), facecolor='#FFFFFF', dpi=100)

    fig.text(0.030, 0.985, f"{cfg['long_name']}  ·  {region_disp}",
             fontsize=14, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.030, 0.96, step_label,
             fontsize=14, fontweight='bold', color='#555555', va='top', ha='left')
    fig.text(0.98, 0.985, "Made By EGTY // Model: EGTY V4",
             fontsize=14, fontweight='bold', color='#111111', va='top', ha='right')
    fig.text(0.98, 0.96, f"Max: {data_max:.2f}  Min: {data_min:.2f}  [{unit}]",
             fontsize=14, fontweight='bold', color='#555555', va='top', ha='right')

    fig.add_artist(matplotlib.lines.Line2D(
        [0.03, 0.98], [0.932, 0.932], transform=fig.transFigure,
        color="black", lw=1.5, zorder=10))

    proj   = ccrs.PlateCarree()
    ax_map = fig.add_axes([0.03, 0.05, 0.95, 0.9], projection=proj)
    ax_map.set_facecolor('#E8F0E4')
    for sp in ax_map.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(2)

    ax_map.add_feature(_OCEAN_L, zorder=0)
    ax_map.add_feature(_LAND_L,  zorder=1)

    im = ax_map.imshow(
        data_2d, extent=extent_data, transform=proj, origin='upper',
        cmap=plt.get_cmap(cfg['cmap'], 60), vmin=vmin, vmax=vmax,
        interpolation='bilinear', zorder=3, alpha=0.82)

    if cfg['contour'] and cfg['n_levels'] > 0:
        lons_m, lats_m = np.meshgrid(lons, lats)
        levels = np.linspace(vmin, vmax, cfg['n_levels'] + 1)
        try:
            cs   = ax_map.contour(lons_m, lats_m, data_2d, levels=levels,
                                  colors='#1A1A2E', linewidths=0.5, alpha=0.55,
                                  transform=proj, zorder=7)
            lbls = ax_map.clabel(cs, inline=True, fontsize=5.5,
                                 fmt='%.0f', use_clabeltext=True)
            for lbl in lbls:
                lbl.set_color('#222222'); lbl.set_fontsize(5.5)
                lbl.set_path_effects([pe.withStroke(linewidth=1.8, foreground='white')])
        except Exception:
            pass

    ax_map.add_feature(_COAST_L,   zorder=6)
    ax_map.add_feature(_BORDERS_L, zorder=5)

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.4, color='#C5CDD5',
                          alpha=0.9, linestyle=':', zorder=4,
                          x_inline=False, y_inline=False)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': '#555555'}
    gl.ylabel_style = {'size': 8, 'color': '#555555'}
    gl.xlocator = mticker.MultipleLocator(30)
    gl.ylocator = mticker.MultipleLocator(20)

    if region_ext is not None:
        ax_map.set_extent(region_ext, crs=proj)

    if msl_data is not None:
        _overlay_msl_isobars(ax_map, lons, lats, msl_data, region_ext, proj)
    if wind_u is not None and wind_v is not None:
        _overlay_wind_barbs(ax_map, lons, lats, wind_u, wind_v, region_ext, proj)

    n_ticks = min(cfg['n_levels'] + 1, 16) if cfg['n_levels'] > 0 else 8
    ticks   = np.linspace(vmin, vmax, n_ticks)
    ax_cb   = fig.add_axes([0.03, 0.03, 0.95, 0.025])
    ax_cb.set_facecolor('#FFFFFF')
    cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal', ticks=ticks)
    cb.outline.set_linewidth(2); cb.outline.set_edgecolor("black")
    cb.ax.tick_params(labelsize=7.5, length=3, width=0.6,
                      color='#555555', labelcolor='#555555', direction='out')
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    cb.ax.set_title(f"[{unit}]", fontsize=7.5, color='#555555', pad=6, loc='right')

    plt.savefig(out_path, dpi=100, bbox_inches=None, facecolor='#FFFFFF')
    plt.close(fig)

# =============================================================================
# 6. 批次繪圖 + GIF
# =============================================================================

def build_all_labels(init_time_str, init_seq_len, pred_steps, step_hours=6):
    try:
        t0 = datetime.strptime(init_time_str[:13], "%Y-%m-%dT%H")
    except ValueError:
        t0 = datetime.strptime(init_time_str[:10], "%Y-%m-%d")

    labels = []
    for i in range(init_seq_len - 1, -1, -1):
        t      = t0 - timedelta(hours=i * step_hours)
        offset = -(i * step_hours)
        labels.append(f"Initial: {t.strftime('%Y-%m-%d  %H:00 UTC')}   ({offset:+d}h)")
    for i in range(1, pred_steps + 1):
        t = t0 + timedelta(hours=i * step_hours)
        labels.append(f"Forecast: {t.strftime('%Y-%m-%d  %H:00 UTC')}   (+{i*step_hours}h)")
    return labels


def compute_wind_speed(data_phys):
    """(T, C, H, W) → (T, H, W)  10m 風速"""
    u = data_phys[:, VAR_NAMES.index('u10'), :, :]
    v = data_phys[:, VAR_NAMES.index('v10'), :, :]
    return np.sqrt(u * u + v * v)


def visualize_variable(data_phys_all, var_name, lons, lats, step_labels, out_dir):
    cfg = VAR_PLOT.get(var_name)
    if cfg is None:
        print(f"[跳過] {var_name} 沒有繪圖設定。")
        return

    frames_arr = (compute_wind_speed(data_phys_all)
                  if var_name == 'wind_speed'
                  else data_phys_all[:, VAR_NAMES.index(var_name), :, :])

    vmin  = cfg['vmin']
    vmax  = cfg['vmax']
    total = len(step_labels)
    print(f"\n[{var_name}]  vmin={vmin:.2f}  vmax={vmax:.2f}  frames={total}")

    need_msl   = var_name in ('wind_speed', 'q_500')
    need_barbs = var_name == 'wind_speed'
    msl_all    = data_phys_all[:, VAR_NAMES.index('msl'), :, :] if need_msl   else None
    u10_all    = data_phys_all[:, VAR_NAMES.index('u10'), :, :] if need_barbs else None
    v10_all    = data_phys_all[:, VAR_NAMES.index('v10'), :, :] if need_barbs else None

    for reg_key in REGIONS:
        reg_dir     = os.path.join(out_dir, var_name, reg_key)
        os.makedirs(reg_dir, exist_ok=True)
        frame_paths = []

        for step_i, (data_2d, label) in enumerate(zip(frames_arr, step_labels)):
            out_path = os.path.join(reg_dir, f"frame_{(6*step_i)-6:03d}.png")
            plot_frame(
                data_2d     = data_2d,
                lons        = lons,
                lats        = lats,
                vmin        = vmin,
                vmax        = vmax,
                cfg         = cfg,
                region_name = reg_key,
                step_label  = label,
                out_path    = out_path,
                wind_u      = u10_all[step_i] if u10_all is not None else None,
                wind_v      = v10_all[step_i] if v10_all is not None else None,
                msl_data    = msl_all[step_i] if msl_all is not None else None,
            )
            frame_paths.append(out_path)

        if GIF:
            gif_path = os.path.join(reg_dir, "animation.gif")
            try:
                import imageio.v2 as imageio
                frames = [imageio.imread(p) for p in frame_paths]
                imageio.mimsave(gif_path, frames, duration=GIF_DURATION, loop=0)
                print(f"  [{var_name}/{reg_key}] GIF → {gif_path}")
            except ImportError:
                print("  [警告] imageio 未安裝，跳過 GIF。  pip install imageio")
            except Exception as e:
                print(f"  [警告] GIF 製作失敗: {e}")

# =============================================================================
# 7. 主流程
# =============================================================================

def main():
    static_tensor = load_static(STATIC_PATHS)

    if not os.path.exists(TEST_NPZ_PATH):
        raise FileNotFoundError(f"找不到輸入資料: {TEST_NPZ_PATH}")
    npz      = np.load(TEST_NPZ_PATH, allow_pickle=True)
    data_std = npz['data']
    times    = npz['times']

    OUT_DIR  = f"./forecast_{times[-1].replace(':', '').replace(' ', '_')}/V4_model"
    tag      = f"forecast_{times[-1].replace(':', '').replace(' ', '_')}"
    existing = sorted(
        [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("forecast_")],
        reverse=True)
    if tag not in existing:
        existing.insert(0, tag)

    while len(existing) > 4:
        oldest = existing.pop()
        try:
            shutil.rmtree(oldest)
            print(f"清理空間：已刪除最舊的資料夾: {oldest}")
        except Exception as e:
            print(f"刪除資料夾 {oldest} 時發生錯誤: {e}")

    forecast_data = [{"folder": d.replace("forecast_", "")} for d in existing]
    with open("forecast_list.json", "w", encoding="utf-8") as f:
        json.dump({"EGTY_V4": forecast_data}, f, indent=4, ensure_ascii=False)
    print(f"清單已更新：已記錄 {len(forecast_data)} 個預報時段。")

    if os.path.exists(OUT_DIR):
        print(f"偵測到目錄 {OUT_DIR} 已存在，略過本次運算。")
        return False

    T, C, H, W = data_std.shape
    lats = np.linspace(90, -90 + 0.5, H)
    lons = np.linspace(-180, 180 - 0.5, W)
    print(f"資料時間: {times}")

    if len(data_std) < SEQ_LEN:
        raise ValueError(f"資料長度 {len(data_std)} < SEQ_LEN {SEQ_LEN}")
    init_seq = data_std[:SEQ_LEN]

    if not os.path.exists(MODEL_PATH):
        print(f"下載模型從 {MODEL_URL} ...")
        resp = requests.get(MODEL_URL)
        resp.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(resp.content)
        print(f"模型已下載至 {MODEL_PATH}")

    n_weather = len(VAR_NAMES)
    n_static  = static_tensor.shape[0] if static_tensor is not None else 0
    model = CircularUNet3D(n_weather + n_static, BASE_CH,
                           n_weather, GROUP_NORM_GROUPS).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"模型已載入: {MODEL_PATH}")

    preds_std = autoregressive_predict(
        model, init_seq, static_tensor, AUTOREGRESSIVE_STEPS, DEVICE)

    means = [FIXED_STATS[v]['mean'] for v in VAR_NAMES]
    stds  = [FIXED_STATS[v]['std']  for v in VAR_NAMES]

    init_phys  = destandardize(init_seq,  means, stds, VAR_NAMES)
    preds_phys = destandardize(preds_std, means, stds, VAR_NAMES)

    # ── 颱風追蹤 ──────────────────────────────────────────────────────────
    active_list  = get_active_typhoon_initials("active_typhoon")
    last_time_str = str(times[-1])
    base_time    = datetime.strptime(last_time_str, "%Y-%m-%dT%H")

    if not active_list:
        print("[資訊] 未發現需要追蹤的活躍颱風。")
    else:
        all_typhoon_tracks = {}
        for ty in active_list:
            # 1. 建立初始點 (t=0)，建議直接使用從 .dat 讀取的真實數值
            init_point = {
                "valid_time":  base_time.strftime("%Y-%m-%d %H:%M:%S"),
                "coordinates": {
                    "lat": float(ty['latitude']), 
                    "lon": float(ty['longitude'])
                },
                "intensity": {
                    "mslp_hpa": float(ty['mslp']),   
                    "max_wind_knots": float(ty['wind'])
                },
            }

            predicted_points = track_typhoon(
                preds_phys, 
                VAR_NAMES, 
                [init_point],
                ty['id'], 
                "V4_Main", 
                base_time
            )

            full_track = [init_point] + predicted_points
            
            # 4. 存入最終結構
            all_typhoon_tracks[ty['id']] = [{
                "sample_id": 0.0,
                "data_points": full_track
            }]

        # 儲存 JSON (保持您原本的 convert_numpy_to_python 邏輯)
        output_json = os.path.join("active_typhoon", "cyclone_data_egty4.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_to_python(all_typhoon_tracks),
                      f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ 所有颱風路徑（含初始點）已儲存至: {output_json}")

    # ── 可視化 ────────────────────────────────────────────────────────────
    all_phys    = np.concatenate([init_phys, preds_phys], axis=0)
    all_phys    = np.roll(all_phys, W // 2, axis=-1)
    step_labels = build_all_labels(str(times[-1]), SEQ_LEN, AUTOREGRESSIVE_STEPS)

    os.makedirs(OUT_DIR, exist_ok=True)
    for vname in VIS_VARS:
        visualize_variable(all_phys, vname, lons, lats, step_labels, OUT_DIR)

    print(f"\n✅ 所有結果已儲存至: {OUT_DIR}")


if __name__ == "__main__":
    main()
