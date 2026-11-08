#!/usr/bin/env python3
# NOTE: device-specific readers in this file are withheld (see CODE_AVAILABILITY.md).
"""ODL R8 stereo+IMU capture analysis (runs on a short-lived AWS EC2 instance).

Input : <root>/<YYYY-MM-DD>/<device>/{records,calibration,logs}
Output: <out>/report.md, summary.json, sessions.csv, segments.csv, samples.csv,
        imu_activity/<session>.csv.gz, sei/<segment>_<eye>.csv.gz
Numbers only - no images leave the server.
"""
import argparse, csv, glob, gzip, json, os, re, statistics, struct, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

try:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FF = 'ffmpeg'

FACE_MODEL = os.environ.get('ODL_FACE_MODEL', '/data/yunet.onnx')
IMG_STEP = 300      # image metrics every 300 frames (10 s @30fps)
FACE_STEP = 900     # face count every 900 frames (30 s)
CAND_FRAME = 1500   # photo candidate per segment at 50 s (faces blurred on server)

# ---------------------------------------------------------------- helpers
def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None

def r(x, n=3):
    return None if x is None else round(float(x), n)

# ---------------------------------------------------------------- record .log
KV = re.compile(r'(\w+)=([^\s]+)')
def parse_log(path):
    d = {'file': os.path.basename(path)}
    for line in open(path, errors='ignore'):
        line = line.strip()
        ts = line[:19]
        if 'RECORD START' in line:
            d['start'] = ts; d.update({'start_' + k: v for k, v in KV.findall(line)})
        elif 'RECORD STOP' in line:
            d['stop'] = ts; d.update({'stop_' + k: v for k, v in KV.findall(line)})
        elif 'STROBE' in line:
            m = re.findall(r's(\d) irq=(\d+) pulse=(\d+) inv_fall=(\d+) restart=(\d+)', line)
            d['strobe_drop'] = (re.search(r'drop=(\d+)', line) or [None, None])[1]
            for s, irq, pulse, inv, rst in m:
                d[f'strobe_s{s}_pulse'] = int(pulse); d[f'strobe_s{s}_restart'] = int(rst); d[f'strobe_s{s}_inv_fall'] = int(inv)
        elif 'close path=' in line:
            m = re.search(r'samples=(\d+) pts=\[(\d+),(\d+)\].*wr_err=(\d+)', line)
            if m: d['imu_samples'], d['imu_pts0'], d['imu_pts1'], d['imu_wr_err'] = map(int, m.groups())
        elif 'close gap' in line:
            m = re.search(r'gap pts g=(\d+)/(\d+)us a=(\d+)/(\d+)us seq g=(\d+)/(\d+) a=(\d+)/(\d+) ovw g=(\d+) a=(\d+)', line)
            if m:
                v = list(map(int, m.groups()))
                d.update(imu_gap_g_n=v[0], imu_gap_g_us=v[1], imu_gap_a_n=v[2], imu_gap_a_us=v[3],
                         imu_seqgap_g=v[4], imu_seqgap_a=v[6], imu_ovw_g=v[8], imu_ovw_a=v[9])
        elif 'close rec' in line:
            m = re.search(r'unwritten g=(\d+) a=(\d+)', line)
            if m: d['imu_unwritten_g'], d['imu_unwritten_a'] = map(int, m.groups())
    if 'stop_frames' in d:
        a, b = d['stop_frames'].split('/'); d['frames_written'], d['frames_expected'] = int(a), int(b)
    return d

# ---------------------------------------------------------------- .imu binary
IMU_DT = np.dtype([('pts', '<u8'), ('seq', '<u4'), ('typ', '<u2'), ('u2', '<u2'), ('sid', '<u4'),
                   ('tseq', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('aux', '<u4')])
assert IMU_DT.itemsize == 40

def read_imu(path):
    with open(path, 'rb') as f:
        head = f.read(64)
        if head[:5] != b'ZXIMU': raise ValueError('not ZXIMU')
        hsz = struct.unpack('<H', head[10:12])[0]
        rec = struct.unpack('<I', head[48:52])[0]
        if rec != 40: raise ValueError(f'unexpected record size {rec}')
        f.seek(hsz)
        a = np.fromfile(f, dtype=IMU_DT)
    return a

def imu_metrics(a):
    out = {'records': int(len(a))}
    types = {int(t): int(n) for t, n in zip(*np.unique(a['typ'], return_counts=True))}
    out['records_by_type'] = types
    res = {}
    for name, typ in (('gyro', 2), ('accel', 4)):
        s = a[a['typ'] == typ]
        if len(s) < 3: continue
        t = s['pts'].astype(np.int64)
        dt = np.diff(t)
        dur = (t[-1] - t[0]) / 1e6
        v = np.stack([s['x'], s['y'], s['z']], 1).astype(np.float64)
        nrm = np.linalg.norm(v, axis=1)
        tq = np.diff(s['tseq'].astype(np.int64))
        res[name] = {
            'n': int(len(s)), 'duration_s': r(dur, 2), 'rate_hz': r((len(s) - 1) / dur if dur > 0 else 0, 3),
            'dt_median_us': r(np.median(dt), 1), 'dt_p99_us': r(pct(dt, 99), 1), 'dt_max_us': int(dt.max()),
            'gaps_gt_5ms': int((dt > 5000).sum()), 'nonmonotonic': int((dt <= 0).sum()),
            'seq_breaks': int((tq != 1).sum()),
            'norm_mean': r(nrm.mean(), 4), 'norm_p99': r(pct(nrm, 99), 4), 'abs_max': r(np.abs(v).max(), 4),
            'aux_min': int(s['aux'].min()), 'aux_max': int(s['aux'].max()),
        }
        if name == 'gyro':
            res[name]['norm_mean_dps'] = r(np.degrees(nrm.mean()), 2)
            res[name]['norm_p99_dps'] = r(np.degrees(pct(nrm, 99)), 2)
    out.update(res)
    g = a[a['typ'] == 2]
    timeline = []
    if len(g):
        t = (g['pts'].astype(np.int64) - int(g['pts'][0])) / 1e6
        sec = t.astype(np.int64)
        nrm = np.degrees(np.linalg.norm(np.stack([g['x'], g['y'], g['z']], 1), axis=1))
        n = int(sec.max()) + 1
        sums = np.bincount(sec, weights=nrm, minlength=n); cnt = np.bincount(sec, minlength=n)
        mx = np.zeros(n); np.maximum.at(mx, sec, nrm)
        timeline = [(i, int(cnt[i]), round(sums[i] / cnt[i], 2) if cnt[i] else None, round(mx[i], 2)) for i in range(n)]
        mean_s = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0)
        out['gyro_still_seconds'] = int((mean_s < 5).sum())        # < 5 deg/s mean
        out['gyro_fast_seconds'] = int((mean_s > 60).sum())        # > 60 deg/s mean
        out['imu_pts0'] = int(a['pts'][0]); out['imu_pts1'] = int(a['pts'][-1])
    return out, timeline

# ---------------------------------------------------------------- SEI per video
SEI_PAT = None  # device-specific layout withheld
def sei_rows(vid):
    raise NotImplementedError('Origin Data Lab device-specific reader withheld; results are in assets/tables. Contact founder@origindatalab.io')

def sei_job(args):
    try:
        return _sei_job(args)
    except Exception as e:
        return {'segment': args[1], 'eye': args[2], 'frames': 0, 'error': str(e)}

def _sei_job(args):
    vid, seg, eye, outdir = args
    t0 = time.time()
    rows = sei_rows(vid)
    os.makedirs(os.path.join(outdir, 'sei'), exist_ok=True)
    with gzip.open(os.path.join(outdir, 'sei', f'{seg}_{eye}.csv.gz'), 'wt', newline='') as fh:
        w = csv.writer(fh); w.writerow(['svh_pts', 'exposure_start_us', 'exposure_us', 'strobe_seq', 'flags']); w.writerows(rows)
    m = {'segment': seg, 'eye': eye, 'frames': len(rows), 'sec': round(time.time() - t0, 1)}
    if len(rows) >= 3:
        pts = np.array([x[0] for x in rows], dtype=np.int64); ex = np.array([x[2] for x in rows], dtype=np.float64)
        st = np.array([x[3] for x in rows], dtype=np.int64); fl = np.array([x[4] for x in rows], dtype=np.int64)
        dt = np.diff(pts); med = float(np.median(dt))
        m.update(pts0=int(pts[0]), pts1=int(pts[-1]), dt_median_us=r(med, 1), dt_max_us=int(dt.max()),
                 drops=int((dt > 1.5 * med).sum()), missing_frames_est=int(np.clip(np.round(dt / med) - 1, 0, None).sum()),
                 nonmonotonic=int((dt <= 0).sum()), strobe_breaks=int((np.diff(st) != 1).sum()),
                 flags_nonzero=int((fl != 0).sum()),
                 exposure_median_us=r(np.median(ex), 0), exposure_p95_us=r(pct(ex, 95), 0), exposure_max_us=int(ex.max()),
                 exposure_gt_8ms=int((ex > 8000).sum()), exposure_gt_16ms=int((ex > 16000).sum()))
        m['_pts'] = pts.tolist(); m['_strobe'] = st.tolist(); m['_exp'] = ex[::10].tolist(); m['_exs'] = [x[1] for x in rows]
    return m

# ---------------------------------------------------------------- stereo / image metrics
_RECT = {}
def load_calib(path):
    import yaml
    c = yaml.safe_load(open(path))
    def kd(cam):
        fx, fy, cx, cy = cam['intrinsics']
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float), np.array(cam['distortion_coeffs'], float).reshape(4, 1)
    K0, D0 = kd(c['cam0']); K1, D1 = kd(c['cam1'])
    T = np.array(c['cam1']['T_cn_cnm1'], float)
    return dict(K0=K0, D0=D0, K1=K1, D1=D1, R=np.ascontiguousarray(T[:3, :3]), t=np.ascontiguousarray(T[:3, 3].reshape(3, 1)), size=tuple(c['cam0']['resolution']),
                baseline_m=float(np.linalg.norm(T[:3, 3])), model=c['cam0'].get('distortion_model'))

def rect_maps(calib_path):
    import cv2
    if calib_path in _RECT: return _RECT[calib_path]
    c = load_calib(calib_path); w, h = c['size']
    R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(c['K0'], c['D0'], c['K1'], c['D1'], (w, h), c['R'], c['t'],
                                                 flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=(w, h), balance=0.0, fov_scale=1.0)
    m0 = cv2.fisheye.initUndistortRectifyMap(c['K0'], c['D0'], R0, P0, (w, h), cv2.CV_16SC2)
    m1 = cv2.fisheye.initUndistortRectifyMap(c['K1'], c['D1'], R1, P1, (w, h), cv2.CV_16SC2)
    _RECT[calib_path] = (m0, m1)
    return _RECT[calib_path]

def epipolar(gl, gr, maps, orb, bf):
    import cv2
    if maps is None:
        L, R = gl, gr
    else:
        (m0a, m0b), (m1a, m1b) = maps
        L = cv2.remap(gl, m0a, m0b, cv2.INTER_LINEAR); R = cv2.remap(gr, m1a, m1b, cv2.INTER_LINEAR)
    kl, dl = orb.detectAndCompute(L, None); kr, dr = orb.detectAndCompute(R, None)
    if dl is None or dr is None or len(kl) < 20 or len(kr) < 20: return None, 0
    dy = []
    for pair in bf.knnMatch(dl, dr, k=2):
        if len(pair) < 2: continue
        a, b = pair
        if a.distance > 0.75 * b.distance: continue
        pl, pr = kl[a.queryIdx].pt, kr[a.trainIdx].pt
        dx = pl[0] - pr[0]
        if 0.5 < dx < 200 and abs(pl[1] - pr[1]) < 30: dy.append(abs(pl[1] - pr[1]))
    if len(dy) < 15: return None, len(dy)
    return float(np.median(dy)), len(dy)

def blur_faces(img, det):
    """Detect faces on a 2x upscaled copy (small, distant faces) and pixelate them with 40% padding.
    Returns (image, n_faces). det must be a YuNet detector."""
    import cv2
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    det.setInputSize((w * 2, h * 2)); det.setScoreThreshold(0.5)
    _, faces = det.detect(big)
    out = img.copy(); n = 0
    if faces is not None:
        for f in faces:
            x, y, fw, fh = [v / 2 for v in f[:4]]
            px, py = fw * 0.4, fh * 0.4
            x0, y0 = max(0, int(x - px)), max(0, int(y - py)); x1, y1 = min(w, int(x + fw + px)), min(h, int(y + fh + py))
            if x1 - x0 < 2 or y1 - y0 < 2: continue
            roi = out[y0:y1, x0:x1]
            small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)), interpolation=cv2.INTER_LINEAR)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST), (0, 0), 3)
            n += 1
    return out, n

def image_job(args):
    try:
        return _image_job(args)
    except Exception as e:
        return {'segment': args[2], 'error': str(e), 'rows': []}

def _image_job(args):
    left, right, seg, calib_path, outdir, first_of_session = args
    import cv2
    cv2.setNumThreads(1)
    t0 = time.time()
    maps = None
    if calib_path:
        try: maps = rect_maps(calib_path)
        except Exception as e: print('rectify failed', e, flush=True)
    orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    face = None
    if os.path.exists(FACE_MODEL) and os.path.getsize(FACE_MODEL) > 100000:
        try:
            face = cv2.FaceDetectorYN.create(FACE_MODEL, '', (800, 600), 0.8)
            blurdet = cv2.FaceDetectorYN.create(FACE_MODEL, '', (1600, 1200), 0.5)
        except Exception: face = None
    cl, cr = cv2.VideoCapture(left), cv2.VideoCapture(right)
    rows = []; i = 0; decoded = 0; cands = []
    cdir = os.path.join(outdir, 'photo_candidates'); os.makedirs(cdir, exist_ok=True)
    while True:
        okl, fl = cl.read(); okr, fr = cr.read()
        if not okl or not okr: break
        decoded += 1
        if i % IMG_STEP == 0:
            row_size = fl.shape[1::-1]
            gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY); gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            row = {'segment': seg, 'frame': i,
                   'bright_L': round(float(gl.mean()), 1), 'bright_R': round(float(gr.mean()), 1),
                   'sharp_L': round(float(cv2.Laplacian(gl, cv2.CV_64F).var()), 1), 'sharp_R': round(float(cv2.Laplacian(gr, cv2.CV_64F).var()), 1),
                   'w': row_size[0], 'h': row_size[1], 'dark_frac_L': round(float((gl < 20).mean()), 3), 'sat_frac_L': round(float((gl > 245).mean()), 3)}
            if maps is not None:
                e, n = epipolar(gl, gr, maps, orb, bf); row['epi_dy_px'] = None if e is None else round(e, 3); row['epi_matches'] = n
                e2, _ = epipolar(gl, gr, None, orb, bf); row['raw_dy_px'] = None if e2 is None else round(e2, 3)
            if face is not None and i % FACE_STEP == 0:
                h, w = fl.shape[:2]; face.setInputSize((w, h))
                _, det = face.detect(fl); row['faces'] = 0 if det is None else int(len(det))
            rows.append(row)
        if i == CAND_FRAME and face is not None:      # no face model -> no photos at all
            try:
                img, nf = blur_faces(fl, blurdet)
                name = f'{seg}_L_f{i:06d}_faces{nf}.jpg'
                cv2.imwrite(os.path.join(cdir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88]); cands.append(name)
                if first_of_session and maps is not None:
                    (m0a, m0b), (m1a, m1b) = maps
                    RL = cv2.remap(fl, m0a, m0b, cv2.INTER_LINEAR); RR = cv2.remap(fr, m1a, m1b, cv2.INTER_LINEAR)
                    RL, n1 = blur_faces(RL, blurdet); RR, n2 = blur_faces(RR, blurdet)
                    sbs = cv2.hconcat([RL, RR])
                    for yy in range(30, sbs.shape[0], 40):
                        cv2.line(sbs, (0, yy), (sbs.shape[1] - 1, yy), (80, 220, 80), 1, cv2.LINE_AA)
                    name2 = f'{seg}_rectified_pair_faces{n1 + n2}.jpg'
                    cv2.imwrite(os.path.join(cdir, name2), sbs, [cv2.IMWRITE_JPEG_QUALITY, 88]); cands.append(name2)
            except Exception as e:
                print('candidate failed', seg, e, flush=True)
        i += 1
    cl.release(); cr.release()
    return {'segment': seg, 'decoded_pairs': decoded, 'rows': rows, 'photos': cands, 'sec': round(time.time() - t0, 1)}

# ---------------------------------------------------------------- charts (no people in them)
ALL_LR = []
def make_charts(out, sessions, sei, samples):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    cd = os.path.join(out, 'charts'); os.makedirs(cd, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.25})
    C1, C2, C3 = '#2563eb', '#f59e0b', '#10b981'
    dates = sorted({s['date'] for s in sessions})

    # 1) epipolar residual by day (calibration stability)
    per = [[x['epi_dy_px'] for x in samples if x['date'] == d and x.get('epi_dy_px') is not None] for d in dates]
    if any(per):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.boxplot([p if p else [np.nan] for p in per], showfliers=False, widths=0.5)
        for k, p in enumerate(per, 1):
            if p: ax.scatter(np.random.default_rng(k).normal(k, 0.06, len(p)), p, s=4, alpha=0.35, color=C1)
        ax.set_xticks(range(1, len(dates) + 1)); ax.set_xticklabels(dates)
        ax.set_ylabel('Rectified vertical residual (px, median per sample)')
        ax.set_title('Stereo calibration held across capture days (same factory calibration)')
        fig.tight_layout(); fig.savefig(os.path.join(cd, 'calibration_stability_by_day.png')); plt.close(fig)

    # 2) left/right sync
    if ALL_LR:
        v = np.array(ALL_LR, float)
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.hist(v, bins=60, color=C1, alpha=0.85)
        ax.set_xlabel('|left - right| exposure start (microseconds)'); ax.set_ylabel('frames (1 in 10 sampled)')
        ax.set_title(f'Left/right sync: median {np.median(v):.0f} us, p99 {np.percentile(v, 99):.0f} us, max {v.max():.0f} us')
        fig.tight_layout(); fig.savefig(os.path.join(cd, 'stereo_sync_histogram.png')); plt.close(fig)

    # 3) exposure time
    ex = np.array([e for m in sei if m.get('eye') == 'left' for e in m.get('_exp', [])], float)
    if len(ex):
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.hist(ex / 1000, bins=60, color=C2, alpha=0.9)
        ax.axvline(8, color='#ef4444', lw=1, ls='--'); ax.text(8.2, ax.get_ylim()[1] * 0.9, '8 ms', color='#ef4444')
        ax.set_xlabel('Exposure time (ms)'); ax.set_ylabel('frames (1 in 10 sampled)')
        ax.set_title(f'Exposure under indoor lighting: median {np.median(ex)/1000:.1f} ms, {100*(ex>8000).mean():.1f}% above 8 ms')
        fig.tight_layout(); fig.savefig(os.path.join(cd, 'exposure_histogram.png')); plt.close(fig)

    # 4) head-motion timeline per day (gyro mean deg/s per second, clock time)
    for d in dates:
        fig, ax = plt.subplots(figsize=(8, 2.6)); drawn = False
        for s in [x for x in sessions if x['date'] == d]:
            f = os.path.join(out, 'imu_activity', s['session'] + '.csv.gz')
            st = s['log'].get('start')
            if not os.path.exists(f) or not st: continue
            t0 = datetime.strptime(st, '%Y-%m-%d %H:%M:%S')
            rows = list(csv.reader(gzip.open(f, 'rt')))[1:]
            xs = [t0 + timedelta(seconds=int(r_[0])) for r_ in rows if r_[2] not in ('', 'None')]
            ys = [float(r_[2]) for r_ in rows if r_[2] not in ('', 'None')]
            if xs:
                k = 10; ys_s = np.convolve(ys, np.ones(k) / k, mode='same')
                ax.plot(xs, ys_s, lw=0.8, color=C3); drawn = True
        if drawn:
            ax.set_ylabel('head rotation (deg/s)'); ax.set_title(f'{d}: head motion during portioning work (10 s smoothing)')
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.tight_layout(); fig.savefig(os.path.join(cd, f'head_motion_{d}.png'))
        plt.close(fig)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--no-images', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, 'imu_activity'), exist_ok=True)
    T0 = time.time()

    days = sorted(d for d in glob.glob(os.path.join(a.root, '*', '*')) if os.path.isdir(os.path.join(d, 'records')))
    sessions = []      # dicts
    sei_tasks, img_tasks = [], []
    for dd in days:
        date = os.path.basename(os.path.dirname(dd)); device = os.path.basename(dd)
        cal = sorted(glob.glob(os.path.join(dd, 'calibration', '*', 'calibration.yaml')))
        cal = cal[0] if cal else None
        lefts = sorted(glob.glob(os.path.join(dd, 'records', '*_left_raw.mp4')))
        by = {}
        for L in lefts:
            base = os.path.basename(L)[:-len('_left_raw.mp4')]
            by.setdefault('_'.join(base.split('_')[:2]), []).append(base)
        for sess, bases in sorted(by.items()):
            s = {'date': date, 'device': device, 'session': sess, 'segments': bases, 'calib': cal}
            logs = sorted(glob.glob(os.path.join(dd, 'records', sess + '_*.log')))
            s['log'] = parse_log(logs[0]) if logs else {}
            s['imu_files'] = sorted(glob.glob(os.path.join(dd, 'records', sess + '_*.imu')))
            sessions.append(s)
            for bi, b in enumerate(bases):
                for eye in ('left', 'right'):
                    v = os.path.join(dd, 'records', f'{b}_{eye}_raw.mp4')
                    if os.path.exists(v): sei_tasks.append((v, b, eye, a.out))
                rv = os.path.join(dd, 'records', f'{b}_right_raw.mp4')
                if not a.no_images and os.path.exists(rv):
                    img_tasks.append((os.path.join(dd, 'records', f'{b}_left_raw.mp4'), rv, b, cal, a.out, bi == 0))
    print(f'days={len(days)} sessions={len(sessions)} videos={len(sei_tasks)} pairs={len(img_tasks)}', flush=True)

    # IMU (main process, numpy)
    for s in sessions:
        mets, tl = [], []
        for f in s['imu_files']:
            try:
                arr = read_imu(f); m, t = imu_metrics(arr); m['file'] = os.path.basename(f); mets.append(m); tl += t
            except Exception as e:
                mets.append({'file': os.path.basename(f), 'error': str(e)})
        s['imu'] = mets
        if tl:
            with gzip.open(os.path.join(a.out, 'imu_activity', s['session'] + '.csv.gz'), 'wt', newline='') as fh:
                w = csv.writer(fh); w.writerow(['second', 'gyro_samples', 'gyro_mean_dps', 'gyro_max_dps']); w.writerows(tl)
        print('imu', s['session'], 'ok', flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        sei = list(ex.map(sei_job, sei_tasks))
    print(f'sei done {time.time()-T0:.0f}s', flush=True)
    img = []
    if img_tasks:
        with ProcessPoolExecutor(a.workers) as ex:
            img = list(ex.map(image_job, img_tasks))
    print(f'images done {time.time()-T0:.0f}s', flush=True)

    sei_by = {(m['segment'], m['eye']): m for m in sei}
    img_by = {m['segment']: m for m in img}

    global ALL_LR
    ALL_LR = []
    seg_rows, sample_rows, summary = [], [], {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sessions': []}
    for s in sessions:
        so = {k: s[k] for k in ('date', 'device', 'session')}
        so['n_segments'] = len(s['segments']); so['log'] = s['log']; so['imu'] = s['imu']
        frames_L = frames_R = 0; drops = 0; missing = 0; pair_off = []; bound = []; exp_all = []
        prev_last = None
        for b in s['segments']:
            L, R = sei_by.get((b, 'left'), {}), sei_by.get((b, 'right'), {})
            frames_L += L.get('frames', 0); frames_R += R.get('frames', 0)
            drops += L.get('drops', 0) + R.get('drops', 0); missing += L.get('missing_frames_est', 0) + R.get('missing_frames_est', 0)
            off = None
            if L.get('_exs') and R.get('_exs'):
                # pair by frame index (strobe counters of the two eyes have a per-session constant offset)
                n = min(len(L['_exs']), len(R['_exs']))
                d = np.array(L['_exs'][:n], dtype=np.int64) - np.array(R['_exs'][:n], dtype=np.int64)
                sc = np.array(L['_strobe'][:n], dtype=np.int64) - np.array(R['_strobe'][:n], dtype=np.int64)
                if n:
                    ALL_LR.extend(np.abs(d[::10]).tolist())
                    off = {'pairs': int(n), 'unpaired_L': int(len(L['_exs']) - n), 'unpaired_R': int(len(R['_exs']) - n),
                           'abs_median_us': r(np.median(np.abs(d)), 1), 'abs_p99_us': r(pct(np.abs(d), 99), 1), 'abs_max_us': int(np.abs(d).max()),
                           'strobe_counter_offset': int(np.median(sc)), 'strobe_offset_constant': bool((sc == sc[0]).all())}
                    pair_off.append(off)
            if L.get('pts0') is not None:
                if prev_last is not None: bound.append(L['pts0'] - prev_last)
                prev_last = L['pts1']
            im = img_by.get(b, {})
            seg_rows.append({'date': s['date'], 'session': s['session'], 'segment': b,
                             'frames_L': L.get('frames'), 'frames_R': R.get('frames'), 'decoded_pairs': im.get('decoded_pairs'),
                             'dt_median_us_L': L.get('dt_median_us'), 'drops_L': L.get('drops'), 'drops_R': R.get('drops'),
                             'missing_est_L': L.get('missing_frames_est'), 'missing_est_R': R.get('missing_frames_est'),
                             'strobe_breaks_L': L.get('strobe_breaks'), 'strobe_breaks_R': R.get('strobe_breaks'),
                             'flags_nonzero_L': L.get('flags_nonzero'),
                             'exp_median_us_L': L.get('exposure_median_us'), 'exp_p95_us_L': L.get('exposure_p95_us'),
                             'exp_gt_8ms_L': L.get('exposure_gt_8ms'),
                             'lr_abs_median_us': off and off['abs_median_us'], 'lr_abs_max_us': off and off['abs_max_us'],
                             'lr_unpaired_L': off and off['unpaired_L'], 'lr_unpaired_R': off and off['unpaired_R'],
                             'strobe_counter_offset': off and off['strobe_counter_offset']})
            for row in im.get('rows', []):
                sample_rows.append({'date': s['date'], 'session': s['session'], **row})
        so['video'] = {'frames_L': frames_L, 'frames_R': frames_R, 'drops_LR': drops, 'missing_est_LR': missing,
                       'segment_boundary_gaps_us': bound,
                       'lr_sync_worst_abs_max_us': max((o['abs_max_us'] for o in pair_off), default=None),
                       'lr_sync_median_of_medians_us': r(statistics.median([o['abs_median_us'] for o in pair_off]), 1) if pair_off else None}
        log = s['log']
        if 'frames_written' in log:
            so['video']['log_frames_written'] = log['frames_written']; so['video']['log_frames_expected'] = log['frames_expected']
        # camera vs IMU time coverage
        imu0 = min((m.get('imu_pts0') for m in s['imu'] if m.get('imu_pts0')), default=None)
        imu1 = max((m.get('imu_pts1') for m in s['imu'] if m.get('imu_pts1')), default=None)
        v0 = sei_by.get((s['segments'][0], 'left'), {}).get('pts0'); v1 = sei_by.get((s['segments'][-1], 'left'), {}).get('pts1')
        if None not in (imu0, imu1, v0, v1):
            so['cam_imu'] = {'video_start_minus_imu_start_us': v0 - imu0, 'imu_end_minus_video_end_us': imu1 - v1,
                             'video_span_s': r((v1 - v0) / 1e6, 2), 'imu_span_s': r((imu1 - imu0) / 1e6, 2)}
        sm = [x for x in sample_rows if x['session'] == s['session'] and x['date'] == s['date']]
        epi = [x['epi_dy_px'] for x in sm if x.get('epi_dy_px') is not None]
        so['image'] = {'samples': len(sm), 'epi_samples': len(epi),
                       'epi_dy_median_px': r(np.median(epi), 3) if epi else None, 'epi_dy_p90_px': r(pct(epi, 90), 3) if epi else None,
                       'sharp_L_median': r(np.median([x['sharp_L'] for x in sm]), 1) if sm else None,
                       'bright_L_median': r(np.median([x['bright_L'] for x in sm]), 1) if sm else None,
                       'face_checks': sum(1 for x in sm if 'faces' in x),
                       'face_checks_with_face': sum(1 for x in sm if x.get('faces', 0) > 0)}
        summary['sessions'].append(so)

    if sessions and sessions[0]['calib']:
        c = load_calib(sessions[0]['calib'])
        summary['calibration'] = {'model': c['model'], 'resolution': c['size'], 'baseline_m': r(c['baseline_m'], 5)}
    summary['runtime_s'] = round(time.time() - T0, 1)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, ensure_ascii=False, default=str)

    def dump(name, rows):
        if not rows: return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(os.path.join(a.out, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    dump('segments.csv', seg_rows); dump('samples.csv', sample_rows)
    sess_rows = []
    for so in summary['sessions']:
        g = next((m.get('gyro', {}) for m in so['imu'] if 'gyro' in m), {})
        ac = next((m.get('accel', {}) for m in so['imu'] if 'accel' in m), {})
        im0 = next((m for m in so['imu'] if 'gyro' in m), {})
        sess_rows.append({'date': so['date'], 'session': so['session'], 'segments': so['n_segments'],
                          'start': so['log'].get('start'), 'stop': so['log'].get('stop'),
                          'rec_s': r(int(so['log'].get('stop_rec_ms', 0)) / 1000, 1) if so['log'].get('stop_rec_ms') else None,
                          'frames_L': so['video']['frames_L'], 'frames_R': so['video']['frames_R'],
                          'log_frames': so['log'].get('stop_frames'), 'drops_LR': so['video']['drops_LR'],
                          'lr_sync_worst_us': so['video']['lr_sync_worst_abs_max_us'],
                          'boundary_gap_max_us': max(so['video']['segment_boundary_gaps_us'], default=None),
                          'gyro_rate_hz': g.get('rate_hz'), 'gyro_gaps_5ms': g.get('gaps_gt_5ms'), 'gyro_dt_max_us': g.get('dt_max_us'),
                          'accel_norm_mean': ac.get('norm_mean'), 'gyro_mean_dps': g.get('norm_mean_dps'),
                          'still_s': im0.get('gyro_still_seconds'), 'fast_s': im0.get('gyro_fast_seconds'),
                          'imu_wr_err': so['log'].get('imu_wr_err'), 'strobe_drop': so['log'].get('strobe_drop'),
                          'epi_dy_median_px': so['image']['epi_dy_median_px'], 'sharp_L_median': so['image']['sharp_L_median'],
                          'face_checks_with_face': so['image']['face_checks_with_face']})
    dump('sessions.csv', sess_rows)

    try:
        make_charts(a.out, sessions, sei, sample_rows)
    except Exception as e:
        print('charts failed', e, flush=True)
    summary['photo_candidates'] = sorted(p for m in img for p in m.get('photos', []))
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, ensure_ascii=False, default=str)

    # short human report
    L = ['# ODL 촬영 분석 요약', '', f"생성: {summary['generated_utc']} · 소요 {summary['runtime_s']}초", '']
    if 'calibration' in summary: L += [f"보정: {summary['calibration']}", '']
    L += ['| 날짜 | 세션 | 녹화(초) | 프레임 L/R | 누락 | 좌우 동기 최대(µs) | 파일 경계 최대(µs) | 자이로 Hz | IMU 끊김(>5ms) | 에피폴라 dy(px) |', '|---|---|---|---|---|---|---|---|---|---|']
    for x in sess_rows:
        L.append(f"| {x['date']} | {x['session']} | {x['rec_s']} | {x['frames_L']}/{x['frames_R']} | {x['drops_LR']} | {x['lr_sync_worst_us']} | {x['boundary_gap_max_us']} | {x['gyro_rate_hz']} | {x['gyro_gaps_5ms']} | {x['epi_dy_median_px']} |")
    open(os.path.join(a.out, 'report.md'), 'w').write('\n'.join(L) + '\n')
    print('done', summary['runtime_s'], flush=True)

if __name__ == '__main__':
    main()
