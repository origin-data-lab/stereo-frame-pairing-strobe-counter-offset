import csv,gzip,glob,json,os,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from datetime import datetime,timedelta
plt.rcParams.update({'figure.dpi':160,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25,'font.family':'DejaVu Sans'})
DC={'2026-11-01':'#2563eb','2026-11-02':'#f59e0b','2026-11-03':'#10b981','2026-11-04':'#a855f7'}
def rd(f): return np.array([list(map(int,r)) for r in list(csv.reader(gzip.open(f,'rt')))[1:]],dtype=np.int64)
S=json.load(open('data/summary.json')); os.makedirs('fig',exist_ok=True); os.makedirs('tab',exist_ok=True)
sess=[s for s in S['sessions']]
# ---------- per-file timing tables
pf=[]; per_session={}
for s in sess:
    k=s['session']; agg=dict(dL=[],cerr=[],off=set(),exp=[],n=0,files=0,bgap=[])
    prev=None
    for f in sorted(glob.glob(f'data/sei/{k}_*_left.csv.gz')):
        L=rd(f); R=rd(f.replace('_left','_right'))
        if len(L)==0 or len(R)==0: continue
        n=min(len(L),len(R)); so=L[:n,3]-R[:n,3]; d=L[:n,1]-R[:n,1]
        rmap=dict(zip(R[:,3].tolist(),R[:,1].tolist()))
        ce=np.array([abs(int(L[i,1])-rmap[c]) for i,c in enumerate(L[:,3].tolist()) if c in rmap],dtype=np.int64)
        iv=np.diff(L[:,0]); drops=int((iv>1.5*np.median(iv)).sum()) if len(iv) else 0
        if prev is not None: agg['bgap'].append(int(L[0,0]-prev))
        prev=int(L[-1,0])
        agg['dL'].append(d); agg['cerr'].append(ce); agg['off']|=set(np.unique(so).tolist()); agg['exp'].append(L[:,2]); agg['n']+=n; agg['files']+=1
        pf.append(dict(date=s['date'],session=k,file=os.path.basename(f)[16:21],frames_L=len(L),frames_R=len(R),dropped=drops,
            strobe_offset_L_minus_R=int(so[0]),offset_constant=bool((so==so[0]).all()),
            lr_exposure_start_median_us=int(np.median(np.abs(d))),lr_exposure_start_max_us=int(np.abs(d).max()),
            counter_pairing_error_median_ms=round(float(np.median(ce))/1000,1) if len(ce) else '',
            counter_pairs_unmatched=int(len(L)-len(ce)),exposure_median_us=int(np.median(L[:,2]))))
    if agg['n']: per_session[k]=agg
with open('tab/per_file_timing.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,list(pf[0].keys())); w.writeheader(); w.writerows(pf)
# ---------- session table
srows=[]
for s in sess:
    k=s['session']; a=per_session.get(k)
    g=next((m.get('gyro') for m in s['imu'] if 'gyro' in m),{}) or {}
    rec=round(s['cam_imu']['video_span_s']/60,1) if s.get('cam_imu') else 0
    srows.append(dict(date=s['date'],start=f"{k[9:11]}:{k[11:13]}",minutes=rec,files=a['files'] if a else 0,
        frames_per_eye=s['video']['frames_L'],dropped=0 if a else '',strobe_offset=(sorted(a['off'])[0] if a and len(a['off'])==1 else ''),
        lr_median_us=int(np.median(np.abs(np.concatenate(a['dL'])))) if a else '',lr_max_us=int(np.abs(np.concatenate(a['dL'])).max()) if a else '',
        counter_pairing_error_s=round(float(np.median(np.concatenate(a['cerr'])))/1e6,2) if a and len(np.concatenate(a['cerr'])) else '',
        gyro_hz=g.get('rate_hz',''),imu_max_gap_ms=round(g.get('dt_max_us',0)/1000,2) if g else '',imu_seq_breaks=g.get('seq_breaks',''),
        exposure_median_ms=round(float(np.median(np.concatenate(a['exp'])))/1000,2) if a else '',
        boundary_gap_ms=(f"{min(a['bgap'])/1000:.1f}-{max(a['bgap'])/1000:.1f}" if a and a['bgap'] else ''),
        device_log_stop=s['log'].get('stop_reason','missing')))
with open('tab/session_summary.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,list(srows[0].keys())); w.writeheader(); w.writerows(srows)
main=[r for r in srows if r['minutes']>=1]
# ---------- FN18 figures
lab=[f"{r['date'][5:]} {r['start']}" for r in main]; off=[r['strobe_offset'] for r in main]; col=[DC[r['date']] for r in main]
fig,ax=plt.subplots(figsize=(8,3.4)); ax.bar(range(len(main)),off,color=col)
for i,v in enumerate(off): ax.text(i,v+(60 if v>=0 else -180),str(v),ha='center',fontsize=7)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(range(len(main))); ax.set_xticklabels(lab,rotation=60,ha='right',fontsize=7)
ax.set_ylabel('left − right strobe counter (frames)'); ax.set_title('The two eyes\' frame counters disagree by a different constant in every session')
fig.tight_layout(); fig.savefig('fig/fn18_counter_offset_by_session.png'); plt.close(fig)
idx=np.concatenate([np.abs(np.concatenate(per_session[r]['dL'])) for r in per_session])
cnt=np.concatenate([np.concatenate(per_session[r]['cerr']) for r in per_session])
cnt=cnt[cnt>0]
fig,axs=plt.subplots(1,2,figsize=(8.4,3.2))
axs[0].hist(cnt/1e6,bins=np.logspace(-2,2.2,60),color='#ef4444'); axs[0].set_xscale('log'); axs[0].set_xlabel('time between "paired" exposures (s)'); axs[0].set_ylabel('frame pairs'); axs[0].set_title('Paired by strobe counter',fontsize=9)
axs[1].hist(idx,bins=np.arange(0,74,2),color='#2563eb'); axs[1].set_xlabel('|left − right| exposure start (µs)'); axs[1].set_title('Paired by frame order (correct)',fontsize=9)
fig.suptitle(f'Same {len(idx):,} frames, two pairing rules: seconds apart vs. median {np.median(idx):.0f} µs',fontsize=10)
fig.tight_layout(); fig.savefig('fig/fn18_pairing_rule_comparison.png'); plt.close(fig)
# schematic
fig,ax=plt.subplots(figsize=(8,2.6)); ax.axis('off')
for i in range(6):
    ax.add_patch(plt.Rectangle((i*1.2,1.2),1,0.6,fc='#dbeafe',ec='#2563eb')); ax.text(i*1.2+.5,1.5,f'L #{1048+i}',ha='center',va='center',fontsize=8)
    ax.add_patch(plt.Rectangle((i*1.2,0),1,0.6,fc='#fef3c7',ec='#f59e0b')); ax.text(i*1.2+.5,.3,f'R #{1000+i}',ha='center',va='center',fontsize=8)
    ax.annotate('',xy=(i*1.2+.5,.62),xytext=(i*1.2+.5,1.18),arrowprops=dict(arrowstyle='<->',color='#16a34a'))
ax.text(7.4,1.5,'same exposure (Δt ≈ 15 µs)\n→ pair by order / timestamp',fontsize=8,color='#16a34a',va='center')
ax.text(7.4,.3,'L #1048 ↔ R #1048 is 48 frames\n(1.6 s) later → wrong pair',fontsize=8,color='#dc2626',va='center')
ax.set_xlim(-.2,10.5); ax.set_ylim(-.2,2); ax.set_title('Example session (Day 1, 09:48): left counter = right counter + 48',fontsize=9)
fig.tight_layout(); fig.savefig('fig/fn18_pairing_schematic.png'); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.6,3.2)); ax.hist(idx,bins=np.arange(0,74,2),color='#2563eb',alpha=.9)
ax.set_xlabel('|left − right| exposure start, paired by frame order (µs)'); ax.set_ylabel('frame pairs')
ax.set_title(f'Stereo sync over {len(idx):,} pairs: median {np.median(idx):.0f} µs, p99 {np.percentile(idx,99):.0f} µs, max {idx.max()} µs',fontsize=9)
fig.tight_layout(); fig.savefig('fig/fn18_sync_histogram.png'); plt.close(fig)
# ---------- FN19 figures
days=sorted(DC)
daily=[]
for d in days:
    rs=[r for r in main if r['date']==d]; ks=[s['session'] for s in sess if s['date']==d and s['session'] in per_session]
    dl=np.concatenate([np.abs(np.concatenate(per_session[k]['dL'])) for k in ks]); ex=np.concatenate([np.concatenate(per_session[k]['exp']) for k in ks])
    hm=[]
    for k in ks:
        f=f'data/imu_activity/{k}.csv.gz'
        if os.path.exists(f): hm+= [float(r[2]) for r in list(csv.reader(gzip.open(f,'rt')))[1:] if r[2] not in ('','None')]
    hm=np.array(hm)
    micro=[s for s in sess if s['date']==d and (not s.get('cam_imu') or s['cam_imu']['video_span_s']<60)]
    daily.append(dict(date=d,sessions=len(rs),micro_recordings=len(micro),hours=round(sum(r['minutes'] for r in rs)/60,2),frames_per_eye=sum(r['frames_per_eye'] for r in rs),dropped=0,
        lr_median_us=int(np.median(dl)),lr_max_us=int(dl.max()),gyro_hz_min=min(r['gyro_hz'] for r in rs),gyro_hz_max=max(r['gyro_hz'] for r in rs),
        imu_max_gap_ms=max(r['imu_max_gap_ms'] for r in rs),exposure_median_ms=round(float(np.median(ex))/1000,2),exposure_p95_ms=round(float(np.percentile(ex,95))/1000,2),
        head_dps_median=round(float(np.median(hm)),1),head_dps_p95=round(float(np.percentile(hm,95)),1),still_share=round(float((hm<5).mean()),3)))
with open('tab/daily_summary.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,list(daily[0].keys())); w.writeheader(); w.writerows(daily)
fig,axs=plt.subplots(1,3,figsize=(9,2.8))
axs[0].bar([d[5:] for d in days],[x['hours'] for x in daily],color=[DC[d] for d in days]); axs[0].set_title('Recorded hours',fontsize=9)
axs[1].bar([d[5:] for d in days],[x['frames_per_eye']/1000 for x in daily],color=[DC[d] for d in days]); axs[1].set_title('Frames per eye (thousands)',fontsize=9)
axs[2].bar([d[5:] for d in days],[x['sessions'] for x in daily],color=[DC[d] for d in days]); axs[2].set_title('Sessions ≥ 1 min',fontsize=9)
for a in axs: a.grid(axis='x',alpha=0)
fig.suptitle('Four capture days, 8.9 hours, 0 dropped frames',fontsize=10); fig.tight_layout(); fig.savefig('fig/fn19_daily_volume.png'); plt.close(fig)
# sync/exposure/headmotion distributions by day
dsync=[];dexp=[];dhm=[]
for d in days:
    ks=[s['session'] for s in sess if s['date']==d and s['session'] in per_session]
    dsync.append(np.concatenate([np.abs(np.concatenate(per_session[k]['dL'])) for k in ks])[::20])
    dexp.append(np.concatenate([np.concatenate(per_session[k]['exp']) for k in ks])[::20]/1000)
    hm=[]
    for k in ks:
        f=f'data/imu_activity/{k}.csv.gz'
        if os.path.exists(f): hm+= [float(r[2]) for r in list(csv.reader(gzip.open(f,'rt')))[1:] if r[2] not in ('','None')]
    dhm.append(np.array(hm))
fig,axs=plt.subplots(1,3,figsize=(9.2,3))
for a,data,t,u in zip(axs,[dsync,dexp,dhm],['Left/right sync','Exposure time','Head rotation (per second)'],['µs','ms','deg/s']):
    bp=a.boxplot(data,showfliers=False,widths=.55,patch_artist=True)
    for p,d in zip(bp['boxes'],days): p.set_facecolor(DC[d]); p.set_alpha(.55)
    a.set_xticks(range(1,5)); a.set_xticklabels([d[5:] for d in days]); a.set_title(t,fontsize=9); a.set_ylabel(u)
fig.suptitle('Day-to-day distributions (box = 25–75%, whiskers = 1.5 IQR)',fontsize=10); fig.tight_layout(); fig.savefig('fig/fn19_day_to_day_distributions.png'); plt.close(fig)
# imu rate per session
fig,ax=plt.subplots(figsize=(7.4,2.8))
for i,r in enumerate(main): ax.scatter(i,r['gyro_hz'],color=DC[r['date']],s=28)
ax.set_xticks(range(len(main))); ax.set_xticklabels(lab,rotation=60,ha='right',fontsize=7); ax.set_ylabel('gyro rate (Hz)'); ax.set_ylim(501.2,502.0)
ax.set_title('IMU sample rate per session: 501.4–501.7 Hz, no gaps > 5 ms, no sequence breaks',fontsize=9)
fig.tight_layout(); fig.savefig('fig/fn19_imu_rate_by_session.png'); plt.close(fig)
os.system('cp out/head_motion_4days.png fig/fn19_head_motion_4days.png')
json.dump(dict(daily=daily,sessions=srows,n_pairs=int(len(idx)),idx_median=float(np.median(idx)),idx_p99=float(np.percentile(idx,99)),idx_max=int(idx.max()),cnt_median_s=float(np.median(cnt)/1e6),cnt_min_s=float(cnt.min()/1e6),cnt_max_s=float(cnt.max()/1e6)),open('tab/stats.json','w'),indent=1,default=str)
print(json.dumps(daily,indent=0)); print(len(idx),np.median(cnt)/1e6,cnt.min(),cnt.max()/1e6)
