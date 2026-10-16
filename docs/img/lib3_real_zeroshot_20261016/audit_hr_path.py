import sys,copy,json
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from unittest.mock import patch
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'pneu_rl/scripts'))
from quick_viz3 import model_bank_reference,build_viz_env
from pneu_env.sim3 import PneuSim
from pneu_env.real.real9 import PneuReal
from pneu_rl.sac_with_loss import SAC
cfg=yaml.safe_load((root/'pneu_rl/models/261015_lib3_allopen30_HR_smooth_seed0/cfg.yaml').read_text());torch.set_num_threads(2)
out=root/'docs/img/lib3_real_zeroshot_20261016';out.mkdir(exist_ok=True)
real=PneuReal(**cfg['obs']);captured=[]
def write():
 b=real.backend;captured.append([b.pos_ctrl,b.neg_ctrl,b.act_pos_ctrl1,b.act_pos_ctrl2,b.act_neg_ctrl1,b.act_neg_ctrl2])
def read():real.backend.time+=.02
vectors=np.r_[np.zeros((1,6)),np.ones((1,6)),np.eye(6),np.random.default_rng(5).uniform(0,1,(100,6))]
with patch.object(real.backend,'write_ctrl_file',side_effect=write),patch.object(real.backend,'read_obs_file',side_effect=read),patch.object(real.backend,'wait'):
 for v in vectors:real.observe(v)
print('MAPPING max error',np.max(np.abs(np.array(captured)-(.8+.2*vectors))),flush=True)
rows=[]
for delay in [.1,0,.02]:
 c=copy.deepcopy(cfg);c['obs']['delay']=delay;c['env']['verbose']=False
 ref,_=model_bank_reference(c);env=build_viz_env(c,PneuSim(**c['obs']),ref);model=SAC(env=env,**c['model']);model.load_model(str(root/'pneu_rl/models/261015_lib3_allopen30_HR_smooth_seed0/model.pth'))
 state,_=env.reset();data=[]
 for i in range(2000):
  a=model.predict(state);state,_,_,_,info=env.step(a);o=info['obs'];data.append([i*.02,o['sen_act_pos'],o['sen_act_neg'],o['ref_act_pos'],o['ref_act_neg'],*(.8+.2*a)])
 df=pd.DataFrame(data,columns=['time','pos','neg','ref_pos','ref_neg']+[f'u{i}' for i in range(6)]);df.to_csv(out/f'delay_audit_{delay:.2f}.csv',index=False);z=df[df.time>=6]
 r=dict(delay=delay,pos_rmse=float(np.sqrt(np.mean((z.pos-z.ref_pos)**2))),neg_rmse=float(np.sqrt(np.mean((z.neg-z.ref_neg)**2))),command_delta=float(np.abs(np.diff(z[[f'u{i}' for i in range(6)]],axis=0)).mean()));rows.append(r);print(r,flush=True)
 env.close()
pd.DataFrame(rows).to_csv(out/'delay_audit_summary.csv',index=False)
