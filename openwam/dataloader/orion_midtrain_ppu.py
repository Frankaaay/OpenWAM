"""PPU path mapping and native-crop left-eye access for the frozen ABC stereo subset."""
import json,bisect
from pathlib import Path
import numpy as np,pandas as pd,torch,pyarrow.parquet as pq,av
from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.orion_midtrain import DROIDOrionDataset,scatter80,mosaic,LAYOUT
from dataloader.pretrain.canonical import ORIONPretrainDataset,interpolate_canonical_tcp
from dataloader.pretrain.windows import timeline_sample_indices
from dataloader.pretrain.action import dual_arm_tcp_target_relative_to_anchor,normalize_dual_arm_absolute_tcp,normalize_dual_arm_relative_tcp

from openwam.dataloader.utils import get_cfg

class DROIDPPUOrionDataset(DROIDOrionDataset):
    def __init__(self,index_path,stats_path,state_root,image_root,split):
        self.state_root=Path(state_root)
        self.image_root=Path(image_root)
        super().__init__(index_path,stats_path,split,'mono-left')
    def record(self,index):
        record=super().record(index)
        original=Path(record['trajectory_h5']).relative_to('/gpfs/jiuquyun')
        record['trajectory_h5']=str(self.state_root/original)
        original=Path(record['table_root']).relative_to('/gpfs/jiuquyun/projects/Frank/droid-prepared-20260920')
        record['table_root']=str(self.image_root/original)
        return record

    @classmethod
    def from_config(cls,cfg,split='train'):
        if get_cfg(cfg,'input_mode','mono-left')!='mono-left':
            raise ValueError('PPU DROID adapter requires mono-left')
        return cls(get_cfg(cfg,'index_path'),get_cfg(cfg,'stats_path'),get_cfg(cfg,'state_root'),get_cfg(cfg,'image_root'),split)


class ABCNativeCropOrionDataset(BaseDataset):
    action_dim=80
    def __init__(self,dataset_dir,stats_path,split):
        self._dataset_dir=Path(dataset_dir)
        with (self._dataset_dir/'catalog.jsonl').open() as f:
            self.episodes=[e for e in map(json.loads,f) if e['source_split']==split]
        self.ends=np.cumsum([e['sparse_windows'] for e in self.episodes])
        self._cum_n_starts=np.concatenate(([0],self.ends))
        self._eps_df=pd.DataFrame({'episode_index':[e['h100_subset_ordinal'] for e in self.episodes]})
        assert self._eps_df['episode_index'].is_unique
        assert [e['split_window_start'] for e in self.episodes]==self._cum_n_starts[:-1].tolist()
        self.stats=json.loads(Path(stats_path).read_text())
        assert self.stats['statistics']['window_stride_rgb']==8
        self.stats={k:torch.tensor(v,dtype=torch.float32) for k,v in self.stats.items() if k in ('state_tcp_mean','state_tcp_std','action_tcp_mean','action_tcp_std','gripper_open_value','gripper_closed_value')}
        self.meta={};self.cache_key=None;self.rows=None
    def __len__(self):return int(self.ends[-1])
    def __getitem__(self,index):
        i=bisect.bisect_right(self.ends,int(index));e=self.episodes[i]
        local=int(index)-(int(self.ends[i-1]) if i else 0);anchor=local*8;phase=int(index)%3
        assert int(index)==e['split_window_start']+local
        timeline=timeline_sample_indices(anchor,rgb_episode_length=e['rgb_target_length'],action_episode_length=e['action_target_length'],layout=LAYOUT)
        key=(e['state_file'],e['episode_index'])
        if self.cache_key!=key:
            self.rows=pq.read_table(key[0],columns=['frame_index','observation.state','task_index'],filters=[('episode_index','=',int(key[1]))],use_threads=False).to_pylist()
            self.rows.sort(key=lambda r:r['frame_index']);self.cache_key=key
            assert [r['frame_index'] for r in self.rows]==list(range(e['length']))
        state=torch.tensor([r['observation.state'] for r in self.rows],dtype=torch.float32).reshape(-1,2,10)
        targets=torch.cat([torch.tensor([anchor*2]),timeline.action_indices.reshape(-1)])
        lo,hi,fraction=ORIONPretrainDataset._state_interpolation_indices(targets,30.,20.,e['length'],phase_offset=phase)
        grid=interpolate_canonical_tcp(state[lo],state[hi],fraction).reshape(-1,20)
        relative=dual_arm_tcp_target_relative_to_anchor(grid[:1],grid[1:]);s=self.stats
        action=normalize_dual_arm_relative_tcp(relative,s['action_tcp_mean'],s['action_tcp_std'],s['gripper_open_value'],s['gripper_closed_value'])
        prop=normalize_dual_arm_absolute_tcp(grid[:1],s['state_tcp_mean'],s['state_tcp_std'],s['gripper_open_value'],s['gripper_closed_value'])
        action,mask=scatter80(action,timeline.action_valid.reshape(-1,1).expand(-1,20))
        prop,pmask=scatter80(prop,torch.ones_like(prop,dtype=torch.bool))
        root=Path(e['root'])
        if e['root'] not in self.meta:
            info=json.loads((root/'meta/info.json').read_text());crop=json.loads((root/'meta/native_crop.json').read_text())
            tasks={int(t['task_index']):t.get('task',t.get('tasks')) for t in pq.read_table(root/'meta/tasks.parquet').to_pylist()}
            self.meta[e['root']]=(info,crop,tasks)
        info,crop,tasks=self.meta[e['root']];camera='observation.images.cam_head_left'
        video=root/info['video_path'].format(video_key=camera,chunk_index=e['videos/'+camera+'/chunk_index'],file_index=e['videos/'+camera+'/file_index'])
        source_indices=timeline.frame_indices.numpy()*3+phase
        wanted=e['videos/'+camera+'/from_timestamp']+source_indices/30.
        frames=[];times=[]
        with av.open(str(video)) as container:
            stream=container.streams.video[0];stream.thread_count=1
            container.seek(max(0,int((wanted[0]-.1)/float(stream.time_base))),stream=stream,backward=True)
            for frame in container.decode(stream):
                t=float(frame.pts*stream.time_base)
                if t>=wanted[0]-.06:frames.append(frame.to_ndarray(format='rgb24'));times.append(t)
                if t>wanted[-1]+.06:break
        picks=np.abs(np.array(times)[:,None]-wanted[None,:]).argmin(axis=0)
        assert np.max(np.abs(np.array(times)[picks]-wanted))<.018,(video,wanted)
        x,y,w,h=crop['crops'][camera];images=[]
        for j,pick in enumerate(picks):
            arr=frames[pick];assert arr.shape[:2]==(h,w)
            square=np.zeros((256,256,3),dtype=np.uint8)
            if timeline.frame_valid[j]:square[y:y+h,x:x+w]=arr
            images.append(square)
        task=tasks[int(self.rows[anchor*3+phase]['task_index'])]
        assert isinstance(task,str) and task.strip()
        video=mosaic([images])
        assert torch.isfinite(action).all() and torch.isfinite(prop).all()
        return dict(video=video,right_video=None,first_frame_image=[video[0]],vace_video=None,action=action,action_mask=mask,proprio=prop,proprio_mask=pmask,video_mask=timeline.frame_valid,prompt="A video recorded from a robot's point of view executing the following instruction: "+task,episode_index=e['h100_subset_ordinal'])

    @classmethod
    def from_config(cls,cfg,split='train'):
        if get_cfg(cfg,'input_mode','mono-left')!='mono-left':
            raise ValueError('Native-crop adapter requires mono-left')
        return cls(get_cfg(cfg,'dataset_dir'),get_cfg(cfg,'stats_path'),split)
