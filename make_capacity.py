"""按固定种子生成容量输入，不计算算子输出。"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parent

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(261126)
    arrays={};meta={'format_version':1,'source':'本地合成容量输入','seed':261126}
    for key in ['Ks','Kc']:arrays[key]=rng.normal(0,.2,(4096,16))
    arrays['V']=rng.normal(0,.2,(4096,16));arrays['Q']=rng.normal(0,.2,(128,16));arrays['b_select']=np.full(128,-2.)
    arrays['valid']=np.ones(4096,dtype='bool');arrays['context_upstream']=rng.normal(0,.1,(128,16))
    meta.update(chunk_width=64,words=['w'+str(i) for i in range(128)],sample_weight=1,confirmed_blocks=64,block_size=64)
    np.savez(output/'arrays.npz',**arrays)
    meta['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='生成可复现的合成容量输入')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();meta=generate(args.output)
    print(json.dumps({'数组数':len(meta['arrays']),'资料SHA256':hashlib.sha256((Path(args.output)/'arrays.npz').read_bytes()).hexdigest()},ensure_ascii=False))
