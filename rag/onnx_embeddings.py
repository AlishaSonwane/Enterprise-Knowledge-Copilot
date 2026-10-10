"""Pinned MiniLM CPU inference; no Torch, network or automatic model download."""
import hashlib
import pathlib
import numpy as np

MODEL_SHA='6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452'

class MiniLM:
    def __init__(self,model_path,tokenizer_path):
        path=pathlib.Path(model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA:
            raise ValueError('ONNX integrity mismatch')
        if hashlib.sha256(pathlib.Path(tokenizer_path).read_bytes()).hexdigest()!='be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037':
            raise ValueError('Tokenizer integrity mismatch')
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer=Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        options=ort.SessionOptions()
        options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def encode(self,texts,batch_size=8,normalize_embeddings=True,**kwargs):
        if batch_size<=0:raise ValueError('Invalid batch size')
        output=[]
        for start in range(0,len(texts),min(batch_size,8)):
            batch=self.tokenizer.encode_batch([t.strip() for t in texts[start:start+min(batch_size,8)]])
            arrays={
                'input_ids':np.asarray([t.ids for t in batch],dtype=np.int64),
                'attention_mask':np.asarray([t.attention_mask for t in batch],dtype=np.int64),
                'token_type_ids':np.asarray([t.type_ids for t in batch],dtype=np.int64)}
            names={i.name for i in self.session.get_inputs()}
            hidden=self.session.run(None,{k:v for k,v in arrays.items() if k in names})[0]
            mask=arrays['attention_mask'].astype(np.float32)[...,None]
            pooled=(hidden*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1e-9)
            if normalize_embeddings:
                pooled=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
            if pooled.shape[1]!=384 or not np.isfinite(pooled).all():raise ValueError('Invalid ONNX embedding')
            output.append(pooled.astype(np.float32))
        return np.concatenate(output) if output else np.empty((0,384),dtype=np.float32)
