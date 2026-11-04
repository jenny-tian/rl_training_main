"""Lossless, bounded little-endian tensor payloads inside ordinary JSON."""
import base64
import math
import numpy as np

DTYPES={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8',
        'torch.int8':'i1','torch.int16':'<i2','torch.int32':'<i4','torch.int64':'<i8',
        'torch.uint8':'u1','torch.bool':'?'}
TAG='__lite3_tensor_v1__'
MAX_BYTES=64*1024**2


def pack(value):
    if isinstance(value,dict):
        if set(value)=={'dtype','shape','values'} and value['dtype'] in DTYPES:
            array=np.asarray(value['values'],dtype=DTYPES[value['dtype']])
            original=np.asarray(value['values'])
            if (list(array.shape)!=value['shape'] or array.nbytes>MAX_BYTES
                    or not np.isfinite(array).all() or not np.array_equal(array,original)):
                raise ValueError('tensor encoding would lose evidence or exceed its bound')
            return {TAG:True,'dtype':value['dtype'],'shape':value['shape'],
                    'data':base64.b64encode(array.tobytes(order='C')).decode('ascii')}
        return {k:pack(v) for k,v in value.items()}
    if isinstance(value,list):return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value,dict):
        if TAG in value:
            if (set(value)!={TAG,'dtype','shape','data'} or value[TAG] is not True
                    or value['dtype'] not in DTYPES or not isinstance(value['shape'],list)
                    or len(value['shape'])>8 or any(type(i) is not int or i<0 for i in value['shape'])):
                raise ValueError('malformed binary tensor metadata')
            dtype=np.dtype(DTYPES[value['dtype']]);size=math.prod(value['shape'])*dtype.itemsize
            if size>MAX_BYTES or not isinstance(value['data'],str) or len(value['data'])>4*((size+2)//3):
                raise ValueError('binary tensor exceeds its declared bound')
            raw=base64.b64decode(value['data'],validate=True)
            if len(raw)!=size:raise ValueError('binary tensor payload length differs')
            array=np.frombuffer(raw,dtype=dtype).reshape(value['shape'])
            if not np.isfinite(array).all():raise ValueError('nonfinite binary tensor evidence')
            return dict(dtype=value['dtype'],shape=value['shape'],values=array.tolist())
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):return [unpack(v) for v in value]
    return value
