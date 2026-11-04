"""Keep yaw-aligned scanner footprints inside the assigned finite Source tile."""
import math
import torch


def scanner_boundary(root, quaternion, velocity, yaw_rate, origins, *, margin=.1, dt=.02,
                     half_extents=(.87,.57), diagnostics=None):
    n=len(root)
    if (root.shape!=(n,3) or origins.shape!=(n,3) or quaternion.shape!=(n,4)
            or velocity.shape!=(n,3) or yaw_rate.shape!=(n,)):
        raise ValueError('scanner boundary state dimensions differ')
    if any(not bool(torch.isfinite(x).all()) for x in (root,quaternion,velocity,yaw_rate,origins)):
        raise FloatingPointError('nonfinite scanner boundary state')
    if not torch.allclose(quaternion.norm(dim=-1),torch.ones(n,device=root.device),atol=1e-4,rtol=0):
        raise ValueError('invalid scanner boundary quaternion')
    if not math.isfinite(margin) or not 0<margin<.5 or dt!=.02 or tuple(half_extents)!=(.87,.57):
        raise ValueError('unregistered scanner boundary contract')
    w,x,y,z=quaternion.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    corners=root.new_tensor([[-.87,-.57],[-.87,.57],[.87,-.57],[.87,.57]])
    def inside(center,angle):
        c,s=angle.cos()[:,None],angle.sin()[:,None]
        px=center[:,None,0]+c*corners[None,:,0]-s*corners[None,:,1]
        py=center[:,None,1]+s*corners[None,:,0]+c*corners[None,:,1]
        return ((px>=-1.5+margin)&(px<=10.5-margin)&(py>=-2+margin)&(py<=2-margin)).all(-1)
    local=root-origins
    current=~inside(local,yaw)
    predicted=~inside(local+velocity*dt,yaw+yaw_rate*dt)
    if diagnostics is not None:
        diagnostics.update(current=current.detach().clone(),predicted=predicted.detach().clone(),
                           margin_m=margin,half_extents_m=list(half_extents))
    return current|predicted
