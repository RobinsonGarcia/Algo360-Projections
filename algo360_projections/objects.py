import numpy
from numbers import Number
from algo360_projections.gnomonic import GnomonicProjector
from algo360_projections.misc import *
from PIL import Image
import torch
# https://numpy.org/doc/stable/user/basics.dispatch.html

class Base(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __array__(self, dtype=None):
        return self._i
    
    def numpy(self):
        return self.__array__()
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    
        if method == '__call__':
            N = None
            scalars = []
            for input in inputs:
                # In this case we accept only scalar numbers or DiagonalArrays.
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(input._i)

                else:
                    return NotImplemented
            return ufunc(*scalars, **kwargs)#self.__class__(ufunc(*scalars, **kwargs))
        else:
            return NotImplemented
        
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs) 
    


        
import numpy
from numbers import Number

class EquirectProjection(Base):
    def from_pil(img):
        return EquirectProjection(np.array(img))
    
    def from_file(file):
        return EquirectProjection.from_pil(Image.open(file))
        
    def __init__(self,eq_img,**kwargs):
        self.eq_img = eq_img
        H,W,_ = eq_img.shape
        self._H = H
        self._W = W
        self._i = eq_img
        self.scanner_shadow_angle=0
        self.projector = GnomonicProjector((H//2,W//4),self.scanner_shadow_angle)
        
    def generate(self):
        
        N = self.angles.shape[1]
        
        a_phi = self.angles[0]
        a_theta = self.angles[1]

        imgs=[]
        for ix in range(N):
            f_img = self.projector.forward(self.eq_img,a_phi[ix]-np.pi/2,a_theta[ix])
            imgs.append(f_img)
        
        return imgs
        
    def cube(self):
        self.points,self.angles = cubemap(n=None,delta_lamb=0,delta_phi=0,scanner_shadow_angle=self.scanner_shadow_angle)
        faces = self.generate()
        cube_kwargs={'mode':'cube','eq_H':self._H,'eq_W':self._W}
        return Faces(np.stack(faces),**cube_kwargs)


    def ico(self,n=1,random_samples=None):
        self.points,self.angles = ico(n,delta_lamb=0,delta_phi=0,scanner_shadow_angle=self.scanner_shadow_angle)
        N = self.angles.shape[1]
        if isinstance(random_samples,int):
            
            ixs = np.random.choice(np.arange(N),random_samples,replace=False)
            self.angles = self.angles[:,ixs]
            self.points = self.points[:,ixs]
        else:
            ixs = np.arange(N)
        faces = self.generate()
        ico_kwargs={'mode':'ico','n':n,'ixs':ixs,'eq_H':self._H,'eq_W':self._W}
        return Faces(np.stack(faces),**ico_kwargs)


    def __repr__(self):
        return f"{self.__class__.__name__}(H={self._H}, W={self._W}, arr={self._i})"
    def torch(self):
        return torch.tensor(self._i).permute(0,1,2)

class Faces(Base):
    def __init__(self,faces,**kwargs):
        super().__init__()
        
        N,H,W,C = faces.shape
        
        self._H=H
        self._W=W
        self._N=N
        
        self.HH,self.WW=(int(H*2),int(W*4))
        
        
        self._i = faces
        self.scanner_shadow_angle = 0
        self.projector = GnomonicProjector((self._H,self._W),self.scanner_shadow_angle)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def generate(self):
        
        N = self.angles.shape[1]
        
        a_phi = self.angles[0]
        a_theta = self.angles[1]

        img = np.zeros((self.eq_H,self.eq_W,3))
        imgs=[]
        for ix in range(N):
            b_img = self.projector.backward(self._i[ix],img,a_phi[ix]-np.pi/2,a_theta[ix],fov=(1,1))
            imgs.append(b_img)
        
        return imgs
        
    def cube(self):
        self.points,self.angles = cubemap(n=None,delta_lamb=0,delta_phi=0,scanner_shadow_angle=self.scanner_shadow_angle)
        eq_img = self.generate()
        return np.max(np.stack(eq_img),axis=0)


    def ico(self):
        n = self.n
        ixs = self.ixs
        self.points,self.angles = ico(n,delta_lamb=0,delta_phi=0,scanner_shadow_angle=self.scanner_shadow_angle)
        self.angles = self.angles[:,ixs]
        self.points = self.points[:,ixs]
        eq_img = self.generate()
        return np.max(np.stack(eq_img),axis=0)
    
    def eq(self):
        if self.mode=='cube':
            return self.cube()
        elif self.mode=='ico':
            return self.ico()
            
        else:
            raise

    def __repr__(self):
        return f"{self.__class__.__name__}(H={self._H}, W={self._W}, arr={self._i})"
    
    def __getitem__(self, item):
         return self._i[item]
    def torch(self):
        return torch.tensor(self._i).permute(0,3,1,2)