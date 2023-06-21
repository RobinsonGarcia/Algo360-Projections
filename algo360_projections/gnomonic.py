import numpy as np


from PIL import Image
import numpy as np
from scipy import ndimage
import math
import numpy as np  
Image.MAX_IMAGE_PIXELS = 933120000


class GnomonicProjector:
    def __init__(self,dims,scanner_shadow_angle):
        self.f_projection=None
        self.b_projection=None
        self.dims=dims
        self.scanner_shadow_angle=scanner_shadow_angle
        pass
    
    def point_forward(self,x,y,phi1,lamb0,fov):
        rho=np.sqrt(x**2+y**2)
        c=np.arctan2(rho,1)
        sinc=np.sin(c)
        cosc=np.cos(c)

        phi=np.arcsin(cosc*np.sin(phi1)+(y*sinc*np.cos(phi1)/rho))
        lamb=lamb0+np.arctan2(x*sinc,rho*np.cos(phi1)*cosc-y*np.sin(phi1)*sinc)
        
        phi=np.where(phi<-np.pi/2,np.pi/2-phi,phi)
        lamb=np.where(lamb<-np.pi,2*np.pi+lamb,lamb)

        phi=np.where(phi>np.pi/2,-np.pi/2+phi,phi)
        lamb=np.where(lamb>np.pi,-2*np.pi+lamb,lamb)
        
        return phi,lamb
    

    def forward(self,img,phi1,lamb0,fov=(1,1)):
        fov_h,fov_w = fov
        
        H,W=self.dims
        x,y=np.meshgrid(np.linspace(-1,1,W)*fov_w,np.linspace(-1,(90-self.scanner_shadow_angle)/90,H)*fov_h)
        
        phi,lamb=self.point_forward(x,y,phi1,lamb0,fov)
        
        mask = (phi>np.pi/3)&(phi<np.pi/2)
        phi=phi/(np.pi/2)
        lamb=lamb/np.pi

        HH,WW,C=img.shape

        phi=(0.5*(phi+1))*(HH-1)#*((180/(180-self.scanner_shadow_angle)))
        lamb=(0.5*(lamb+1))*(WW-1)

        

        o_img=[ndimage.map_coordinates(img[:,:,i], np.stack([phi,lamb]),order=0,prefilter=True,mode="grid-wrap") for i in range(C)]
        o_img=np.stack(o_img,axis=-1)
        #o_img[mask]=0


        self.f_projection=o_img
        self.phi1=phi1
        self.lamb0=lamb0
        self.fov=fov
        return o_img
    
    def point_backward(self,phi,lamb,phi1,lamb0,fov):
        fov_h,fov_w = fov
        cosc=np.sin(phi1)*np.sin(phi)+np.cos(phi1)*np.cos(phi)*np.cos(lamb-lamb0)

        K=1/cosc
        x=K*np.cos(phi)*np.sin(lamb-lamb0)/fov_w
        y=K*(np.cos(phi1)*np.sin(phi)-np.sin(phi1)*np.cos(phi)*np.cos(lamb-lamb0))/fov_h

        x=0.5*(x+1)
        y=0.5*(y+1)

        HH, WW = self.dims
        
        x=x*(WW-1)
        y=y*(HH-1)
        self.cosc=cosc
        return x,y
    
    def backward(self,face,img,phi1=None,lamb0=None,fov=None):
        HH, WW = self.dims
        H = int(2*HH*(180-self.scanner_shadow_angle)/180)
        W = int(WW*4)
        H,W,_=img.shape
  
        #u,v=np.meshgrid(np.linspace(-1,1,W),np.linspace(-np.pi/2,np.pi*(90-self.scanner_shadow_angle)/180,H))

        u,v=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))

        phi=v*(np.pi/2)
        lamb=u*np.pi
        
       
        x,y=self.point_backward(phi,lamb,phi1,lamb0,fov)

        coords = np.stack([x,y])


        oo=[ndimage.map_coordinates(face.T[i,:,:], coords,order=3,prefilter=True)*(self.cosc>=0) for i in range(3)]
        
        oo=np.stack(oo,axis=-1)
        self.b_projection=oo
        return oo


        