from math import sqrt,log10
import numpy as np

#first metric
def PSNR(original, denoised): 
    mse = np.mean((original - denoised) ** 2) 
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr
  
#second metric

def mse(image1,image2):
  err=np.sum((image1.astype("float")-image2.astype("float"))**2)
  err/=float(image1.shape[0]*image1.shape[1])
  return err
#third metric

def compute_ssim(prediction, target, window_size=11):
  """Calculates structural similarity index between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  window_size : int
    Size of the Gaussian kernel used for computing SSIM
  """
  from metrics import pytorch_ssim

  if not isinstance(prediction, Variable):
    prediction = Variable(prediction, volatile=True)
  if not isinstance(target, Variable):
     target = Variable(target, volatile=True)

  ssim = pytorch_ssim.ssim(prediction, target, window_size=window_size).data[0]
  return ssim 
