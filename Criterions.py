import numpy as np
class Criterion :
    def __init__(self) -> None:
         pass
   
    def step_error(self,y,y_pred):
        pass
    def step_d_error(self,y,y_pred):
        pass
    
    
class MSE(Criterion):
    def __init__(self) -> None:
        super().__init__()
    def step_error(self,y,y_pred):
        step_error=(y-y_pred)**2
        return np.mean(step_error)
    def step_d_error(self,y,y_pred):
        step_d_error=-2*(y-y_pred)
        
        return np.mean(step_d_error)
    
class BinaryCrossEntropy(Criterion):
    def __init__(self) -> None:
        super().__init__()
    def step_error(self,y_true,y_proba):
        y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
        step_error=-(y_true*np.log(y_proba)+(1-y_true)*np.log(1-y_proba))
        return np.mean(step_error)
    def step_d_error(self, y_true,y_proba):
        y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
        step_d_error=-(y_true/y_proba - (1-y_true)/(1-y_proba))
        return np.mean(step_d_error)
    