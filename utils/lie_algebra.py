import torch
from liegroups.torch import SO3, SE3
from utils.lie_algebra_full import se3_log, se3_exp, se3_inv_left_jacobian, se3_left_jacobian, se3_adjoint

#See: Galleo et al. (2013) 'A compact formula for the derivative of a 3D Rotation...'
class so3_exp_fn(torch.autograd.Function):
    
    @staticmethod
    def forward(self, phi):
        angle = phi.norm()
        I = phi.new_empty((3,3))
        torch.nn.init.eye_(I)
      
        if angle < 1e-8:
            R = I + SO3.wedge(phi)
            self.save_for_backward(phi, R)
            return R

        axis = phi / angle
        s = torch.sin(angle)
        c = torch.cos(angle)

        outer_prod_axis = axis.view(3,1).mm(axis.view(1,3))
        R = c * I + (1. - c) * outer_prod_axis + s * SO3.wedge(axis)
        
        self.save_for_backward(phi, R)
        return R
    
    @staticmethod   
    def backward(self, grad_output):
        phi, R = self.saved_tensors
        grad = grad_output.new_empty((3,3,3))
        e_0 = grad_output.new_tensor([1,0,0]).view(3,1)
        e_1 = grad_output.new_tensor([0,1,0]).view(3,1)
        e_2 = grad_output.new_tensor([0,0,1]).view(3,1)
        I = grad_output.new_empty((3,3))
        torch.nn.init.eye_(I)
        
        if phi.norm() < 1e-8:
            grad[0,:,:] = SO3.wedge(e_0)
            grad[1,:,:] = SO3.wedge(e_1)
            grad[2,:,:] = SO3.wedge(e_2)
        else:
            
            fact = 1./(phi.norm()**2)
            phi_wedge = SO3.wedge(phi) 
            ImR = (I-R)
            
            grad[0,:,:] = fact*(phi[0]*phi_wedge + SO3.wedge(phi_wedge.mm(ImR.mm(e_0)))).mm(R)
            grad[1,:,:] = fact*(phi[1]*phi_wedge + SO3.wedge(phi_wedge.mm(ImR.mm(e_1)))).mm(R)
            grad[2,:,:] = fact*(phi[2]*phi_wedge + SO3.wedge(phi_wedge.mm(ImR.mm(e_2)))).mm(R)
        
        out = (grad_output*grad).sum((1,2)).view(3,1)
        
        return out

def so3_exp_with_deriv(phi):
    return so3_exp_fn.apply(phi)


def se3_log_exp(tau_a, tau_b):
    return se3_log_exp_fn.apply(tau_a, tau_b)


def se3_log_exp_mid(tau_a, tau_b):
    return se3_log_exp_mid_fn.apply(tau_a, tau_b)

#Implements log(exp(a)*exp(b)) with analytical derivatives wrt a and b (using middle perturbations)
#Go go gadget lie algebra


#Left perturbation
class se3_log_exp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tau_a, tau_b):
        tau_out = se3_log(se3_exp(tau_a).bmm(se3_exp(tau_b)))
        ctx.save_for_backward(tau_a, tau_b, tau_out)

        return tau_out
        
    @staticmethod
    def backward(ctx, grad_output):
        tau_a, tau_b, _ = ctx.saved_tensors
        adj_tau_a = se3_adjoint(se3_exp(tau_a))

        dtau_a = grad_output
        dtau_b = grad_output.view(-1,1,6).bmm(adj_tau_a)
        
        return dtau_a, dtau_b

#Middle perturbations
class se3_log_exp_mid_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tau_a, tau_b):
        tau_out = se3_log(se3_exp(tau_a).bmm(se3_exp(tau_b)))
        ctx.save_for_backward(tau_a, tau_b, tau_out)
        return tau_out.squeeze()
        
    @staticmethod
    def backward(ctx, grad_output):
        tau_a, tau_b, tau_out = ctx.saved_tensors
        inv_J_tau_out = se3_inv_left_jacobian(tau_out)
        adj_tau_a = se3_adjoint(se3_exp(tau_a))

        dtau_a = grad_output.view(-1,1,6).bmm(inv_J_tau_out.bmm(se3_left_jacobian(tau_a)))
        dtau_b = grad_output.view(-1,1,6).bmm(inv_J_tau_out.bmm(adj_tau_a).bmm(se3_left_jacobian(tau_b)))
        
        return dtau_a, dtau_b