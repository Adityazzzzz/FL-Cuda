import torch
from collections import defaultdict

class SAM:
    """
    Sharpness-Aware Minimization (SAM) Optimizer.
    Used by FedBS to find flat, generalized minima instead of sharp, fragile valleys.
    """
    def __init__(self, optimizer, model, rho=0.1):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
    
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        # Calculate the overall gradient norm across all parameters
        for p in self.model.parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
            
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        
        # Take a step in the direction of steepest ascent (finding the sharpest loss)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            
            eps = self.state[p].get("eps")
            if eps is None: 
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            
            eps[...] = p.grad[...] 
            eps.mul_(self.rho / grad_norm) 
            p.add_(eps) 
            
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        # Step back to the original weights, then apply the descent step 
        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"]) 
            
        self.optimizer.step()
        self.optimizer.zero_grad()