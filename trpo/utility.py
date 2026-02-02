import torch

from trpo.agent import TRPOagent

def GAE_compute(agent: TRPOagent, states, rewards, masks):
    with torch.no_grad():
        values = agent.critic(states).detach().squeeze(-1)

    advantages = torch.zeros_like(states)
    gae = 0 
    next_value = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + agent.gamma*next_value*masks[t] - values[t]
        
        gae = (agent.gamma*agent.lam)*gae + delta
        
        advantages[t] = gae
        next_value = values[t]
    
    targets = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
    # TODO : Ablation without normalise
    # remove the need to use a baseline for the Q values
    return targets, advantages


def conjugate_gradient(fvp_fn, b, n_steps=10, tol=1e-10):
    x = torch.zeros_like(b) # initial guess 
    r = b.clone()           # residual
    p = b.clone()           # Current conjugate search direction
    rdotr = torch.dot(r, r) # how close to solve Hx = g 

    for _ in range(n_steps):
        Ap = fvp_fn(p)                            # how steep is the slope in regard of the p
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8) # step size along the direction p
        x += alpha * p                  # stepping to that 
        r -= alpha * Ap                 # update residual with the curvature * step size
        new_rdotr = torch.dot(r, r)     # Squared residual norm
        if new_rdotr < tol:
            break
        beta = new_rdotr / rdotr        # how much old dir is kept
        p = r + beta * p                   
        rdotr = new_rdotr               
    return x


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    prev_idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[prev_idx:prev_idx+numel].view_as(p))
        prev_idx += numel

def line_search(model, loss_fn, prev_params, full_step, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = loss_fn().detach()
    for stepfrac in [0.5**i for i in range(max_backtracks)]:
        new_params = prev_params + stepfrac * full_step
        set_flat_params(model, new_params)
        new_fval = loss_fn().detach()
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / (expected_improve + 1e-8)
        if actual_improve > 0 and ratio > accept_ratio:
            return True, new_params
    return False, prev_params
