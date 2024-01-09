import torch.optim as optim
from models import PokerNet
from play import get_policy_loss


def hard_osg(policy_b, n_epochs=20, n_sample=20, alpha=0.1):
    """Algorithm 2: Hard-OSG"""
    policy_o_hat = PokerNet()

    for _ in range(n_epochs):
        # eq.(7)
        policy_b.train()
        policy_o_hat.eval()
        optimizer_b = optim.Adam(policy_b.parameters(), lr=alpha)
        optimizer_b.zero_grad()
        loss_o_hat_b = get_policy_loss(
            policy_b, policy_o_hat, n_sample=n_sample
        )  # eq.(8)
        loss_o_hat_b.backward()
        optimizer_b.step()

        # eq.(9)
        policy_o_hat.train()
        policy_b.eval()
        optimizer_o_hat = optim.Adam(policy_o_hat.parameters(), lr=alpha)
        optimizer_o_hat.zero_grad()
        loss_b_o_hat = get_policy_loss(
            policy_o_hat, policy_b, n_sample=n_sample
        )  # eq.(10)
        loss_b_o_hat.backward()
        optimizer_o_hat.step()

    return policy_o_hat
