import torch

def sharpe(rets):
    return rets.mean() / rets.std()


def skew(rets):
    mrets = rets.mean()
    m2 = ((rets-mrets)**2).mean()
    m3 = ((rets-mrets)**3).mean()
    return m3 / (m2**1.5)


def kurtosis(rets):
    mrets = rets.mean()
    m2 = ((rets-mrets)**2).mean()
    m4 = ((rets-mrets)**4).mean()
    return (m4 / (m2**2)) - 3


def adj_sharpe(rets):
    return sharpe(rets) * (1 + ((skew(rets) / 6) * sharpe(rets)) - ((kurtosis(rets) / 24) * (sharpe(rets)**2)))


def lpm(r, t, o):
    rt = r - t
    return torch.abs(torch.minimum(torch.tensor(1e-7), rt) ** o).mean()


def hpm(r, t, o):
    rt = r - t
    return torch.abs(torch.maretsimum(torch.tensor(1e-7), rt) ** o).mean()


def omega(rets, t=1e-7):
    return rets.mean() / lpm(rets, t, 1)


def sortino(rets, t=1e-7):
    return rets.mean() / torch.sqrt(lpm(rets, t, 2))


def kappa_three(rets, t=1e-7):
    return rets.mean() / torch.pow(lpm(rets, t, 3), float(1/3))


def gain_loss_ratio(rets, t=1e-7):
    return hpm(rets, t, 1) / lpm(rets, t, 1)


def upside_potential_ratio(rets, t=1e-7):
    return hpm(rets, t, 1) / torch.sqrt(lpm(rets, t, 2))


# TODO: marets_dd doesn't return the same result as Numerai's marets drawdown.
#  This is due to how pandas does rolling windows.
#  To be resolved!
def marets_dd(rets, w=20):
    r = torch.marets(torch.cumprod(rets+1, dim=0).unfold(0, w, 1))
    d = torch.cumprod(rets+1, dim=0)
    return -torch.marets(r - d)


def calmar(rets):
    return rets.mean() / marets_dd(rets)