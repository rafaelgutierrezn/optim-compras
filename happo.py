"""
HAPPO demo — optimización de compras multi-agente.
Un agente por proveedor. Objetivo: aprender a pedir minimizando costes.

HAPPO vs PPO: los agentes se actualizan en orden aleatorio y cada uno
usa el ratio acumulado de los agentes ya actualizados antes que él.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from data_loader import load_instance
from model import build_and_solve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hiperparámetros ──────────────────────────────────────────────────────────
EPISODES    = 3000
BATCH_EPS   = 20     # episodios por actualización (batch más grande = aprendizaje estable)
LR          = 3e-4
CLIP        = 0.2    # epsilon PPO
GAMMA       = 0.99
PENALTY     = 5.0    # coste por unidad de demanda no cubierta
MAX_Q       = 60.0   # cota superior de pedido por producto
REWARD_SCALE = 150.0 # normaliza recompensas al rango [-1, 0]


# ── Entorno ───────────────────────────────────────────────────────────────────
class Env:
    def __init__(self, data):
        self.d     = data
        self.T     = len(data["periodos"])
        self.prods = data["productos"]
        self.provs = data["proveedores"]
        self.ages  = {i: list(range(data["vida_util"][i] + 1)) for i in self.prods}

        inv_size     = sum(len(self.ages[i]) for i in self.prods)
        self.obs_dim = inv_size + len(self.prods) + 1          # inv + demanda + t/T
        self.act_dim = {                                        # acción heterogénea
            j: len([i for i in self.prods if (i, j) in data["compatibilidad"]])
            for j in self.provs
        }

    def _obs(self):
        inv = [self.inv.get((i, a), 0.0) / MAX_Q for i in self.prods for a in self.ages[i]]
        dem = [self.d["demanda"].get((i, self.t), 0.0) / MAX_Q for i in self.prods]
        return np.array(inv + dem + [self.t / self.T], dtype=np.float32)

    def reset(self):
        self.t = 1
        self.total_cost = 0.0
        self.inv = {}
        for (i, a), qty in self.d["stock_inicial"].items():
            self.inv[(i, a + 1)] = qty          # stock inicial envejece 1 antes de t=1
        return self._obs()

    def step(self, actions):
        d = self.d

        # 1. Órdenes → coste compra + coste fijo
        orders = {i: 0.0 for i in self.prods}
        cost_buy = cost_fix = 0.0
        for j in self.provs:
            prods_j = [i for i in self.prods if (i, j) in d["compatibilidad"]]
            total_j = 0.0
            for k, i in enumerate(prods_j):
                qty = float(np.clip(actions[j][k], 0, MAX_Q))
                orders[i] += qty
                cost_buy  += d["precio"][i, j] * qty
                total_j   += qty
            if total_j > 0.01:
                cost_fix += d["costo_fijo"][j]

        for i in self.prods:
            self.inv[(i, 0)] = self.inv.get((i, 0), 0.0) + orders[i]

        # 2. Coste de mantenimiento (inventario actual, incluidas las compras)
        cost_hold = sum(d["costo_mant"][i] * self.inv.get((i, a), 0.0)
                        for i in self.prods for a in self.ages[i])

        # 3. Cubrir demanda FIFO (edad mayor primero = más urgente)
        penalty = 0.0
        for i in self.prods:
            rem = d["demanda"].get((i, self.t), 0.0)
            for a in range(d["vida_util"][i], -1, -1):
                consume = min(self.inv.get((i, a), 0.0), rem)
                self.inv[(i, a)] = self.inv.get((i, a), 0.0) - consume
                rem -= consume
                if rem < 1e-6:
                    break
            penalty += rem * PENALTY

        # 4. Merma + envejecimiento
        cost_waste = 0.0
        new_inv = {}
        for i in self.prods:
            Li = d["vida_util"][i]
            cost_waste += d["costo_merma"][i] * self.inv.get((i, Li), 0.0)
            for a in range(1, Li + 1):
                new_inv[(i, a)] = self.inv.get((i, a - 1), 0.0)
            new_inv[(i, 0)] = 0.0
        self.inv = new_inv

        cost = cost_buy + cost_fix + cost_hold + cost_waste + penalty
        self.total_cost += cost
        self.t += 1
        return self._obs(), -cost / REWARD_SCALE, self.t > self.T


# ── Red Actor-Crítico (MLP compartida) ───────────────────────────────────────
class AC(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net    = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 64),      nn.Tanh())
        self.mean   = nn.Linear(64, act_dim)
        self.logstd = nn.Parameter(torch.ones(act_dim) * 2.0)  # std inicial ≈ 7.4
        self.value  = nn.Linear(64, 1)

    def forward(self, x):
        h    = self.net(x)
        mean = torch.relu(self.mean(h))            # empieza cerca de 0, crece con la demanda
        std  = self.logstd.exp().clamp(0.5, 10.0)  # std grande al inicio = más exploración
        return mean, std, self.value(h).squeeze(-1)

    def act(self, obs):
        mean, std, v = self.forward(obs)
        dist = Normal(mean, std)
        a    = dist.sample().clamp(0, MAX_Q)
        return a, dist.log_prob(a).sum(-1), v

    def logp_and_val(self, obs, act):
        mean, std, v = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(act).sum(-1), v, dist.entropy().sum(-1)


# ── Actualización HAPPO secuencial ───────────────────────────────────────────
def happo_update(nets, opts, bufs, order):
    """
    Actualiza los agentes en 'order' (permutación aleatoria).
    M = ratio acumulado de los agentes ya actualizados antes del agente actual.
    """
    M = None

    for j in order:
        obs      = torch.FloatTensor(np.array(bufs[j]["obs"])).to(DEVICE)
        act      = torch.FloatTensor(np.array(bufs[j]["act"])).to(DEVICE)
        logp_old = torch.FloatTensor(bufs[j]["logp"]).to(DEVICE)
        ret      = torch.FloatTensor(bufs[j]["ret"]).to(DEVICE)
        adv      = (ret - ret.mean()) / (ret.std() + 1e-8)

        logp_new, val, ent = nets[j].logp_and_val(obs, act)
        ratio = (logp_new - logp_old).exp()

        # Clave HAPPO: ponderar por el ratio acumulado de agentes anteriores
        r = ratio * M.detach() if M is not None else ratio

        loss = ( -torch.min(r * adv, torch.clamp(r, 1 - CLIP, 1 + CLIP) * adv).mean()
                 + 0.5 * (ret - val).pow(2).mean()
                 - 0.01 * ent.mean() )

        opts[j].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(nets[j].parameters(), 0.5)
        opts[j].step()

        # Actualizar M para el siguiente agente en el orden
        with torch.no_grad():
            lp2, _, _ = nets[j].logp_and_val(obs, act)
            r_after   = (lp2 - logp_old).exp()
        M = r_after if M is None else M * r_after


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train(data, episodes=EPISODES):
    env  = Env(data)
    nets = {j: AC(env.obs_dim, env.act_dim[j]).to(DEVICE) for j in data["proveedores"]}
    opts = {j: torch.optim.Adam(nets[j].parameters(), lr=LR) for j in data["proveedores"]}
    best = float("inf")
    print(f"  Dispositivo: {DEVICE}")

    # Acumulador de batch (BATCH_EPS episodios antes de actualizar)
    batch = {j: {"obs": [], "act": [], "logp": [], "ret": []} for j in data["proveedores"]}

    for ep in range(1, episodes + 1):
        obs  = env.reset()
        bufs = {j: {"obs": [], "act": [], "logp": [], "rew": []} for j in data["proveedores"]}
        done = False

        while not done:
            o = torch.FloatTensor(obs).to(DEVICE)
            actions = {}
            with torch.no_grad():
                for j in data["proveedores"]:
                    a, lp, _ = nets[j].act(o)
                    actions[j] = a.cpu().numpy()
                    bufs[j]["obs"].append(obs)
                    bufs[j]["act"].append(actions[j])
                    bufs[j]["logp"].append(lp.item())

            obs, rew, done = env.step(actions)

            for j in data["proveedores"]:
                bufs[j]["rew"].append(rew)

        # Retornos descontados + acumular en batch
        for j in data["proveedores"]:
            R, ret = 0.0, []
            for r in reversed(bufs[j]["rew"]):
                R = r + GAMMA * R
                ret.insert(0, R)
            batch[j]["obs"].extend(bufs[j]["obs"])
            batch[j]["act"].extend(bufs[j]["act"])
            batch[j]["logp"].extend(bufs[j]["logp"])
            batch[j]["ret"].extend(ret)

        # Actualizar cada BATCH_EPS episodios
        if ep % BATCH_EPS == 0:
            order = list(np.random.permutation(data["proveedores"]))
            happo_update(nets, opts, batch, order)
            batch = {j: {"obs": [], "act": [], "logp": [], "ret": []} for j in data["proveedores"]}

        best = min(best, env.total_cost)
        if ep % 200 == 0:
            print(f"Ep {ep:4d} | Coste: {env.total_cost:8.2f} | Mejor: {best:.2f}")

    return best


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_instance("data/instances/demo_small")

    milp = build_and_solve(data)
    print(f"Referencia MILP (optimo): {milp['costo_total']:.2f}\n")

    print("Entrenando HAPPO...")
    best = train(data)

    print(f"\nMejor coste HAPPO : {best:.2f}")
    print(f"Optimo MILP       : {milp['costo_total']:.2f}")
    print(f"Gap               : {(best - milp['costo_total']) / milp['costo_total'] * 100:.1f}%")
