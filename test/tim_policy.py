import numpy as np
import os
from ray.rllib.policy.policy import Policy
import torch
from torch import nn
from torch.distributions import Categorical
import torch.functional as F
import warnings
from abc import ABC, abstractmethod

# Need an added import for competition submission?
# Post an issue to the github and we will work to get it added into the system!

# NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry


def T(a, device="cpu", dtype=torch.float32, debug=False):
    if debug:
        print("T: ", a)
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a).to(device)
    elif not torch.is_tensor(a):
        return torch.from_numpy(np.array(a), dtype=dtype).to(device)
    elif a.device != device:
        return a.to(device)
    else:
        return a.to(device)


def get_multi_discrete_one_hot(x, discrete_action_dims, debug=False):
    onehot = torch.zeros((x.shape[0], sum(discrete_action_dims)), device=x.device)
    start = 0
    for i, dim in enumerate(discrete_action_dims):
        onehot[torch.arange(x.shape[0]), x[:, i].long() + start] = 1
        start += dim
    if debug:
        print(f"get_multi_discrete_one_hot: {x}, {discrete_action_dims}, {onehot}")
        # input()
    return onehot


def normgrad(parameters, grad_clip=0.5):
    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)


def _orthogonal_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(ABC):

    @abstractmethod
    def train_action(self, observations, action_mask=None, step=False):
        # discrete actions, continuous actions, discrete
        # log probs, continuous log probs, value for a single observation
        return 0, 0, 0, 0, 0

    @abstractmethod
    def ego_actions(self, observations, action_mask=None):
        return 0

    @abstractmethod
    def imitation_learn(self, observations, actions):
        return 0, 0  # actor loss, critic loss

    @abstractmethod
    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    @abstractmethod
    def expected_V(self, obs, legal_action):
        print("expected_V not implemeted")
        return 0

    # performs reinforcement learning update on a batch of data
    # may be critic only if doing offline RL or an actor that is delayed
    @abstractmethod
    def reinforcement_learn(self, batch, agent_num=0, critic_only=False, debug=False):
        return 0, 0  # actor loss, critic loss

    @abstractmethod
    def save(self, checkpoint_path):
        print("Save not implemeted")

    @abstractmethod
    def load(self, checkpoint_path):
        print("Load not implemented")


class ffEncoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dims,
        activation="relu",
        device="cpu",
        orthogonal_init=False,
        dropout=0.6,
    ):
        super(ffEncoder, self).__init__()
        activations = {
            "relu": torch.nn.functional.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "none": lambda x: x,
        }
        assert activation in activations, "Invalid activation function"
        self.activation = activations[activation]
        self.drop = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList()
        # print(obs_dim, hidden_dims)
        for i in range(len(hidden_dims)):
            if i == 0:
                self.encoder.append(nn.Linear(obs_dim, hidden_dims[i]))
            else:
                self.encoder.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if orthogonal_init:
                _orthogonal_init(self.encoder[-1])
        self.float()
        self.to(device)
        self.device = device
        # self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x, debug=False):
        if debug:
            print(f"ffEncoder: x {x}")
        x = T(x, self.device).float()
        if debug:
            print(f"ffEncoder after T: x {x}")
        if debug:
            interlist = []
            interlist.append(x)
        for layer in self.encoder:
            if layer == self.encoder[0] and self.drop > 0:
                x = self.activation(self.dropout(layer(x)))
            else:
                x = self.activation(layer(x))
            if debug:
                interlist.append(x)
        # if x contains nan, print the intermediate list and encoder weights
        if torch.isnan(x).any():
            if debug:
                print(f"Intermediate list: {interlist}")
            for layer in self.encoder:
                print(f"Layer {layer.weight}")
        return x


class QS(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[2],
        hidden_dims=[64, 64],
        encoder=None,
        activation="relu",
        orthogonal=False,
        dropout=0.0,
        dueling=False,
        device="cpu",
        n_c_action_bins=11,
        head_hidden_dim=64,  # adding a layer size gives hidden layers to heads
    ):
        super(QS, self).__init__()

        # guard code
        if head_hidden_dim is None:
            head_hidden_dim = 0
        if continuous_action_dim is None:
            continuous_action_dim = 0
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ffEncoder(
                obs_dim, hidden_dims, activation, device, orthogonal, dropout
            )
        if discrete_action_dims is not None:
            if isinstance(discrete_action_dims, int):
                discrete_action_dims = [discrete_action_dims]
            if len(discrete_action_dims) == 0:
                ValueError(
                    "discrete_action_dims should not be empty, use [x] for a single discrete action with cardonality 'x'"
                )
            if min(discrete_action_dims) < 1:
                ValueError(
                    "discrete_action_dims should not contain values less than 1, use [x] for a single discrete action with cardonality 'x'"
                )
        # setting needed self variables
        self.disc_action_dims = discrete_action_dims
        self.cont_action_dims = continuous_action_dim
        self.device = device
        self.dueling = dueling
        self.last_hidden_dim = head_hidden_dim
        if head_hidden_dim == 0:
            self.last_hidden_dim = hidden_dims[-1]
        self.joint_heads_hidden_layer = None
        self.advantage_heads = None
        self.tot_adv_size = continuous_action_dim * n_c_action_bins
        if discrete_action_dims is not None:
            self.tot_adv_size += sum(discrete_action_dims)

        # set up hidden layer for the adv and V heads
        if head_hidden_dim != 0:
            self.joint_heads_hidden_layer = nn.Linear(
                hidden_dims[-1], head_hidden_dim * (2 if self.dueling else 1)
            )
            if orthogonal:
                _orthogonal_init(self.joint_heads_hidden_layer)
        # set up the adv heads if this isn't just a V network
        if self.cont_action_dims > 0 or self.disc_action_dims is not None:
            self.advantage_heads = nn.Linear(
                self.last_hidden_dim,
                self.tot_adv_size,
            )

        # set up the value head if dueling is True
        self.value_head = nn.Linear(self.last_hidden_dim, 1) if self.dueling else None
        # print(
        #    f"initialized QS with: {self.joint_heads_hidden_layer}, {self.value_head}, {self.advantage_heads}\n  d_dim: {discrete_action_dims}, c_dim: {continuous_action_dim}, h_dim: {hidden_dims}, head_hidden_dim: {head_hidden_dim}"
        # )
        self.to(device)

    def forward(self, x, action_mask=None):
        # TODO: action mask implementation
        x = T(x, self.device)
        x = self.encoder(x)
        values = 0

        # If the heads have their own hidden layer for a 2 layer dueling network
        if self.joint_heads_hidden_layer is not None:
            x = torch.nn.functional.relu(self.joint_heads_hidden_layer(x))
        if self.dueling:
            values = self.value_head(x[:, : self.last_hidden_dim])

        if self.advantage_heads is not None:
            advantages = self.advantage_heads(x[:, -self.last_hidden_dim :])

        tot_disc_dims = 0
        disc_advantages = None
        cont_advantages = None
        if self.disc_action_dims is not None:
            tot_disc_dims = sum(self.disc_action_dims)
            disc_advantages = []
            start = 0
            for i, dim in enumerate(self.disc_action_dims):
                end = start + dim
                disc_advantages.append(advantages[:, start:end])
                if (
                    self.dueling
                ):  # These are mean zero when dueling or Q values when not
                    disc_advantages[-1] = disc_advantages[-1] - disc_advantages[
                        -1
                    ].mean(dim=-1, keepdim=True)
                start = end

        if self.cont_action_dims > 0:
            cont_advantages = (
                advantages[:, tot_disc_dims:]
                .view(advantages.shape[0], self.cont_action_dims, -1)
                .transpose(0, 1)  # TODO: figure out if it is worth it to transpose
            )  # transposed because then discrete and continuous output same dim order
            if self.dueling:  # These are mean zero when dueling or Q values when not
                cont_advantages = cont_advantages - cont_advantages.mean(
                    dim=-1, keepdim=True
                )

        return values, disc_advantages, cont_advantages


class DQN(nn.Module, Agent):
    def __init__(
        self,
        obs_dim=10,
        discrete_action_dims=None,  # np.array([2]),
        continuous_action_dims: int = None,  # 2,
        min_actions=None,  # np.array([-1,-1]),
        max_actions=None,  # ,np.array([1,1]),
        hidden_dims=[64, 64],  # first is obs dim if encoder provded
        head_hidden_dim=0,  # if None then no head hidden layer
        gamma=0.99,
        lr=3e-5,
        imitation_lr=1e-5,
        dueling=False,
        n_c_action_bins=10,
        munchausen=0,  # turns it into munchausen dqn
        entropy=0,  # turns it into soft-dqn
        activation="relu",
        orthogonal=False,
        init_eps=0.9,
        eps_decay_half_life=10000,
        device="cpu",
        eval_mode=False,
        name="DQN",
        clip_grad=1.0,
        load_from_checkpoint_path=None,
        encoder=None,
        conservative=False,
        imitation_type="cross_entropy",  # or "reward"
    ):
        super(DQN, self).__init__()
        self.clip_grad = clip_grad
        if load_from_checkpoint_path is not None:
            self.load(load_from_checkpoint_path)
            return
        self.eval_mode = eval_mode
        self.imitation_type = imitation_type
        self.entropy_loss_coef = entropy  # use soft Q learning entropy loss or not H(Q)
        self.dqn_type = "EGreedy"
        if self.entropy_loss_coef > 0:
            self.dqn_type = "Soft"
        if self.entropy_loss_coef > 0 and munchausen > 0:
            self.dqn_type = "Munchausen"

        self.obs_dim = obs_dim  # size of observation
        self.discrete_action_dims = discrete_action_dims
        self.imitation_lr = imitation_lr
        # cardonality for each discrete action

        self.continuous_action_dims = continuous_action_dims
        # number of continuous actions

        self.name = name
        self.min_actions = min_actions  # min continuous action value
        self.max_actions = max_actions  # max continuous action value
        if max_actions is not None:
            self.np_action_ranges = self.max_actions - self.min_actions
            self.action_ranges = torch.from_numpy(self.np_action_ranges).to(device)
            self.np_action_means = (self.max_actions + self.min_actions) / 2
            self.action_means = torch.from_numpy(self.np_action_means).to(device)
        self.gamma = gamma
        self.lr = lr
        self.dueling = (
            dueling  # whether or not to learn True: V+Adv = Q or False: Adv = Q
        )
        self.n_c_action_bins = n_c_action_bins  # number of discrete action bins to discretize continuous actions
        self.munchausen = munchausen  # munchausen amount
        self.twin = False  # min(double q) to reduce bias
        self.init_eps = init_eps  # starting eps_greedy epsilon
        self.eps = self.init_eps
        self.eps_decay_half_life = (
            eps_decay_half_life  # eps cut in half every 'half_life' frames
        )
        self.step = 0
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.orthogonal = orthogonal
        self.Q1 = QS(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dims,
            discrete_action_dims=discrete_action_dims,
            hidden_dims=hidden_dims,
            activation=activation,
            orthogonal=orthogonal,
            dueling=dueling,
            n_c_action_bins=n_c_action_bins,
            device=device,
            encoder=encoder,  # pass encoder if using one for observations (like in visual DQN)
            head_hidden_dim=head_hidden_dim,  # if None then no head hidden layer
        )

        self.Q1.to(device)

        self.conservative = conservative
        self.device = device
        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr)
        self.to(device)

        # These can be saved to remake the same DQN
        # TODO: check that this is suffucuent
        self.attrs = [
            "step",
            "entropy_loss_coef",
            "munchausen",
            "discrete_action_dims",
            "continuous_action_dims",
            "min_actions",
            "max_actions",
            "gamma",
            "lr",
            "dueling",
            "n_c_action_bins",
            "init_eps",
            "eps_decay_half_life",
            "device",
            "eval_mode",
            "hidden_dims",
            "activation",
        ]

    def _cont_from_q(self, cont_act):
        return (
            torch.argmax(cont_act, dim=-1).squeeze(-1) / (self.n_c_action_bins - 1)
            - 0.5
        ) * self.action_ranges + self.action_means

    def _cont_from_soft_q(self, cont_act):
        return (
            Categorical(logits=cont_act).sample() / (self.n_c_action_bins - 1) - 0.5
        ) * self.action_ranges + self.action_means

    def _discretize_actions(self, continuous_actions):
        # print(continuous_actions.shape)
        return torch.clamp(  # inverse of _cont_from_q
            torch.round(
                ((continuous_actions - self.action_means) / self.action_ranges + 0.5)
                * (self.n_c_action_bins - 1)
            ).to(torch.int64),
            0,
            self.n_c_action_bins - 1,
        )

    def _e_greedy_train_action(
        self, observations, action_mask=None, step=False, debug=False
    ):
        disc_act, cont_act = None, None
        if self.init_eps > 0.0:
            self.eps = self.init_eps * (
                1 - self.step / (self.step + self.eps_decay_half_life)
            )
        value = 0
        if self.init_eps > 0.0 and np.random.rand() < self.eps:
            # print("  Taking random action")
            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                disc_act = np.zeros(
                    shape=len(self.discrete_action_dims), dtype=np.int32
                )
                for i in range(len(self.discrete_action_dims)):
                    disc_act[i] = np.random.randint(0, self.discrete_action_dims[i])

            if self.continuous_action_dims > 0:
                cont_act = (
                    np.random.rand(self.continuous_action_dims) - 0.5
                ) * self.np_action_ranges + self.np_action_means
            # print(disc_act)
        else:
            # print("  Taking greedy action")
            with torch.no_grad():
                # print("Getting value from Q1 for soft action selection")
                value, disc_act, cont_act = self.Q1(observations, action_mask)
                # print("done with that")
                # select actions from q function
                # print(value, disc_act, cont_act)
                if (
                    self.discrete_action_dims is not None
                    and len(self.discrete_action_dims) > 0
                ):
                    d_act = np.zeros(len(disc_act), dtype=np.int32)
                    for i, da in enumerate(disc_act):
                        d_act[i] = torch.argmax(da).detach().cpu().item()
                    disc_act = d_act
                if self.continuous_action_dims > 0:
                    if debug:
                        print(
                            f"  cont act {cont_act}, argmax: {torch.argmax(cont_act,dim=-1).detach().cpu()}"
                        )
                        print(
                            f"  Trying to store this in actions {((torch.argmax(cont_act,dim=-1)/ (self.n_c_action_bins - 1) -0.5)* self.action_ranges+ self.action_means)} calculated from da: {cont_act} with ranges: {self.action_ranges} and means: {self.action_means}"
                        )
                    cont_act = self._cont_from_q(cont_act).cpu().numpy()
        return disc_act, cont_act

    def _soft_train_action(self, observations, action_mask, step, debug):
        disc_act, cont_act = None, None
        with torch.no_grad():

            value, disc_act, cont_act = self.Q1(observations, action_mask)
            # print("Done with that")
            if len(self.discrete_action_dims) > 0:
                dact = np.zeros(len(disc_act), dtype=np.int64)
                for i, da in enumerate(disc_act):
                    dact[i] = Categorical(logits=da).sample().cpu().item()
                disc_act = dact  # had to store da temporarily to keep using disc_act
            if self.continuous_action_dims > 0:
                if debug:
                    print(
                        f"  cont act {cont_act}, argmax: {torch.argmax(cont_act,dim=-1).detach().cpu()}"
                    )
                    print(
                        f"  Trying to store this in actions {((torch.argmax(cont_act,dim=-1)/ (self.n_c_action_bins - 1) -0.5)* self.action_ranges+ self.action_means)} calculated from da: {cont_act} with ranges: {self.action_ranges} and means: {self.action_means}"
                    )
                cont_act = self._cont_from_soft_q(cont_act).cpu().numpy()
        return disc_act, cont_act

    def train_action(self, observations, action_mask=None, step=False, debug=False):
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            # print(f"  train_action expanding dims to {observations.shape}")
        disc_act, cont_act = self._e_greedy_train_action(
            observations, action_mask, step, debug
        )
        self.step += int(step)
        return disc_act, cont_act, 0, 0, 0

    def ego_actions(self, observations, action_mask=None):
        return 0

    def _bc_cross_entropy_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        discrete_loss = 0
        continuous_loss = 0
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            for i in range(len(self.discrete_action_dims)):
                discrete_loss += nn.CrossEntropyLoss()(
                    disc_adv[i], disc_act[:, i]
                )  # for discrete action 1

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            continuous_actions = self._discretize_actions(cont_act)
            # print(continuous_actions)
            for i in range(self.continuous_action_dims):
                continuous_loss += nn.CrossEntropyLoss()(
                    cont_adv[i], continuous_actions[:, i]
                )

        return discrete_loss, continuous_loss

    def _reward_imitation_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        discrete_loss = 0
        continuous_loss = 0
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            for i in range(len(self.discrete_action_dims)):
                best_q, best_a = torch.max(disc_adv[i], -1)
                mask = best_a != disc_act[:, i]
                discrete_loss += nn.MSELoss(reduction="none")(
                    best_q + mask, best_q.detach()
                ).mean()

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            continuous_actions = self._discretize_actions(cont_act)
            # print(continuous_actions)
            for i in range(len(self.continuous_action_dims)):
                best_q, best_a = torch.max(cont_adv[i], -1)
                mask = best_a != continuous_actions[:, i]
                continuous_loss += nn.MSELoss(reduction="none")(
                    best_q + mask, best_q.detach()
                ).mean()
        return discrete_loss, continuous_loss

    def imitation_learn(self, observations, continuous_actions, discrete_actions):
        values, disc_adv, cont_adv = self.Q1(observations)
        if self.eval_mode:
            return 0, 0
        else:
            dloss, closs = 0, 0
            if self.imitation_type == "cross_entropy":
                dloss, closs = self._bc_cross_entropy_loss(
                    disc_adv, cont_adv, discrete_actions, continuous_actions
                )
            else:
                dloss, closs = self._reward_imitation_loss()
            loss = dloss + closs
            if loss == 0:
                warnings.warn(
                    "Loss is 0, not updating. Most likely due to continuous and discrete actions being None,0 respectively"
                )
                return 0, 0
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.clip_grad,
                    error_if_nonfinite=True,
                )
            self.optimizer.step()
            if dloss != 0:
                dloss = dloss.item()
            if closs != 0:
                closs = closs.item()
            return dloss, closs

    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action=None, debug=False):
        with torch.no_grad():
            value, dac, cac = self.Q1(obs, legal_action)
            if debug:
                print(f"value: {value}, dac: {dac}, cac: {cac}, eps: {self.eps}")
            if self.dueling:
                return value  # TODO make sure this doesnt need to be item()

            dq = 0
            n = 0
            if len(self.discrete_action_dims) > 0:
                n += 1
                for hi, h in enumerate(dac):
                    a = torch.argmax(h, dim=-1)
                    bestq = h[a].item()
                    h[a] = 0
                    if legal_action is not None:
                        if torch.sum(legal_action[hi]) == 1:
                            otherq = (
                                bestq  # no other choices so 100% * only legal choice
                            )
                        else:
                            otherq = torch.sum(  # average of other choices
                                h * legal_action[hi], dim=-1
                            ) / (torch.sum(legal_action[hi], dim=-1) - 1)
                    else:
                        otherq = torch.sum(h, dim=-1) / (
                            self.discrete_action_dims[hi] - 1
                        )
                        if debug:
                            print(
                                f"{otherq} = self.eps * {torch.sum(h, dim=-1)} / ({self.discrete_action_dims[hi] - 1})"
                            )

                    qmean = (1 - self.eps) * bestq + self.eps * otherq
                    if debug:
                        print(
                            f"dq: {qmean} = {(1 - self.eps)} * {bestq} + {self.eps} * {otherq}"
                        )
                    dq += qmean
                dq = dq / len(self.discrete_action_dims)
            cq = 0
            if self.continuous_action_dims > 0:
                n += 1
                for h in cac:
                    a = torch.argmax(h, dim=-1)

                    bestq = h[a].item()
                    h[a] = 0
                    otherq = torch.sum(h, dim=-1) / (self.n_c_action_bins - 1)

                    if debug:
                        print(
                            f"cq: {(1 - self.eps) * bestq + self.eps * otherq} = {(1 - self.eps)} * {bestq} + {self.eps} * ({otherq})"
                        )
                    cq += (1 - self.eps) * bestq + self.eps * otherq
                cq = cq / self.continuous_action_dims

            return value + (cq + dq) / (max(n, 1))

    def cql_loss(self, disc_adv, cont_adv, disc_act, cont_act, debug=False):
        """Computes the CQL loss for a batch of Q-values and actions."""

        cql_loss = 0
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            for i in range(len(self.discrete_action_dims)):
                if debug:
                    print(
                        f"disc_adv[{i}]: {disc_adv[:,i].shape}, disc_act: {disc_act[:, i].shape}"
                    )
                logsumexp = torch.logsumexp(disc_adv[:, i], dim=-1, keepdim=True)
                q_a = disc_adv[:, i].gather(1, disc_act[:, i].unsqueeze(-1))
                if debug:
                    print(f"q_a: {q_a.shape}, logsumexp: {logsumexp.shape}")
                cql_loss += (logsumexp - q_a).mean()

        if self.continuous_action_dims > 0:
            for i in range(self.continuous_action_dims):
                if debug:
                    print(
                        f"disc_adv[{i}]: {cont_adv[:,i].shape}, disc_act: {cont_act[:, i].shape}"
                    )
                logsumexp = torch.logsumexp(cont_adv[:, i], dim=-1, keepdim=True)
                q_a = cont_adv[:, i].gather(1, cont_act[:, i].unsqueeze(-1))
                if debug:
                    print(f"q_a: {q_a.shape}, logsumexp: {logsumexp.shape}")
                cql_loss += (logsumexp - q_a).mean()

        return cql_loss / 5

    # torch no grad called in reinfrocement learn so no need here
    def _target(
        self,
        values,
        advantages,
        rewards,
        terminated,
        action_dim=None,
        jagged=True,
        debug=True,
    ):
        if jagged:  # discrete action bins are jagget
            vals = values.squeeze(-1) if self.dueling else 0  # make it a column vector
            # action_dim = len(self.discrete_action_dims)
            Q_ = torch.zeros(
                size=(advantages[0].shape[0], len(action_dim)),
                device=self.device,
                dtype=torch.float32,
            )
            if debug:
                print("Jagged target() q shape, adv shape, and value shape")
                print(Q_.shape, vals)
                for i in range(len(action_dim)):
                    print("  " + str(advantages[i].shape))
                # print(advantages)

            for i in range(len(action_dim)):
                # Treat actions as probabalistic if using soft Q or m-dqn
                if self.dqn_type == "Munchausen" or self.dqn_type == "Soft":
                    lprobs = torch.log_softmax(
                        advantages[i] / self.entropy_loss_coef, dim=-1
                    )
                    probs = torch.exp(lprobs)

                    if debug:
                        print(
                            f"  M-DQN or Soft DQN: adv(max_a): {torch.max(advantages[i], dim=-1).values.shape}, vals: {vals.shape if self.dueling else values}"
                        )

                    q_vals = vals + advantages[i]

                    Q_[:, i] = torch.sum(
                        probs * (q_vals - self.entropy_loss_coef * lprobs), dim=-1
                    )
                else:
                    if debug:
                        print(
                            f"  Standard DQN: adv(max_a): {torch.max(advantages[i], dim=-1).values.shape}, vals: {vals.shape if self.dueling else values}"
                        )
                    Q_[:, i] = torch.max(advantages[i], dim=-1).values + vals

        else:  # continuous bins are not jagged
            if debug:
                print(f"  Not Jagget target advantages: {advantages.shape}")
            advantages = advantages.transpose(0, 1)
            # Treat actions as probabalistic if using soft Q or m-dqn
            if self.dqn_type == "Munchausen" or self.dqn_type == "Soft":
                lprobs = torch.log_softmax(advantages / self.entropy_loss_coef, dim=-1)
                probs = torch.exp(lprobs)
                if debug:
                    print(
                        f"  M-DQN or Soft DQN: adv(max_a): {torch.max(advantages, dim=-1).values.shape}, vals: {values.shape if self.dueling else values}"
                    )
                if self.dueling:
                    vals = values.unsqueeze(-1).expand(
                        advantages.shape
                    )  # make it a column vector
                else:
                    vals = 0
                q_vals = vals + advantages
                Q_ = torch.sum(
                    probs * (q_vals - self.entropy_loss_coef * lprobs), dim=-1
                )
                if debug:
                    print(
                        f"  vals shape: {vals.shape if self.dueling else 0} Q_: {Q_.shape}, rewards: {rewards.unsqueeze(-1).shape}, terminated: {terminated.unsqueeze(-1).shape}"
                    )
            else:
                if self.dueling:
                    vals = values
                else:
                    vals = 0
                if debug:
                    print(
                        f"  Standard DQN: adv(max_a): {torch.max(advantages, dim=-1).values.shape}, vals: {vals.shape if self.dueling else values}"
                    )
                Q_ = torch.max(advantages, dim=-1).values + vals

        if debug:
            print(
                f"  Q_: {Q_.shape}, rewards: {rewards.unsqueeze(-1).shape}, terminated: {terminated.unsqueeze(-1).shape}"
            )

        targets = (
            rewards.unsqueeze(-1) + (self.gamma * (1 - terminated)).unsqueeze(-1) * Q_
        )
        if debug:
            print(f"  targets: {targets.shape}")
        return targets

    def reinforcement_learn(
        self,
        discrete_actions,
        continuous_actions,
        obs,
        obs_,
        global_rewards,
        terminated,
        agent_num=0,
        critic_only=False,
        debug=False,
    ):
        if self.eval_mode:
            return 0, 0
        dqloss, cqloss = 0, 0

        if discrete_actions[0] is not None:
            discrete_actions = discrete_actions[agent_num]
        if continuous_actions[0] is not None:
            continuous_actions = self._discretize_actions(continuous_actions[agent_num])
        if debug:
            print(
                f"Discrete actions: {discrete_actions.shape}, Continuous actions: {continuous_actions.shape}"
            )
            print(
                f"Batch obs: {obs[agent_num].shape}, Batch obs_: {obs_[agent_num].shape}"
            )
        discrete_target = 0
        continuous_target = 0
        values, disc_adv, cont_adv = self.Q1(obs[agent_num])
        if cont_adv is not None:
            cont_adv = cont_adv.transpose(0, 1)
        with torch.no_grad():
            next_values, next_disc_adv, next_cont_adv = self.Q1(obs_[agent_num])
            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                if debug:
                    print("Testing discrete Targets")
                discrete_target = self._target(
                    values=next_values,
                    advantages=next_disc_adv,
                    rewards=global_rewards,
                    terminated=terminated,
                    action_dim=self.discrete_action_dims,
                    jagged=True,
                    debug=debug,
                )
                if self.dqn_type == "Munchausen":
                    for i in range(len(self.discrete_action_dims)):
                        temp_disc_adv = disc_adv[i].detach()
                        # if munchausen add tau*alpha*lp(a|s) to target
                        discrete_target[:, i] = discrete_target[
                            :, i
                        ] + self.entropy_loss_coef * self.munchausen * (
                            Categorical(
                                logits=temp_disc_adv / self.entropy_loss_coef
                            ).log_prob(discrete_actions[:, i])
                        )
            if (
                self.continuous_action_dims is not None
                and self.continuous_action_dims > 0
            ):
                if debug:
                    print("Testing continuous Targets")
                continuous_target = self._target(
                    values=next_values,
                    advantages=next_cont_adv,
                    rewards=global_rewards,
                    terminated=terminated,
                    jagged=False,
                    debug=debug,
                )
                if self.dqn_type == "Munchausen":
                    temp_cont_adv = cont_adv.detach()

                    continuous_target = (
                        continuous_target
                        + self.entropy_loss_coef
                        * self.munchausen
                        * torch.log_softmax(
                            temp_cont_adv / self.entropy_loss_coef, dim=-1
                        )
                        .gather(dim=-1, index=continuous_actions.unsqueeze(-1))
                        .squeeze(-1)
                    )

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            if debug:
                print(
                    f"Calculating cQ: cont_advs.shape: {cont_adv.shape}, continuous_actions: {continuous_actions.unsqueeze(-1).shape}, vals: {values.shape if self.dueling else 0}"
                )
            cQ = torch.gather(
                input=cont_adv,
                dim=-1,
                index=continuous_actions.unsqueeze(-1),
            ).squeeze(-1) + (values if self.dueling else 0)

            if debug:
                print(f"cQ: {cQ.shape}, continuous_target: {continuous_target.shape}")

        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:

            dQ = torch.zeros(
                size=(global_rewards.shape[0], len(self.discrete_action_dims)),
                device=self.device,
                dtype=torch.float32,
            )
            for d in range(len(self.discrete_action_dims)):
                if debug:
                    print(
                        f"  disc_adv: {disc_adv[d].shape}, disc_act: {discrete_actions[:, d].unsqueeze(-1).shape}"
                    )
                dQ[:, d] = (
                    torch.gather(
                        disc_adv[d],
                        dim=-1,
                        index=discrete_actions[:, d].unsqueeze(-1),
                    )
                    + values
                ).squeeze(-1)

        dqloss, cqloss = 0, 0
        trainable = False
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            dqloss = (dQ - discrete_target) ** 2
            dqloss = dqloss.mean()
            trainable = True

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            cqloss = (cQ - continuous_target) ** 2
            cqloss = cqloss.mean()
            trainable = True

        consqloss = 0
        if self.conservative:
            consqloss = self.cql_loss(
                disc_adv=disc_adv,
                cont_adv=cont_adv,
                disc_act=discrete_actions,
                cont_act=continuous_actions,
            )
        if trainable:
            loss = dqloss + cqloss + consqloss
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.clip_grad,
                    error_if_nonfinite=True,
                )
            self.optimizer.step()
        else:
            warnings.warn(
                "Action dims both zero so there is nothing to train. Not updating the model."
            )
        if dqloss != 0:
            dqloss = dqloss.item()
        if cqloss != 0:
            cqloss = cqloss.item()
        return dqloss, cqloss  # actor loss, critic loss

    def save(self, checkpoint_path):
        if self.eval_mode:
            print("Not saving because model in eval mode")
            return
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.Q1.state_dict(), checkpoint_path + "/Q1")

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        if not os.path.exists(checkpoint_path):
            return 0

        self.dqn_type = "EGreedy"
        if self.max_actions is not None:
            self.np_action_ranges = self.max_actions - self.min_actions
            self.action_ranges = torch.from_numpy(self.np_action_ranges).to(self.device)
            self.np_action_means = (self.max_actions + self.min_actions) / 2
            self.action_means = torch.from_numpy(self.np_action_means).to(self.device)

        if self.Q1 is None:
            self.Q1 = QS(
                obs_dim=self.obs_dim,
                continuous_action_dim=self.continuous_action_dims,
                discrete_action_dims=self.discrete_action_dims,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                orthogonal=self.orthogonal,
                dueling=self.dueling,
                n_c_action_bins=self.n_c_action_bins,
                device=self.device,
            )
        self.Q1.load_state_dict(torch.load(checkpoint_path + "/Q1", weights_only=True))
        self.Q1.to(self.device)

        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.to(self.device)

    def __str__(self):
        st = ""

        for i in self.__dict__.keys():
            st += f"i: {self.__dict__[i]}"

        return st


# Load in your trained model and return the corresponding agent action based on the information provided in step()
class hybrid_agent:

    # Add Variables required for solution
    class PID:
        def __init__(self, kp, ki, kd):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.integral = 0
            self.prev_error = 0

        def step(self, error):
            derivative = error - self.prev_error
            self.integral += error * (5 / (np.abs(derivative) + 1))

            self.prev_error = error
            return self.kp * error + self.ki * self.integral + self.kd * derivative

    def __init__(
        self, policy_type="dqn", TRAINING=False, anum=0, test=False, path=None
    ):
        self.policy_type = policy_type
        self.manager_act = 0
        self.policy_angle = 0.0
        self.policy_throttle = 0.0
        # print("timAgentInit")
        # Load in policy or anything else you want to load/do here
        # NOTE: You can only load from files that are in the same directory as the solution.py or a subdirectory
        self.obs_dim = 61
        # Load in learned policies see examples below:
        self.manager = DQN(
            obs_dim=self.obs_dim,
            discrete_action_dims=[7],
            continuous_action_dims=0,
            hidden_dims=[128, 128],
            head_hidden_dim=64,
            gamma=0.99,
            lr=3e-4,
            imitation_lr=1e-4,
            entropy=0.0,
            munchausen=0.0,
            imitation_type="cross_entropy",
            n_c_action_bins=2,
            init_eps=0.25,
            eps_decay_half_life=10000,
            name="timManager",
            device="cuda",
        )
        self.policy_one = DQN(
            obs_dim=self.obs_dim,
            discrete_action_dims=None,
            continuous_action_dims=2,
            hidden_dims=[128, 128],
            head_hidden_dim=64,
            gamma=0.99,
            lr=3e-4,
            imitation_lr=1e-4,
            entropy=0.0,
            munchausen=0.0,
            imitation_type="cross_entropy",
            n_c_action_bins=11,
            init_eps=0.25,
            eps_decay_half_life=10000,
            name="timAgent",
            device="cuda",
            max_actions=np.array([3.0, 90.0]),
            min_actions=np.array([0.0, -90.0]),
            conservative=False,
        )
        # Save blue agent models
        if path is None:
            self.policy_one.load(f"./models/blue_agent_{anum}_model/")
            self.manager.load(f"./models/blue_agent_{anum}_manager/")
        else:
            self.policy_one.load(path + f"_model/")
            self.manager.load(path + f"_manager/")
        if test:
            print("Loading in test mode")
            self.policy_one.init_eps = 0.0
            self.manager.init_eps = 0.0
        a1 = np.random.rand(self.obs_dim)

        # print(f"policy_one: {self.policy_one.train_action(observations=a1)}")
        # print(f"manager: {self.manager.train_action(observations=a1)}")

        # exit()

        # Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        # self.policy_two = #Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        # self.policy_three = #Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        self.counter = 0
        self.old_dist = 0
        self.last_speed = 0.01
        self.last_pos = np.array([0, 0])

        self.last_heading = 1.5
        self.last_acc = 0
        self.lastldot = 0
        self.last_bearing_from_me = 0
        self.turnpid = self.PID(30, 0, 3)

    def _train_action(
        self,
        agent_id: str,
        full_obs_normalized: dict,
        full_obs: dict,
        global_state: dict,
    ):
        policy_throttle = 3.0
        policy_angle = 0.0
        my_name, my_idx = self._get_my_name_and_idx(agent_id, full_obs_normalized)
        disc_manager_act, cont_manager_act, _1, _2, _3 = self.manager.train_action(
            full_obs_normalized[agent_id], step=True
        )
        if disc_manager_act[0] < 3:  # pronav targeting someone
            policy_angle = self.pronav_tim2(full_obs[my_name], disc_manager_act[0])
        elif disc_manager_act[0] == 3:  # dodge
            policy_angle, _1 = self.dodge_obstacles(full_obs[my_name])
        elif disc_manager_act[0] == 4:  # go to flag
            policy_angle = self.go_to_flag(full_obs[my_name])
        elif disc_manager_act[0] == 5:  # go to home
            policy_angle = self.go_to_home(full_obs[my_name])
        elif disc_manager_act[0] == 6:  # take NN policy
            # print(
            #    f"Taking NN policy: {full_obs_normalized[my_name].shape} {full_obs_normalized[my_name]}"
            # )
            pd_act, pc_act, _1, _2, _2 = self.policy_one.train_action(
                full_obs_normalized[my_name], step=True
            )
            # print(f"pd_act: {pd_act}, pc_act: {pc_act}")
            policy_angle = pc_act[0]
            policy_throttle = pc_act[1]

        self.manager_act = disc_manager_act[0]
        self.policy_angle = policy_angle
        self.policy_throttle = policy_throttle
        # print(
        #    f"manager_act: {disc_manager_act[0]}, policy_angle: {policy_angle}, policy_throttle: {policy_throttle}"
        # )
        # print(disc_manager_act[0], policy_throttle, policy_angle)
        return disc_manager_act[0], policy_throttle, policy_angle

    def _get_my_name_and_idx(self, agent_id: str, full_obs_normalized: dict):
        self.lastwallh = full_obs_normalized[agent_id][4]
        self.counter += 1
        agents = list(full_obs_normalized.keys())
        my_name = agent_id
        if agent_id in agents:  # Todo more preprocessing for this
            my_name = agent_id
        else:
            my_name = agents[int(agent_id)]
        my_idx = agents.index(my_name)
        return my_name, my_idx

    def _full_heuristic(self, agent_id, full_obs: dict, full_obs_normalized: dict):
        my_name, my_idx = self._get_my_name_and_idx(agent_id, full_obs_normalized)
        if self.policy_type == "defender":
            closest_opponent = self.closest_opponent_to_home(
                full_obs_normalized[my_name]
            )
            pnt_angle = self.pronav_tim2(full_obs[my_name], closest_opponent + 2)
            # print("defender")
            # print([3, pnt_angle])
            return [3, pnt_angle]
        if self.policy_type == "attacker":
            d_angle, weight = self.dodge_obstacles(full_obs[my_name])
            if weight > 10:
                # input(f"weight: {weight}, d_angle: {d_angle}")
                # print([3, d_angle])
                return [3, d_angle]
            to_flag_angle = self.go_to_flag(full_obs[my_name])
            to_home_angle = self.go_to_home(full_obs[my_name])
            if full_obs[my_name][15] > 0:
                # print("attacker1")
                # print([3, to_home_angle])
                return [3, to_home_angle]

            # print("attacker2")
            # print([3, to_flag_angle])
            return [
                3,
                to_flag_angle,
            ]
        ac = [3, np.random.uniform(-180, 180)]
        return ac

    # Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(
        self,
        agent_id: str,
        full_obs_normalized: dict,
        full_obs: dict,
        global_state: dict,
    ):
        if self.policy_type == "defender" or self.policy_type == "attacker":
            return self._full_heuristic(agent_id, full_obs, full_obs_normalized)

        my_name, my_idx = self._get_my_name_and_idx(agent_id, full_obs_normalized)
        # If I have the flag go home
        if full_obs[my_name][15] > 0:
            self.manager_act = 4
            to_home_angle = self.go_to_home(full_obs[my_name])
            return [3, to_home_angle]

        # If Im about to hit a wall dont
        d_angle, weight = self.dodge_obstacles(full_obs[my_name])
        if weight > 10:
            self.manager_act = 3
            return [3, d_angle]

        # If I am the closest to the opponent flag that is in play, go to it
        closest_entity_to_flag = self.closest_to_their_flag(
            full_obs_normalized[my_name]
        )
        if closest_entity_to_flag == 0:
            self.manager_act = 4
            to_flag_angle = self.go_to_flag(full_obs[my_name])
            return [3, to_flag_angle]

        # If the enemy has the flag and I have my tag, go to them
        for i in range(3):
            if full_obs[my_name][25 + (i + 2) * 8] > 0 and full_obs[my_name][17] > 0.5:
                # print("DEFENDING")
                self.manager_act = i
                pnt_angle = self.pronav_tim2(full_obs[my_name], i + 2)
                return [3, pnt_angle]

        if self.policy_type == "dqn":
            _, policy_angle, policy_throttle = self._train_action(
                agent_id,
                full_obs_normalized,
                full_obs,
                global_state,
            )
            # print("DQN")
            # print([policy_angle, policy_throttle])
            return [policy_angle, policy_throttle]
        # print(full_obs_normalized[agent_id].shape)

        ac = [3, np.random.uniform(-180, 180)]
        # print("fallback")
        # print(ac)
        return ac  # [1.0, random.uniform(-180, 180)]

        """
        Discrete Action Space:
            0 - [1.0,  180] right?
            1 - [1.0,  135]
            2 - [1.0,  90]
            3 - [1.0,   45]
            4 - [1.0,    0] no turn
            5 - [1.0, -45] left
            6 - [1.0,  -90]
            7 - [1.0, -135]
            8 - [0.5, 180]right?
            9 - [0.5,  135]
            10 - [0.5,   90]
            11 - [0.5,  45]
            12 - [0.5,    0] no turn
            13 - [0.5,  -45] left
            14 - [0.5, -90]
            15 - [0.5, -135]
            16 - [0.0,    0] no op
            # ACTION MAP:


        Default Observation Space (per agent):
            0 - Opponent home relative bearing (clockwise degrees)
            1 - Opponent home distance (meters)
            2 - Home relative bearing (clockwise degrees)
            3 - Home distance (meters)
            4 - Wall 0 relative bearing (clockwise degrees)
            5 - Wall 0 distance (meters)
            6 - Wall 1 relative bearing (clockwise degrees)
            7 - Wall 1 distance (meters)
            8 - Wall 2 relative bearing (clockwise degrees)
            9 - Wall 2 distance (meters)
            10- Wall 3 relative bearing (clockwise degrees)
            11- Wall 3 distance (meters)
            12- Scrimmage line bearing (clockwise degrees)
            13- Scrimmage line distance (meters)
            14- Own speed (meters per second)
            15- Has flag status (boolean)
            16- On side status (boolean)
            17- Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            18- Is tagged status (boolean)
            19- Team score (cummulative flag captures by agent's team)
            20- Opponent score (cummulative flag captures by opposing team)
              - For each other agent (teammates first):
              21- Bearing from you (clockwise degrees)
              22- Distance (meters)
              23- Heading of other agent relative to the vector to you (clockwise degrees)
              24- Speed (meters per second)
              25- Has flag status (boolean)
              26- On side status (boolean)
              27- Tagging cooldown (seconds)
              28- Is tagged status (boolean)

        Lidar Observation Space (per agent):
            - Opponent home relative bearing (clockwise degrees)
            - Opponent home distance (meters)
            - Home relative bearing (clockwise degrees)
            - Home distance (meters)
            - Scrimmage line bearing (clockwise degrees)
            - Scrimmage line distance (meters)
            - Own speed (meters per second)
            - Has flag status (boolean)
            - Team has opponent's flag status (boolean)
            - Opponent has team's flag status (boolean)
            - On side status (boolean)
            - Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            - Is tagged status (boolean)
            - Team score (cummulative flag captures by agent's team)
            - Opponent score (cummulative flag captures by opposing team)
            - Lidar ray distances (meters)
            - Lidar ray labels (see lidar_detection_classes in config.py)

        Note 1: the angles are 0 when the agent is pointed directly at the object
                and increase in the clockwise direction
        Note 2: when normalized, the boolean args are -1 False and +1 True
        Note 3: the values are normalized by default
        Note 4: units with 'meters' are either in actual meters or mercator xy meters depending if
                self.gps_env is True or not (except for speed which is always meters per second)

        Developer Note 1: changes here should be reflected in _register_state_elements.
        Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
        """
        if agent_id == "agent_0" or agent_id == "agent_3":
            return self.policy_one.compute_single_action(
                full_obs_normalized[agent_id], explore=False
            )[0]
        elif agent_id == "agent_1" or agent_id == "agent_4":
            return self.policy_two.compute_single_action(
                full_obs_normalized[agent_id], explore=False
            )[0]
        else:
            return self.policy_three.compute_single_action(
                full_obs_normalized[agent_id], explore=False
            )[0]

    def closest_to_their_flag(self, obs):
        # if a teammate has the flag, return them
        for i in range(2):
            if obs[25 + (i) * 8] > 0:
                return i

        op_home_bearing = obs[0] * np.pi
        op_home_dist = obs[1] + 1
        smallest_dist = op_home_dist
        closest_opponent = 0
        for i in range(5):
            object_dist_from_flag = np.sqrt(
                (
                    op_home_dist * np.cos(op_home_bearing)
                    - (obs[22 + 8 * i] + 1) * np.cos(obs[21 + 8 * i] * np.pi)
                )
                ** 2
                + (
                    op_home_dist * np.sin(op_home_bearing)
                    - (obs[22 + 8 * i] + 1) * np.sin(obs[21 + 8 * i] * np.pi)
                )
                ** 2
            )
            if object_dist_from_flag < smallest_dist:
                if i < 2 or obs[27 + (i) * 8] > 0:
                    smallest_dist = object_dist_from_flag
                    closest_opponent = i

        return closest_opponent

    def closest_opponent_to_home(self, obs):
        # Find the closest opponent to my home
        home_bearing = obs[2] * np.pi
        home_dist = obs[3] + 1
        opponent_bearings = [obs[37] * np.pi, obs[45] * np.pi, obs[53] * np.pi]
        opponent_dists = [obs[38] + 1, obs[46] + 1, obs[54] + 1]
        closest_opponent = 0
        closest_dist = 1000
        for i in range(3):
            if obs[25 + 8 * (i + 2)] > 0:
                closest_opponent = i
                break
            opponent_dist_from_home = np.sqrt(
                (
                    home_dist * np.cos(home_bearing)
                    - opponent_dists[i] * np.cos(opponent_bearings[i])
                )
                ** 2
                + (
                    home_dist * np.sin(home_bearing)
                    - opponent_dists[i] * np.sin(opponent_bearings[i])
                )
                ** 2
            )
            if opponent_dist_from_home < closest_dist:
                closest_dist = opponent_dist_from_home
                closest_opponent = i
        return closest_opponent

    def dodge_obstacles(self, obs):
        weights = np.zeros(4)
        weights[0] = 0.2 / ((obs[5] + 1) ** 2)
        weights[1] = 0.2 / ((obs[7] + 1) ** 2)
        weights[2] = 0.2 / ((obs[9] + 1) ** 2)
        weights[3] = 0.2 / ((obs[11] + 1) ** 2)

        angles = np.zeros(4)
        angle = 0
        weight = 0
        for i in range(4):
            if obs[4 + 2 * i] > 0:
                angles[i] = -180 - obs[4 + 2 * i] * 180
            else:
                angles[i] = 180 - obs[4 + 2 * i] * 180
            if weights[i] > weight and abs(obs[4 + 2 * i] * 180) < 90:
                angle = np.clip(angles[i], -70, 70)
                weight = weights[i]
        return angle, weight

    def go_to_flag(self, obs):
        return np.clip(180 * obs[0], -60, 60)

    def go_to_home(self, obs):
        return np.clip(180 * obs[12], -60, 60)

    def cross_2d(self, v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def to_unit_circle(self, theta):
        return np.pi / 2 - (theta * np.pi)

    def proportional_navigation(self, obs, target_idx):
        # print(obs)
        # proportional navigation from the boats reference frame
        #     y
        #    / \
        #     |
        #     b - > x
        #
        self_V = np.array([0, 0])
        bearing_from_me = self.to_unit_circle(obs[21 + 8 * target_idx])
        dist = (
            obs[22 + 8 * target_idx] + 1
        )  # all these plus 1's are from the normalization -1 to 1
        target_x = dist * np.cos(bearing_from_me)
        target_y = dist * np.sin(bearing_from_me)
        target_heading = -obs[23 + 8 * target_idx] * np.pi + bearing_from_me - np.pi
        target_speed = obs[24 + 8 * target_idx] + 1
        Vr = np.array(
            [
                target_speed * np.cos(target_heading),
                target_speed * np.sin(target_heading) - (obs[14] + 1),
            ]
        )
        print(
            f"target degree bearing: {bearing_from_me}, dist: {obs[22 + 8 * target_idx]}"
        )
        print(f"my_speed: {obs[14] + 1}, target_speed: {target_speed}, relative: {Vr}")

        print(target_x, target_y, target_heading, target_speed)
        R = np.array([target_x, target_y])

        # print("R", R, "Vr", Vr)
        Omega = self.cross_2d(R, Vr) / np.dot(R, R)
        R_norm = R / np.maximum(np.abs(R), np.zeros_like(R) + 0.01)
        N = 50
        # print("N * np.abs(Vr) * R_norm", N * np.abs(Vr) * R_norm, "Omega", Omega)
        acc = N * np.abs(Vr) * R_norm * Omega
        turn = acc[0] / max((obs[14] + 1) ** 2, 0.01)

        input(turn)
        acc[1] = 3
        return np.clip(turn, -90.0, 90.0)

    def future_navigation(self, obs, target_idx, lead_weight=0.1):
        # future navigation from the boats reference frame
        #     y
        #    / \
        #     |
        #     b - > x
        #
        self_V = np.array([0, 0])
        bearing_from_me = self.radian_in_bounds(
            self.to_unit_circle(obs[21 + 8 * target_idx])
        )
        dist = (
            obs[22 + 8 * target_idx] + 1
        )  # all these plus 1's are from the normalization -1 to 1
        target_x = dist * np.cos(bearing_from_me)
        target_y = dist * np.sin(bearing_from_me)
        target_heading = self.radian_in_bounds(
            -obs[23 + 8 * target_idx] * np.pi + bearing_from_me - np.pi
        )
        target_speed = obs[24 + 8 * target_idx] + 1
        Vr = np.array(
            [
                target_speed * np.cos(target_heading),
                target_speed * np.sin(target_heading) - (obs[14] + 1),
            ]
        )
        R = np.array([target_x, target_y])

        target = R + Vr * lead_weight * np.clip(1 + obs[22 + 8 * target_idx], 0.05, 1.0)
        angle = 90 - np.arctan2(target[1], target[0]) * 360 / (2 * np.pi)
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return np.clip(angle, -70, 70)

    def radian_heading_to_ego_degrees(self, radian_heading):
        h = 90 - radian_heading * 180 / np.pi
        while h > 180:
            h -= 360
        while h < -180:
            h += 360
        return h

    def radian_in_bounds(self, radian):
        while radian > 2 * np.pi:
            radian -= 2 * np.pi
        while radian < 0:
            radian += 2 * np.pi
        return radian

    def signed_angle(self, v1, v2):
        return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])

    def signed_angle2(self, a1, a2):
        return (a2 - a1 + np.pi) % (2 * np.pi) - np.pi

    def pronav_tim(self, obs, target_idx):
        bearing_from_me = self.to_unit_circle(obs[21 + 8 * target_idx])
        bearing_to_me = bearing_from_me + np.pi
        dist = (
            obs[22 + 8 * target_idx] + 1
        )  # all these plus 1's are from the normalization -1 to 1

        target_xb = dist * np.cos(bearing_from_me)
        target_yb = dist * np.sin(bearing_from_me)
        target_pos = np.array([target_xb, target_yb])

        target_heading = -obs[23 + 8 * target_idx] * np.pi + bearing_to_me
        target_speed = obs[24 + 8 * target_idx] + 1
        target_Vb = np.array(
            [
                target_speed * np.cos(target_heading),
                target_speed * np.sin(target_heading) - (obs[14] + 1),
            ]
        )
        # found empirically, idk wtf is going on with the unit conversion in this environment
        unit_conversion = 0.02683284878730774
        tau = 0.1  # simulation speed, we also need a conversion from normalized bs to actual speed

        target_next_xb = target_xb + target_Vb[0] * unit_conversion
        target_next_yb = target_yb + target_Vb[1] * unit_conversion
        target_next_pos = np.array([target_next_xb, target_next_yb])

        h1 = self.radian_in_bounds(np.arctan2(target_yb, target_xb))
        h2 = self.radian_in_bounds(np.arctan2(target_next_yb, target_next_xb))
        lambdadot = self.signed_angle(target_pos, target_next_pos)

        print(
            "target_xb: ",
            target_xb,
            ", target_yb: ",
            target_yb,
            ", target_heading: ",
            self.radian_heading_to_ego_degrees(bearing_from_me),
        )
        print(
            "h1: ",
            self.radian_heading_to_ego_degrees(h1),
            "h2: ",
            self.radian_heading_to_ego_degrees(h2),
            "Ldot: ",
            lambdadot / 2 / np.pi * 360,
        )
        print("target_Vb: ", target_Vb)
        curr_speed = max((obs[14] + 1), 0.01)
        avg_speed = (curr_speed + self.last_speed) / 2
        st = (
            "op hom dist:"
            + str(obs[1] + 1)
            + ", speed: "
            + str(obs[14] + 1)
            + "\nS "
            + str((obs[14] + 1) * 3 / 2)
            + ", D "
            + str((obs[1] + 1) * 89)
            + ", est_s: "
            + str(abs((obs[1] + 1) - self.old_dist) * 89 / tau)
            + ", ratio: "
            + str(abs((obs[1] + 1) - self.old_dist) / curr_speed / tau)
            + ", ratio2: "
            + str(abs((obs[1] + 1) - self.old_dist) / self.last_speed / tau)
        )
        self.old_dist = obs[1] + 1

        heading = self.radian_in_bounds(obs[4] * np.pi + np.pi)
        angle_change = self.signed_angle2(
            self.last_heading,
            heading,
        )

        print(f"angle_change: {angle_change:0.3f}, ldot: {lambdadot:0.3f} ")
        # lambdadot -= angle_change  # for the moving body frame, change los
        centripetal_acc = self.centripetal_acceleration(
            avg_speed, np.abs(angle_change), 1.0
        )
        centripetal_acc = (
            np.sign(angle_change) * centripetal_acc  # * 0.5 + 0.5 * self.last_acc
        )

        N = 10
        acc = N * lambdadot / 2
        error = centripetal_acc - acc
        turn = np.clip(self.turnpid.step(error), -30, 30)

        print(
            f"centripetal_acc: {np.clip(-acc * 30, -60, 60):0.3f}, target_accel: {acc} turn: {turn:0.3f}"
        )

        self.last_speed = max((obs[14] + 1), 0.01)
        self.last_heading = heading
        self.last_acc = centripetal_acc
        self.lastldot = lambdadot
        # input(obs[4] + 1)
        return np.clip(-acc * 30, -60, 60)

    def pronav_tim2(self, obs, target_idx):
        # print(21 + 8 * target_idx)
        # Get heading from south wall if we are closer to north wall than south
        if obs[9] + 1 > obs[5] + 1:
            heading = self.radian_in_bounds(obs[8] * np.pi + np.pi + np.pi / 2)
        else:
            heading = self.radian_in_bounds(obs[4] * np.pi + np.pi / 2)

        bearing_from_me = self.radian_in_bounds(
            self.to_unit_circle(obs[21 + 8 * target_idx])
        )
        inertial_bearing_from_me = self.radian_in_bounds(
            heading + bearing_from_me - np.pi / 2
        )
        inertial_direction_of_target = self.radian_in_bounds(
            inertial_bearing_from_me - np.pi - obs[23 + 8 * target_idx] * np.pi
        )

        off = self.signed_angle2(heading, inertial_bearing_from_me)
        if (
            abs(off) > 0.5 * np.pi
            or (obs[22 + 8 * target_idx] + 1) > 0.5
            or abs(self.signed_angle2(heading, inertial_direction_of_target))
            > np.pi / 2
        ):
            return self.future_navigation(obs, target_idx, lead_weight=0.3)

        R_b = np.array([obs[11] + 1, obs[9] + 1])  # our position
        R = np.array(
            [
                (obs[22 + 8 * target_idx] + 1) * np.cos(inertial_bearing_from_me),
                (obs[22 + 8 * target_idx] + 1) * np.sin(inertial_bearing_from_me),
            ]
        )

        S_b = np.array(obs[14] + 1) * 0.02683281898498535  # speed in mystery units
        V_b = np.array(
            [
                S_b * np.cos(heading),
                S_b * np.sin(heading),
            ]
        )
        V_t = (
            np.array(
                [
                    (obs[24 + 8 * target_idx] + 1)
                    * np.cos(inertial_direction_of_target),
                    (obs[24 + 8 * target_idx] + 1)
                    * np.sin(inertial_direction_of_target),
                ]
            )
            * 0.02683281898498535
        )
        V_r = V_t - V_b

        # print(
        #    f"heading (deg): {heading * 180 / np.pi:0.3f}, Rb: {R_b}, bearing: {inertial_bearing_from_me * 180 / np.pi:0.3f}, target_dir: {inertial_direction_of_target*180/np.pi:0.3f}"
        # )

        Omega = self.cross_2d(R, V_r) / np.dot(R, R)
        R_mag = np.sqrt(np.sum(np.square(R)))
        V_mag = np.sqrt(np.sum(np.square(V_r)))

        # print(
        #    f"R: {R}, Vr: {V_r}, Omega: {Omega:0.3f}, R_mag: {R_mag:0.3f} V_mag: {V_mag:0.3f}"
        # )

        N = 3
        a = -N * V_mag * Omega * R / R_mag
        # print(-a[1] * 300 / max(S_b**2, 0.1), -60, 60)
        self.last_speed = V_b
        self.last_pos = R_b.copy()
        # input()
        return np.clip(-a[1] * 300 / max(S_b**2, 0.1), -60, 60)

    def centripetal_acceleration(self, avg_speed, delta_angle, delta_time):
        """
        Calculate centripetal acceleration.

        Parameters:
        avg_speed (float): Average speed (v).
        delta_angle (float): Change in angle (Δθ in radians).
        delta_time (float): Change in time (Δt).

        Returns:
        float: Centripetal acceleration (a_c).
        """
        return avg_speed * (delta_angle / delta_time)

    def turn(self, obs, acc, tau=0.1):
        heading = self.radian_in_bounds(obs[4] * np.pi + np.pi)
        angle_change = self.signed_angle2(
            self.last_heading,
            heading,
        )
        speed = max((obs[14] + 1), 0.01)
        avg_speed = (speed + self.last_speed) / 2

        centripetal_acc = self.centripetal_acceleration(
            avg_speed, np.abs(angle_change), tau
        )
        centripetal_acc = np.sign(angle_change) * centripetal_acc
        error = (centripetal_acc + self.last_acc) / 2 - acc
        turn = np.clip(self.turnpid.step(error), -70, 70)

        print(
            f"angle_change: {angle_change:0.3f}, speed: {speed:0.3f}, avg_speed: {avg_speed:0.3f}, centripetal_acc: {centripetal_acc:0.3f}, error: {error:0.3f}, turn: {turn:0.3f}"
        )

        self.last_speed = speed
        self.last_heading = heading
        self.last_acc = centripetal_acc
        return turn
