# authors: anonymous
import numpy as np
from scipy.optimize import linprog
import itertools


# Computes the non-bootstrapping mask
def compute_mask(nb_states, nb_actions, epsilon, delta, batch):
    N_wedge = 2*(np.log((2*nb_states*nb_actions*2**nb_states)/delta))/epsilon**2
    return compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch), N_wedge


def compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch):
    count_state_action = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch:
        count_state_action[state, action] += 1
    return count_state_action > N_wedge


# Computes the transition errors for all state-action pairs
def compute_errors(nb_states, nb_actions, delta, batch):
    count_state_action = np.zeros((nb_states, nb_actions))
    errors = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch:
        count_state_action[state, action] += 1
    for state in range(nb_states):
        for action in range(nb_actions):
            if count_state_action[state, action] == 0:
                errors[state, action] = np.inf
            else:
                errors[state, action] = np.sqrt(
                    2*(np.log(2*(nb_states*nb_actions*2**nb_states)/delta))/count_state_action[state, action]
                )
    return errors


def policy_evaluation_exact(pi, r, p, gamma):
    """
    Evaluate policy by taking the inverse
    Args:
      pi: policy, array of shape |S| x |A|
      r: the true rewards, array of shape |S| x |A|
      p: the true state transition probabilities, array of shape |S| x |A| x |S|
    Return:
      v: 1D array with updated state values
    """
    # Rewards according to policy: Hadamard product and row-wise sum
    r_pi = np.einsum('ij,ij->i', pi, r)

    # Policy-weighted transitions:
    # multiply p by pi by broadcasting pi, then sum second axis
    # result is an array of shape |S| x |S|
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
    return v, r + gamma*np.einsum('i, jki->jk', v, p)


def state_action_density(pi, p):
    x_0 = np.zeros(pi.shape[0])
    x_0[0] = 1
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    d = np.dot(x_0, np.linalg.inv((np.eye(p_pi.shape[0]) - p_pi)))
    print(d)
    print(d.sum())
    dxa = np.minimum(1,np.einsum('i, ij->ij', d, pi))
    dxaf = dxa[:,2:]
    xs = [1,2,3,5,7,10,15,20,30,50,70,100,150,200,300,500,700,1000,1500,2000,3000,5000,7000,10000]
    ys = []
    zs = []
    for x in xs:
        y = (x*dxa*(1-dxa)**(x-1)).sum()
        z = (x*dxaf*(1-dxaf)**(x-1)).sum()
        ys.append(y)
        zs.append(z)
    print(ys)
    print(zs)
    print(dxa)
    print(dxaf)


def softmax(q, temp):
    exp = np.exp(temp*(q - np.max(q, axis=1)[:,None]))
    pi = exp / np.sum(exp, axis=1)[:,None]
    return pi


class spibb():
    # gamma is the discount factor,
    # nb_states is the number of states in the MDP,
    # nb_actions is the number of actions in the MDP,
    # pi_b is the baseline policy,
    # mask is the mask where the one does not need to bootstrap,
    # model is the transition model,
    # reward is the reward model,
    # space denotes the type of policy bootstrapping,
    # q_pib_est is the MC estimator of the state values of baseline policy,
    # max_nb_it is the maximal number of policy improvement
    def __init__(self, gamma, nb_states, nb_actions, pi_b, mask, model, reward, space,
                 q_pib_est=None, errors=None, epsilon=None, lmbda=0, max_nb_it=99999):
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.P = model
        self.pi_b = pi_b
        self.pi_b_masked = self.pi_b.copy()
        self.pi_b_masked[mask] = 0
        self.mask = mask
        self.R = reward.reshape(self.nb_states * self.nb_actions)
        self.space = space
        self.q_pib_est_masked = None
        if q_pib_est is not None:
            self.q_pib_est_masked = q_pib_est.copy()
            self.q_pib_est_masked[mask] = 0
        self.errors = errors
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.max_nb_it = max_nb_it

    # starts a new episode (during the policy exploitation)
    def new_episode(self):
        self.has_bootstrapped = False

    # trains the policy
    def fit(self):
        pi = self.pi_b.copy()
        q = np.zeros((self.nb_states, self.nb_actions))
        old_q = np.ones((self.nb_states, self.nb_actions))
        nb_sa = self.nb_states * self.nb_actions
        nb_it = 0
        old_pi = None
        while np.linalg.norm(q - old_q) > 0.000000001 and nb_it < self.max_nb_it:
            old_q = q.copy()
            M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
            q = np.dot(np.linalg.inv(M), self.R).reshape(self.nb_states, self.nb_actions)
            if self.q_pib_est_masked is not None:
                q += self.q_pib_est_masked
            pi = self.update_pi(q, old_pi)
            old_pi = pi
            nb_it += 1
            if nb_it > 1000:
                with open("notconverging.txt", "a") as myfile:
                    myfile.write(str(self.space) + " epsilon=" + str(self.epsilon) + " is not converging. \n")
                break
        self.pi = pi
        self.q = q

    # does the policy improvement inside the policy iteration loop
    def update_pi(self, q, old_pi=None):
        if self.space == 'Pi_b_SPIBB':
            pi = self.pi_b_masked.copy()
            for s in range(self.nb_states):
                if len(q[s, self.mask[s]]) > 0:
                    pi_b_masked_sum = np.sum(self.pi_b_masked[s])
                    pi[s][np.where(self.mask[s])[0][np.argmax(q[s, self.mask[s]])]] = 1 - pi_b_masked_sum
        elif self.space == 'Pi_0_SPIBB':
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                if len(q[s, self.mask[s]]) > 0:
                    pi[s][np.where(self.mask[s])[0][np.argmax(q[s, self.mask[s]])]] = 1
                else:
                    pi[s] = self.pi_b[s].copy()
        elif self.space == 'Pi_leq_b_SPIBB':
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                A = np.argsort(-q[s, :])
                pi_current_sum = 0
                for a in A:
                    if self.mask[s, a] or self.pi_b[s, a] > 1 - pi_current_sum:
                        pi[s, a] = 1 - pi_current_sum
                        break
                    else:
                        pi[s, a] = self.pi_b[s, a]
                        pi_current_sum += pi[s, a]
        elif self.space == 'Soft_SPIBB_max_Q':
            new_pi = self.pi_b.copy()
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                inf_err_idx = self.errors[s] == np.inf  # infinite errors
                if np.sum(inf_err_idx) < self.nb_actions:  # if there exists a finite error
                    a_star = np.where(~inf_err_idx)[0][np.argmax(q[s, ~inf_err_idx])]  # optimal action
                    e_star = self.errors[s, a_star]  # by definition, e_star < np.inf
                    idx = np.logical_not(inf_err_idx)  # idx is the set of (s,a) with finite errors except (s,a*)
                    idx[a_star] = False
                    err_tot = (e_star * abs(1 - np.sum(self.pi_b[s, inf_err_idx]) - self.pi_b[s, a_star])) + self.pi_b[s, idx].dot(self.errors[s, idx])
                    rate = err_tot / self.epsilon
                    if rate <= 1.000000001:
                        new_pi[s, idx] = 0
                    else:
                        new_pi[s, idx] = new_pi[s, idx] * (1 - 1/rate)
                    new_pi[s, a_star] = 1 - np.sum(self.pi_b[s, inf_err_idx]) - np.sum(new_pi[s, idx])
                # Local policy improvement check, required for convergence
                if old_pi is not None:
                    new_local_v = new_pi[s, :].dot(q[s, :])
                    old_local_v = old_pi[s, :].dot(q[s, :])
                    if new_local_v >= old_local_v:
                        pi[s] = new_pi[s]
                    else:
                        pi[s] = old_pi[s]
                else:
                    pi[s] = new_pi[s]
        elif self.space == 'Soft_SPIBB_max_Q_Le':
            new_pi = self.pi_b.copy()
            pi = np.zeros(self.pi_b_masked.shape)
            lmbda = self.lmbda
            for s in range(self.nb_states):
                inf_err_idx = self.errors[s] == np.inf  # infinite errors
                if np.sum(inf_err_idx) < self.nb_actions:  # if there exists a finite error
                    a_star = np.where(~inf_err_idx)[0][np.argmax(q[s, ~inf_err_idx] - lmbda*self.errors[s, ~inf_err_idx])]  # optimal action
                    e_star = self.errors[s, a_star]  # by definition, e_star < np.inf
                    idx = np.logical_not(inf_err_idx)  # idx is the set of (s,a) with finite errors except (s,a*)
                    idx[a_star] = False
                    err_tot = (e_star * abs(1 - np.sum(self.pi_b[s, inf_err_idx]) - self.pi_b[s, a_star])) + self.pi_b[s, idx].dot(self.errors[s, idx])
                    rate = err_tot / self.epsilon
                    if rate <= 1.000000001:
                        new_pi[s, idx] = 0
                    else:
                        new_pi[s, idx] = new_pi[s, idx] * (1 - 1/rate)
                    new_pi[s, a_star] = 1 - np.sum(self.pi_b[s, inf_err_idx]) - np.sum(new_pi[s, idx])
                # Local policy improvement check, required for convergence
                if old_pi is not None:
                    new_local_v = new_pi[s, :].dot(q[s, :])
                    old_local_v = old_pi[s, :].dot(q[s, :])
                    if new_local_v >= old_local_v:
                        pi[s] = new_pi[s]
                    else:
                        pi[s] = old_pi[s]
                else:
                    pi[s] = new_pi[s]
        elif self.space == 'Soft_SPIBB_sort_Q':
            pi = np.zeros(self.pi_b_masked.shape)
            pi_t = self.pi_b.copy()
            for s in range(self.nb_states):
                breakflag = False
                excedent = 0
                A_bot = np.argsort(q[s, :]) # increasing order
                allowed_error = self.epsilon
                for a_bot in A_bot:
                    mass_bot = min(pi_t[s, a_bot], allowed_error/(2*self.errors[s, a_bot]))
                    #  A_top is sorted in decreasing order :
                    A_top = np.argsort(-(q[s, :]-q[s, a_bot])/self.errors[s, :])
                    for a_top in A_top:
                        if a_top == a_bot:
                            break
                        mass_top = min(mass_bot, allowed_error/(2*self.errors[s, a_top]))
                        if mass_top > 0:
                            mass_bot -= mass_top
                            pi_t[s, a_bot] -= mass_top
                            pi_t[s, a_top] += mass_top
                            allowed_error -= mass_top*(self.errors[s, a_bot] + self.errors[s, a_top])
                            if mass_bot == 0:
                                break
                # Local policy improvement check, required for convergence
                if old_pi is not None:
                    new_local_v = pi_t[s, :].dot(q[s, :])
                    old_local_v = old_pi[s, :].dot(q[s, :])
                    if new_local_v >= old_local_v:
                        pi[s] = pi_t[s]
                    else:
                        pi[s] = old_pi[s]
                else:
                    pi[s] = pi_t[s]
        elif self.space == 'Soft_SPIBB_sort_Q_Le':
            pi = np.zeros(self.pi_b_masked.shape)
            pi_t = self.pi_b.copy()
            lmbda = self.lmbda
            for s in range(self.nb_states):
                breakflag = False
                excedent = 0
                A_bot = np.argsort(q[s, :] - lmbda*self.errors[s, :]) # increasing order
                allowed_error = self.epsilon
                for a_bot in A_bot:
                    mass_bot = min(pi_t[s, a_bot], allowed_error/(2*self.errors[s, a_bot]))
                    #  A_top is sorted in decreasing order :
                    A_top = np.argsort(-(q[s, :]-lmbda*self.errors[s, :]-q[s, a_bot]+lmbda*self.errors[s, a_bot])/self.errors[s, :])
                    for a_top in A_top:
                        if a_top == a_bot:
                            break
                        mass_top = min(mass_bot, allowed_error/(2*self.errors[s, a_top]))
                        if mass_top > 0:
                            mass_bot -= mass_top
                            pi_t[s, a_bot] -= mass_top
                            pi_t[s, a_top] += mass_top
                            allowed_error -= mass_top*(self.errors[s, a_bot] + self.errors[s, a_top])
                            if mass_bot == 0:
                                break
                # Local policy improvement check, required for convergence
                if old_pi is not None:
                    new_local_v = pi_t[s, :].dot(q[s, :])
                    old_local_v = old_pi[s, :].dot(q[s, :])
                    if new_local_v >= old_local_v:
                        pi[s] = pi_t[s]
                    else:
                        if new_local_v < old_local_v - 10**-8:
                            print('policy improvement failure: ' + str(new_local_v-old_local_v))
                        pi[s] = old_pi[s]
                else:
                    pi[s] = pi_t[s]
        elif self.space == 'Soft_SPIBB_simplex':
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                finite_err_idx = self.errors[s] < np.inf
                c = np.zeros(2*self.nb_actions)
                c[0:self.nb_actions] = -q[s, :]
                Aeq = np.zeros(2*self.nb_actions)
                Aeq[0:self.nb_actions] = 1
                Aub = np.zeros(2*self.nb_actions)
                Aub[self.nb_actions:2*self.nb_actions][finite_err_idx] = self.errors[s, finite_err_idx]
                Aeq = [Aeq]
                beq = [1]
                Aub = [Aub]
                bub = [self.epsilon]
                if finite_err_idx.sum() == 0:
                    pi[s] = self.pi_b[s]
                else:
                    for idx in range(len(finite_err_idx)):
                        if not finite_err_idx[idx]:
                            new_Aeq = np.zeros(2*self.nb_actions)
                            new_Aeq[idx] = 1
                            Aeq.append(new_Aeq)
                            beq.append(self.pi_b[s, idx])
                        else:
                            new_Aub = np.zeros(2*self.nb_actions)
                            new_Aub[idx] = 1
                            new_Aub[idx+self.nb_actions] = -1
                            Aub.append(new_Aub)
                            bub.append(self.pi_b[s, idx])
                            new_Aub_2 = np.zeros(2*self.nb_actions)
                            new_Aub_2[idx] = -1
                            new_Aub_2[idx+self.nb_actions] = -1
                            Aub.append(new_Aub_2)
                            bub.append(-self.pi_b[s, idx])
                    res = linprog(c, A_eq=Aeq, b_eq=beq, A_ub=Aub, b_ub=bub)
                    pi[s] = [p if p >= 0 else 0.0 for p in res.x[0:self.nb_actions]]  # Fix rounding error
        elif self.space == 'Soft_SPIBB_simplex_2':
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                finite_err_idx = self.errors[s] < np.inf
                if finite_err_idx.sum() == 0:
                    pi[s] = self.pi_b[s]
                else:
                    c = -q[s, :]
                    Aeq = [np.ones(self.nb_actions)]
                    beq = [1]
                    for idx in range(len(finite_err_idx)):
                        if not finite_err_idx[idx]:
                            new_Aeq = np.zeros(self.nb_actions)
                            new_Aeq[idx] = 1
                            Aeq.append(new_Aeq)
                            beq.append(self.pi_b[s, idx])
                    n_finite = finite_err_idx.sum()
                    Aub = np.zeros((2**n_finite, self.nb_actions))
                    Aub[:, finite_err_idx] = self.errors[s, finite_err_idx]*np.asarray(list(map(list, itertools.product([-1, 1], repeat=n_finite))))
                    bub = self.epsilon + (self.pi_b[s, :]*Aub).sum(axis=1)
                    res = linprog(c, A_eq=Aeq, b_eq=beq, A_ub=Aub, b_ub=bub)
                    pi[s] = [p if p >= 0 else 0.0 for p in res.x]  # Correct rounding error
        # 'default' behaviour is used when there is no constraint in the the policy improvement projection
        else:
            pi = np.zeros(self.pi_b_masked.shape)
            for s in range(self.nb_states):
                pi[s, np.argmax(q[s, :])] = 1
        return pi

    # implements the trained policy
    def predict(self, state, bootstrap):
        if self.has_bootstrapped:
            choice = np.random.choice(self.nb_actions, 1, p=self.pi_b[state])
        else:
            choice = np.random.choice(self.nb_actions, 1, p=self.pi[state])
            # np.sum(self.P[state, choice]) equals 0 when bootstrapping, and 1 when not bootstrapping
            if bootstrap and np.sum(self.P[state, choice]) < 0.5:
                self.has_bootstrapped = True
        return choice
