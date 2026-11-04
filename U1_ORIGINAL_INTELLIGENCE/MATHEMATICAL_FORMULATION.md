# U1 — Mathematical Formulation

## 1. Environment class: hidden-law entity worlds

_Version history: v1 (glow-only pokes, dense laws) was rejected as too easy (`results/dev/env_v1_findings.md`);
v2 was used in attempt 1; **v3** (below) is used in revision R1. The code (`U1_IMPLEMENTATION/u1/env.py`) is
authoritative._

A **universe** fixes hidden laws over roles $\mathcal R=\{0,\dots,7\}$:
* roles 0–4 are known (training and test), role 5 is unseen (test only), and roles 6–7 appear only in the broad
  training distribution of one comparator;
* each role has a distinct 3-bit probe signature $\sigma_r$ and a distinct 3-bit appearance code $\kappa_r$;
* a symmetric reaction law $L$ has 3–5 reacting pairs among the known roles, every known role reacts with at least
  one, and roles 5–7 react with 1–2 known roles each and never with themselves or each other;
* every condition is checked to be samplable (v3; attempt 1's generator lacked this check).

A **world** is one episode with $n=5$ entities, horizon $T=20$, and 1–2 reacting pairs. Entity $e$ has a hidden
role $r_e$ and appearance $x_e\in\mathbb R^4$:
* for a known role, $x_{e,0..2}=\pm2$ according to $\kappa_{r_e}$, each sign flipped with probability $1-\rho$,
  plus noise;
* for other roles the signs are random;
* $x_{e,3}$ is noise.

Actions, $|\mathcal A|=15$:
* $\mathrm{probe}(e)$ (reward 0) returns 3 bits, each equal to $\sigma_{r_e}$'s bit with probability 0.9;
* $\mathrm{pair}(i,j)$ reacts with probability $p_{hi}=0.9$ if $L_{r_ir_j}=1$, else $p_{lo}=0.05$; reward =
  reaction.

The learner observes $x$, its actions, outcomes and rewards — never roles, laws, $\rho$, the cue dimensions, the
condition or change times.

Conditions (`EXPERIMENTS_AND_BASELINES.md`):

| condition | roles | cue validity | change |
|---|---|---|---|
| training-like | 2–3 distinct known roles | $\rho=0.9$ | — |
| cue reversed | as training | $\rho=0.1$ | — |
| recombined | 4–5 distinct known roles | 0.9 | — |
| unseen concept | every reacting pair involves role 5 | 0.9 | — |
| role switch | as training | 0.9 | at $t=10$ one entity changes role so that every reacting pair stops and 1–2 others react |

Broad training (comparators only): $\rho\sim U(0.3,1)$, switches at random $t\in[5,15]$ in 25% of worlds, roles
6–7 in 25% of worlds.

## 2. The learner (IRA: interventional role abstraction)

### 2.1 Persistent knowledge (the library)
$\mathcal K=\{1,\dots,K\}$ concepts, each with sufficient statistics accumulated over past worlds:
* probe-bit counts $(G^+_{kb},G^-_{kb})$ for each bit $b$;
* reaction counts for each concept pair $(S_{kl},F_{kl})$;
* appearance-sign counts per feature $d$: $(A^+_{kd},A^-_{kd})$;
* membership count $N_k$.

Knowledge grows by **adding** counts and concepts. Nothing learned earlier is retrained or overwritten.

### 2.2 Within-world state
Candidate labels $\mathcal C=\mathcal K_{\text{top}}\cup\{\nu_1,\dots,\nu_M\}$ (the most frequent known concepts plus
$M$ fresh labels for not-yet-known concepts). The state is the posterior over assignments
$z\in\mathcal C^{n}$ (enumerated exactly, $|\mathcal C|^n\le 9^5$) and over cue reliability $w\in W=\{0,\tfrac12,1\}$,
given the world's evidence $\mathcal D_t=\{(a_s,o_s)\}_{s<t}$.

### 2.3 Generative model used by the learner
Priors (Beta pseudo-counts). $\bar r$ is the mean posterior reaction rate over the library's concept pairs, each
pair counted once (R1). Attempt 1 pooled all of the agent's outcomes, which its own exploitation biased up to ≈ 0.78
against a true rate ≈ 0.18.
$$\theta^{\text{react}}_{kl}\sim\mathrm{Beta}(\bar r+S_{kl},\,1-\bar r+F_{kl}),\qquad
\theta^{\text{bit}}_{kb}\sim\mathrm{Beta}(\tfrac12+G^+_{kb},\,\tfrac12+G^-_{kb}),$$
with zero counts for any pair or concept involving a fresh label.
Membership prior (Chinese-restaurant style, independent per entity): $P(z_e=k)\propto N_k$, $P(z_e=\nu_m)\propto \alpha/M$.
Appearance likelihood with reliability $w$:
$$P(s_{ed}\mid z_e=k,w)=w\,q_{kd}^{s_{ed}}(1-q_{kd})^{1-s_{ed}}+(1-w)\tfrac12,\qquad q_{kd}=\frac{1+A^+_{kd}}{2+A^+_{kd}+A^-_{kd}},$$
where $s_{ed}=\mathbb 1[x_{ed}>0]$ and $q_{\nu d}=\tfrac12$.

Posterior (exact, by enumeration):
$$P(z,w\mid\mathcal D_t,x)\propto P(w)\prod_e P(z_e)\prod_{e,d}P(s_{ed}\mid z_e,w)\;\prod_{(k,l)}\frac{B(\alpha_{kl}+S^{z}_{kl},\,\beta_{kl}+F^{z}_{kl})}{B(\alpha_{kl},\beta_{kl})}\prod_{k}\frac{B(\alpha^g_k+G^{z}_k,\,\beta^g_k+H^{z}_k)}{B(\alpha^g_k,\beta^g_k)}$$
where $S^z_{kl},F^z_{kl}$ ($G^z_{kb},H^z_{kb}$) are the world's reaction (probe-bit) outcomes aggregated by the concept pair
(concept) that $z$ assigns to each action. Laws for unseen concept pairs are thereby learned *within* the world.

### 2.4 Acting: experiments that discriminate between concept assignments
Predictive success probability of action $a$: $\hat p_a=\sum_{z}P(z\mid\cdot)\,\mathbb E[\theta_{c(a,z)}\mid z,\mathcal D_t]$.

Information gain is the mutual information between the outcome and the assignment **and** the laws (R1):
$$\mathrm{IG}_a=H(\hat p_a)-\sum_z P(z\mid\cdot)\,\mathbb E_{\theta\sim\mathrm{Beta}(\alpha_{c},\beta_{c})}\big[H(o\mid\theta)\big],\qquad
\mathbb E[H_b(\theta)]=\psi(\alpha+\beta+1)-\tfrac{\alpha}{\alpha+\beta}\psi(\alpha+1)-\tfrac{\beta}{\alpha+\beta}\psi(\beta+1).$$
For a probe, $o$ is three bits that are independent given $(z,\theta)$, so the expected conditional entropy is a sum
over bits; $H(\hat p_a)$ is the entropy of the 8-outcome predictive.

**Attempt 1 differed, and this was a defect.** It used $H_b(\mathbb E[\theta\mid z])$, i.e. information about $z$
only. With no probe counts in the library, every assignment predicts 0.5 per bit, so a probe's information gain is
exactly 0. The agent therefore never probed (critique F1).
Action: $a_t=\arg\max_a\big(r(a)\,\hat p_a+\beta_t\,\mathrm{IG}_a\big)$ with $r(\mathrm{pair})=1$, $r(\mathrm{poke})=0$,
$\beta_t=\beta(1-t/T)$.

### 2.5 Contradiction-triggered re-identification
Surprise of an outcome $o$ of action $a$: $u=-\log P(o\mid\mathcal D_t)$. For each entity, if the summed surprise of its
last two observations exceeds $\tau$, the entity's earlier evidence is discarded and its concept is re-inferred
(the learner treats the contradiction as a change in that entity).

### 2.6 Consolidation after a world (learning across worlds)
$z^\star=\arg\max_z P(z\mid\mathcal D_T,x)$. Fresh labels used in $z^\star$ become new library concepts; the world's
non-discarded counts are added to the concepts and concept pairs $z^\star$ assigns (entities with at least one
observation).

**Re-clustering (added in development, D-2).** Additive MAP consolidation alone collapsed the library. During
training, every 50 worlds the library is therefore rebuilt by collapsed Gibbs sampling over the concept of every
stored entity under the same model. This is the infinite relational model of Kemp et al. (2006), applied to the
agent's own stored worlds. The library is consequently *not* purely additive during training. Only the
accumulation test uses the additive path alone.

### 2.7 Ablations (each removes one mechanism, keeping the rest)
| ablation | change |
|---|---|
| `ira_noactive` | $\beta=0$, otherwise identical (R1; attempt 1 used Thompson sampling of $z$, now `ira_thompson`) |
| `ira_closed` | $M=0$: no fresh labels (closed concept set) |
| `ira_nocuetrack` | $w$ fixed at 1 (perceptual cues trusted as learned) |
| `ira_nochange` | no contradiction-triggered re-identification |
| `entity_bayes` | no concepts: a Beta-Bernoulli law per action with the library's pooled rate as prior, Thompson sampling |

## 3. What the formulation predicts

* Cue reversal: with $w$ inferred, consequences overrule appearance within a few outcomes; with $w=1$ the learner is
  misled; a learner that associated appearance with value during training (as a meta-learner may) is misled too.
* Recombination: concept-pair laws compose regardless of which concepts co-occurred during training.
* Unseen concept: a fresh label wins the posterior once outcomes contradict every known concept; its laws are learned
  within the world and consolidated afterwards.
* Change: the surprise rule re-identifies the changed entity.
* Accumulation: consolidated concepts make later worlds with the same unseen concept easier, without retraining.

## 4. Falsifiers
Fixed in `EXPERIMENTS_AND_BASELINES.md` §4 before the confirmatory run.
