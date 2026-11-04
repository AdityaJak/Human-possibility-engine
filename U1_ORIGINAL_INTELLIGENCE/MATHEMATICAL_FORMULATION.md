# U1 — Mathematical Formulation

## 1. Environment class: hidden-law entity worlds

A **universe** fixes hidden causal laws shared by all worlds: a set of roles $\mathcal R=\{0,\dots,R-1\}$
($R=6$; roles $0$–$4$ occur in training, role $5$ only in the unseen-concept condition), a glow propensity
$g_r\in\{0.1,0.9\}$ per role, and a symmetric reaction law $L\in\{0,1\}^{R\times R}$.

A **world** (one episode) has $n=5$ entities. Entity $e$ has a hidden role $r_e$ and an observed appearance
$x_e\in\mathbb R^{4}$. Appearance is fresh noise in every world, $x_{e}\sim\mathcal N(0,I)$, except that one dimension
carries a **cue**: $x_{e,0}=\pm 2$ with sign $+$ for roles $\{0,1\}$ and $-$ otherwise, with probability $\rho$ (else the
sign is flipped). The learner is not told which dimension is the cue.

Actions $a\in\mathcal A$, $|\mathcal A|=15$: $\mathrm{poke}(e)$ for each entity (outcome glow $\sim
\mathrm{Bern}(g_{r_e})$, reward 0) and $\mathrm{pair}(i,j)$ for each unordered pair (outcome reaction $\sim
\mathrm{Bern}(p_{hi})$ if $L_{r_ir_j}=1$ else $\mathrm{Bern}(p_{lo})$, reward = outcome; $p_{hi}=0.9$, $p_{lo}=0.05$).
Horizon $T=20$. The learner observes $x$ and, after each action, the binary outcome and reward. It never observes
roles, $g$, $L$, $\rho$, the cue dimension or the condition.

Conditions (§ `EXPERIMENTS_AND_BASELINES.md`): training-like (2–3 distinct known roles, $\rho=0.9$); cue reversed
($\rho=0.1$); recombined (4–5 distinct known roles); unseen concept (the only reacting pairs involve role 5); role
switch (one entity's role changes at $t=10$).

## 2. The learner (IRA: interventional role abstraction)

### 2.1 Persistent knowledge (the library)
$\mathcal K=\{1,\dots,K\}$ concepts, each with sufficient statistics accumulated over past worlds:
* glow counts $(G^+_k,G^-_k)$;
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
Priors (Beta pseudo-counts; $\bar r$ = pooled reaction rate of the library):
$$\theta^{\text{react}}_{kl}\sim\mathrm{Beta}(\bar r+S_{kl},\,1-\bar r+F_{kl}),\qquad
\theta^{\text{glow}}_{k}\sim\mathrm{Beta}(\tfrac12+G^+_k,\,\tfrac12+G^-_k),$$
with zero counts for any pair or concept involving a fresh label.
Membership prior (Chinese-restaurant style, independent per entity): $P(z_e=k)\propto N_k$, $P(z_e=\nu_m)\propto \alpha/M$.
Appearance likelihood with reliability $w$:
$$P(s_{ed}\mid z_e=k,w)=w\,q_{kd}^{s_{ed}}(1-q_{kd})^{1-s_{ed}}+(1-w)\tfrac12,\qquad q_{kd}=\frac{1+A^+_{kd}}{2+A^+_{kd}+A^-_{kd}},$$
where $s_{ed}=\mathbb 1[x_{ed}>0]$ and $q_{\nu d}=\tfrac12$.

Posterior (exact, by enumeration):
$$P(z,w\mid\mathcal D_t,x)\propto P(w)\prod_e P(z_e)\prod_{e,d}P(s_{ed}\mid z_e,w)\;\prod_{(k,l)}\frac{B(\alpha_{kl}+S^{z}_{kl},\,\beta_{kl}+F^{z}_{kl})}{B(\alpha_{kl},\beta_{kl})}\prod_{k}\frac{B(\alpha^g_k+G^{z}_k,\,\beta^g_k+H^{z}_k)}{B(\alpha^g_k,\beta^g_k)}$$
where $S^z_{kl},F^z_{kl}$ ($G^z_k,H^z_k$) are the world's reaction (glow) outcomes aggregated by the concept pair
(concept) that $z$ assigns to each action. Laws for unseen concept pairs are thereby learned *within* the world.

### 2.4 Acting: experiments that discriminate between concept assignments
Predictive success probability of action $a$: $\hat p_a=\sum_{z}P(z\mid\cdot)\,\mathbb E[\theta_{c(a,z)}\mid z,\mathcal D_t]$.
Information gain (mutual information between the outcome and the assignment/law posterior):
$$\mathrm{IG}_a=H_b(\hat p_a)-\sum_z P(z\mid\cdot)\,H_b\big(\mathbb E[\theta_{c(a,z)}\mid z,\mathcal D_t]\big).$$
Action: $a_t=\arg\max_a\big(r(a)\,\hat p_a+\beta_t\,\mathrm{IG}_a\big)$ with $r(\mathrm{pair})=1$, $r(\mathrm{poke})=0$,
$\beta_t=\beta(1-t/T)$.

### 2.5 Contradiction-triggered re-identification
Surprise of an outcome $o$ of action $a$: $u=-\log P(o\mid\mathcal D_t)$. For each entity, if the summed surprise of its
last two observations exceeds $\tau$, the entity's earlier evidence is discarded and its concept is re-inferred
(the learner treats the contradiction as a change in that entity).

### 2.6 Consolidation after a world (learning across worlds)
$z^\star=\arg\max_z P(z\mid\mathcal D_T,x)$. Fresh labels used in $z^\star$ become new library concepts; all of the
world's counts are added to the concepts and concept pairs $z^\star$ assigns.

### 2.7 Ablations (each removes one mechanism, keeping the rest)
| ablation | change |
|---|---|
| `ira_noactive` | $\beta=0$; actions by Thompson sampling of $z$ (passive identification) |
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
