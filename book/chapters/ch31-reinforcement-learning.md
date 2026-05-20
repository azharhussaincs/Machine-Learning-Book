# Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is the most distinctive paradigm in Machine Learning. There's
no dataset of correct answers. Instead, an **agent** learns by **doing** — taking actions
in an **environment**, receiving **rewards** or **penalties**, and gradually figuring out
a strategy that maximises its long-term reward. It's how you'd train a dog with treats, how
a child learns to ride a bike, and how AlphaGo learned to beat the world's best Go players.

RL powers some of AI's most spectacular achievements: mastering Atari games from raw
pixels, beating champions at Go and StarCraft, controlling robots, optimising data
centres, and — crucially — fine-tuning chatbots with human feedback (RLHF, Chapter 39).

::: keyidea
RL learns from **interaction and reward**, not from labelled examples. The agent must
balance **exploring** (trying new actions to discover good ones) against **exploiting**
(using what it already knows). Its goal is a **policy** — a strategy mapping situations to
actions — that maximises **long-term** reward, not just the immediate one.
:::

By the end of this chapter you will be able to:

- Explain the RL loop and its vocabulary (agent, environment, state, action, reward,
  policy).
- Understand **return**, the **discount factor**, and **Q-values**.
- Understand the **exploration vs exploitation** trade-off.
- Understand and implement **Q-learning** from scratch.
- Know about Deep RL and where RL is used.

## The reinforcement learning loop

Recall the loop from Chapter 4: the agent observes the **state**, takes an **action**, and
the environment returns a **reward** and a **new state**. This repeats, and the agent
learns from the stream of rewards.

![The reinforcement learning loop. The agent observes the state, chooses an action via its policy, and the environment responds with a reward and the next state. Over many steps the agent learns the policy that maximises long-term reward.](assets/images/ch31_rl_loop.png)

**Vocabulary:**

- **Agent** — the learner/decision-maker.
- **Environment** — the world the agent acts in.
- **State (s)** — the current situation.
- **Action (a)** — a choice available to the agent.
- **Reward (r)** — immediate numeric feedback.
- **Policy (π)** — the agent's strategy: which action to take in each state.
- **Episode** — one full run from start to a terminal state (e.g. one game).

## Return and the discount factor

The agent doesn't maximise the *immediate* reward — it maximises the **return**, the total
(discounted) future reward:

<div class="equation"><img class="eq" src="assets/images/eq_ch31_return.png" alt="discounted return"></div>

The **discount factor γ** (gamma, between 0 and 1) makes future rewards worth slightly less
than immediate ones — like money, a reward now is worth more than the same reward later. A
γ near 1 makes the agent far-sighted (values the long term); near 0 makes it short-sighted.

::: note
The discount factor is what makes RL care about **long-term** consequences. A move that
gives no immediate reward but sets up a winning position later still gets value, because
its future rewards flow back through γ. This is why RL can learn strategy, not just greedy
grabs.
:::

## Exploration vs exploitation

A fundamental dilemma: should the agent **exploit** the best action it knows, or **explore**
a new action that *might* be even better? Too much exploitation and it gets stuck in a
mediocre habit; too much exploration and it never settles on a good strategy.

![Exploration vs exploitation: exploit the known-good restaurant, or explore a new one that might be better? The ε-greedy strategy mostly exploits but explores randomly a small fraction (ε) of the time.](assets/images/ch31_explore.png)

The common solution is **ε-greedy**: with probability **ε** (e.g. 0.1) take a random action
(explore), otherwise take the best-known action (exploit). Often ε starts high and decays
as the agent learns.

::: warning
**Exploration is not optional.** Our first code attempt below (a corridor where the agent
starts at one end) initially *failed* — with weak exploration and a greedy bias, the agent
never reached the goal, so it learned nothing (all values stayed zero). Only after
improving exploration did it learn. **If your RL agent isn't learning, suspect
insufficient exploration.**
:::

## Q-learning

**Q-learning** is a foundational RL algorithm. It learns a **Q-value** `Q(s, a)` — the
expected long-term return of taking action `a` in state `s` and behaving optimally
afterward. Once learned, the best policy is simply: **in each state, pick the action with
the highest Q-value.**

Q-values are learned by repeatedly applying the **Q-learning update rule** (a form of the
Bellman equation):

<div class="equation"><img class="eq" src="assets/images/eq_ch31_qlearning.png" alt="Q-learning update"></div>

In words: nudge `Q(s,a)` toward the **reward just received** plus the **discounted best
future value** of the next state. Here **α** is the learning rate and **γ** the discount
factor. The bracketed term is the **temporal-difference error** — the gap between the new
estimate and the old one.

## Practical: Q-learning from scratch

Let's teach an agent to navigate a simple corridor of 6 states (0–5), where state 5 is the
**goal** (reward +1). Actions: move left or right. The agent must learn to go right.

```python
import numpy as np

n_states, goal = 6, 5
Q = np.zeros((n_states, 2))          # Q[state, action]; action 0=left, 1=right
alpha, gamma, eps = 0.1, 0.9, 0.3    # learning rate, discount, exploration rate
rng = np.random.default_rng(0)

for episode in range(2000):
    s = rng.integers(0, goal)        # random start (ensures the agent explores all states)
    for _ in range(50):
        # ε-greedy action choice
        a = rng.integers(2) if rng.random() < eps else int(np.argmax(Q[s]))
        s2 = max(0, s - 1) if a == 0 else min(goal, s + 1)   # take the action
        r = 1.0 if s2 == goal else 0.0                       # reward only at the goal
        # Q-learning update
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
        s = s2
        if s == goal:
            break

print("Learned Q-table (rounded):")
print(np.round(Q, 2))
policy = ["RIGHT" if np.argmax(Q[s]) == 1 else "LEFT" for s in range(goal)]
print("Learned policy (states 0-4):", policy)
```

**Output:**
```text
Learned Q-table (rounded):
[[0.59 0.66]
 [0.59 0.73]
 [0.66 0.81]
 [0.73 0.9 ]
 [0.81 1.  ]
 [0.   0.  ]]
Learned policy (states 0-4): ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT']
```

### Explanation

- The agent learned the correct policy: **always go RIGHT** to reach the goal.
- Look at the Q-values for the "right" action: **1.0** at state 4 (one step from the goal),
  then **0.9, 0.81, 0.73, 0.66** as states get farther — each discounted by γ=0.9. The
  reward at the goal **propagated backward** through the corridor, exactly as the Bellman
  update intends.
- Note we used **random start states** and **ε=0.3 exploration** — without enough
  exploration (our first attempt), the agent never found the goal and learned nothing.

::: keyidea
You just built a complete RL agent from scratch. The pattern — *act (ε-greedy) → observe
reward and next state → update Q toward reward + discounted future value → repeat* — is the
heart of value-based RL. Deep RL replaces the Q-table with a neural network, but the core
idea is identical.
:::

## Deep Reinforcement Learning

A Q-*table* only works for small, discrete state spaces. Real problems (Atari pixels, robot
sensors) have astronomically many states. **Deep Reinforcement Learning** replaces the
table with a **neural network** that *predicts* Q-values from the state:

- **DQN (Deep Q-Network)** — a neural net learns Q-values; famously mastered Atari games
  from raw pixels (2013–2015).
- **Policy gradient / Actor-Critic methods** (REINFORCE, A2C, PPO) — directly learn the
  policy; used for robotics and continuous control.
- **AlphaGo / AlphaZero** — combined deep RL with tree search to beat world champions.
- **RLHF (Reinforcement Learning from Human Feedback)** — uses human preferences as the
  reward to align Large Language Models (Chapter 39).

## Advantages, disadvantages, and use cases

| Advantages | Disadvantages |
|---|---|
| Learns sequential decision-making | Needs many interactions (sample-inefficient) |
| No labelled dataset required | Training is slow and unstable |
| Can discover novel strategies | Reward design is tricky (reward hacking) |
| Optimises long-term goals | Hard to debug; exploration is delicate |

**Use cases:** game-playing AI, robotics and control, autonomous vehicles, recommendation
systems (long-term engagement), resource/energy optimisation, finance, and aligning LLMs
(RLHF).

## Common mistakes & misconceptions

::: warning
**Mistake 1 — Too little exploration.** The classic failure: the agent never discovers the
reward and learns nothing (as our first attempt showed). Ensure adequate ε / exploration.
:::

- **Mistake 2 — Poor reward design** — agents exploit loopholes ("reward hacking") to get
  reward without doing the intended task.
- **Mistake 3 — Using RL for problems that are really supervised learning** — RL is for
  *sequential decisions with feedback*, and is far harder; don't use it unnecessarily.
- **Mistake 4 — Expecting fast training** — RL is sample-inefficient and often slow/unstable.
- **Mistake 5 — Ignoring the discount factor's effect** on far- vs short-sightedness.
- **Mistake 6 — Forgetting that a Q-table doesn't scale** — large state spaces need Deep RL.

## Best practices

- **Balance exploration and exploitation** (ε-greedy, often with decay).
- **Design rewards carefully** to reflect the true goal and avoid loopholes.
- **Use RL only for genuine sequential-decision problems.**
- **Use Deep RL** (function approximation) for large/continuous state spaces.
- **Tune the discount factor** for the desired time horizon.
- **Expect slow, noisy training**; use proven libraries (Gymnasium, Stable-Baselines3) for
  real work.

## Chapter Summary

- **Reinforcement Learning** trains an **agent** to maximise long-term **reward** by
  interacting with an **environment** — taking **actions**, observing **states** and
  **rewards**, and learning a **policy**. No labelled dataset is needed.
- The agent maximises the **return** (discounted future reward), with the **discount factor
  γ** controlling far- vs short-sightedness, and must balance **exploration vs
  exploitation** (e.g. ε-greedy).
- **Q-learning** learns **Q-values** (long-term value of state-action pairs) via the Bellman
  update `Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') − Q(s,a)]`; the best policy picks the
  highest-Q action.
- We built a Q-learning agent from scratch that learned to traverse a corridor — and saw
  first-hand that **insufficient exploration breaks RL**.
- **Deep RL** replaces the Q-table with a neural network (DQN, policy gradients), powering
  Atari, AlphaGo, robotics, and **RLHF** for LLMs.

---

::: {.qband}
Practice Zone — Chapter 31
:::

## Multiple-Choice Questions (MCQs)

**Q1.** In RL, the agent learns from:
a) Labelled examples  b) Rewards from interacting with an environment  c) Clusters  d)
Unlabelled data only

**Q2.** The agent's strategy mapping states to actions is the:
a) Reward  b) Policy  c) State  d) Episode

**Q3.** The discount factor γ controls:
a) The learning rate  b) How much future rewards are valued  c) Exploration  d) The number
of states

**Q4.** ε-greedy is used to balance:
a) Bias and variance  b) Exploration and exploitation  c) Precision and recall  d) Train and
test

**Q5.** Q(s, a) represents the:
a) Immediate reward  b) Expected long-term return of taking a in s  c) Number of states
d) Policy

**Q6.** If an RL agent never reaches the goal and learns nothing, the likely cause is:
a) Too much data  b) Insufficient exploration  c) Too many features  d) Scaling

**Q7.** Deep RL replaces the Q-table with a:
a) Decision tree  b) Neural network  c) Cluster  d) Dataset

**Q8.** RLHF is used to:
a) Cluster data  b) Align LLMs using human-preference rewards  c) Reduce dimensions  d)
Detect outliers

### MCQ Answers
**1:** b. **2:** b. **3:** b. **4:** b. **5:** b. **6:** b. **7:** b. **8:** b.

## Interview Questions (with answers)

**Q1. How does reinforcement learning differ from supervised learning?**
*Answer:* Supervised learning trains on a fixed dataset of input-output pairs to map inputs
to known correct outputs. RL has no labelled answers; an agent interacts with an
environment, receiving rewards for its actions, and learns a policy that maximises
long-term reward through trial and error. RL handles sequential decision-making, whereas
supervised learning predicts from static examples.

**Q2. Explain the exploration vs exploitation trade-off.**
*Answer:* Exploitation uses the best action known so far to get reward now; exploration
tries other actions to discover potentially better ones. Too much exploitation risks
getting stuck in a suboptimal habit; too much exploration wastes reward and never converges.
ε-greedy balances them by exploring with small probability ε and exploiting otherwise,
often decaying ε over time.

**Q3. What is a Q-value and how is it learned?**
*Answer:* Q(s,a) is the expected long-term (discounted) return of taking action a in state
s and acting optimally afterward. It's learned by the Q-learning update, which nudges
Q(s,a) toward the observed reward plus the discounted maximum Q-value of the next state —
propagating future rewards backward through states.

**Q4. What role does the discount factor play?**
*Answer:* The discount factor γ (0–1) weights future rewards relative to immediate ones. A
γ near 1 makes the agent far-sighted (values long-term outcomes), near 0 makes it myopic
(focuses on immediate reward). It lets RL value actions whose payoff comes later, enabling
strategic behaviour.

**Q5. Why is reward design important and risky?**
*Answer:* The agent optimises whatever the reward specifies, not what you intended. A
poorly designed reward can be "hacked" — the agent finds loopholes that earn reward without
achieving the real goal. Careful, aligned reward design is essential and notoriously
difficult.

## Scenario-Based Questions (with answers)

**Q1.** *You're training a game-playing agent and it isn't improving at all — its value
estimates stay near zero. What's the most likely problem?*
*Answer:* Insufficient exploration — the agent never reaches the rewarding states, so no
reward signal propagates back. Increase ε (or use random/varied start states, optimistic
initialisation, or reward shaping) so the agent discovers the reward, then learning can
begin.

**Q2.** *Your cleaning-robot RL agent learns to bump into walls repeatedly because you gave
it +1 for each "movement". What went wrong?*
*Answer:* Reward hacking due to poor reward design — it maximises movement reward without
cleaning. Redesign the reward to reflect the true objective (e.g. reward area cleaned,
penalise collisions and time), aligning incentives with the intended task.

**Q3.** *A colleague wants to use RL to predict house prices from features. Is RL
appropriate?*
*Answer:* No — that's a static supervised regression problem (Chapter 17), not a sequential
decision process with feedback. RL would be far more complex and sample-inefficient for no
benefit. Use RL only for problems involving sequences of actions and rewards.

## Logic-Based Questions (with answers)

**Q1.** In the corridor, why does Q for "right" equal 1.0 at state 4 but only ~0.66 at
state 0?
*Answer:* State 4 is one step from the goal (reward 1), so its right-action value is the
full reward. Farther states' values are the goal reward discounted by γ for each extra step
(0.9, 0.81, …), so state 0's value is much smaller — future reward is worth less the
farther away it is.

**Q2.** Why does maximising immediate reward sometimes lead to worse long-term outcomes?
*Answer:* Because a greedy choice now can forfeit larger future rewards (e.g. a chess move
that grabs a pawn but loses the game). The discounted return accounts for the future, so RL
can prefer actions with no immediate reward that lead to better long-term outcomes.

**Q3.** Why must a Q-table be abandoned for problems like Atari from pixels?
*Answer:* The number of distinct pixel states is astronomically large, so a table can't
store or learn a value for each. A neural network (Deep Q-Network) instead *generalises* —
predicting Q-values for unseen states from learned patterns.

## Practical Questions (with answers)

**Q1.** In the Q-learning update, what does the term `gamma * np.max(Q[s2])` represent?
*Answer:* The discounted estimate of the best achievable future return from the next state
s2 — the agent's current belief about how good it is to be in s2 if it acts optimally
afterward.

**Q2.** Why did using random start states help the agent learn?
*Answer:* It ensures the agent sometimes starts near the goal, discovers the reward, and
propagates value backward to earlier states — overcoming the weak exploration that left the
all-zero Q-table in the first attempt.

**Q3.** What does ε control, and what happens if ε = 0?
*Answer:* ε is the exploration probability. With ε = 0 the agent never explores (pure
exploitation), so if its initial greedy actions never reach reward, it can get stuck and
never learn — exactly the failure mode to avoid.

## Long Questions (with answers)

**Q1. Explain the reinforcement learning framework and the Q-learning algorithm, including
how rewards propagate and the role of exploration.**

*Answer:* In **reinforcement learning**, an **agent** interacts with an **environment** over
discrete steps. At each step it observes a **state** s, selects an **action** a according to
its **policy**, and the environment returns a **reward** r and a new state s′. The agent's
objective is to maximise the **return** — the sum of future rewards discounted by γ — so it
values long-term outcomes, not just immediate reward. **Q-learning** learns a **Q-value**
Q(s,a), the expected return of taking a in s and acting optimally thereafter, via the update
Q(s,a) ← Q(s,a) + α[r + γ·maxₐ′Q(s′,a′) − Q(s,a)]: it nudges the current estimate toward the
reward just received plus the discounted best value of the next state, with learning rate α.
Through repeated updates, reward earned at terminal/goal states **propagates backward**:
the state next to the goal gets the full reward, the state before it gets that discounted by
γ, and so on — exactly as seen in the corridor where right-action values were 1.0, 0.9,
0.81, 0.73, 0.66 receding from the goal. Crucially, the agent must **explore** to discover
rewards; using an ε-greedy policy (random action with probability ε, best-known action
otherwise) ensures it tries enough actions. Without sufficient exploration the agent may
never reach reward and learn nothing — the failure observed before random starts and higher
ε were used. The optimal policy is then simply to pick, in each state, the action with the
highest learned Q-value.

**Q2. Discuss where reinforcement learning excels, its challenges, and how Deep RL extends
it, with real examples.**

*Answer:* RL **excels** at **sequential decision-making** problems where an agent must take a
series of actions to achieve a long-term goal and where labelled "correct" answers don't
exist — it can even discover novel strategies humans never taught it. Landmark successes
include **DQN** mastering Atari games directly from pixels, **AlphaGo/AlphaZero** defeating
world champions at Go and chess by combining deep RL with search, robotics and continuous
control via **policy-gradient/actor-critic** methods (e.g. PPO), data-centre energy
optimisation, and **RLHF**, which aligns Large Language Models using human-preference rewards
(Chapter 39). Its **challenges** are significant: RL is **sample-inefficient** (it may need
millions of interactions), training is **slow and unstable**, **reward design is hard** and
prone to "reward hacking", and **exploration** is delicate — too little and the agent never
finds reward, too much and it never converges. **Deep RL** extends classical RL by replacing
the Q-table (which only works for small, discrete state spaces) with a **neural network**
that approximates Q-values or the policy directly from raw, high-dimensional states like
images or sensor readings; this **generalisation** is what lets RL scale to real-world
complexity. The trade-off is added instability and compute cost, mitigated in practice by
techniques like experience replay, target networks, and proven libraries — but the core loop
of act, observe reward, and update toward reward-plus-discounted-future-value remains the
same as the simple Q-learning agent built in this chapter.

## Exercises

1. Define agent, environment, state, action, reward, and policy in your own words.
2. Explain the exploration vs exploitation trade-off with a real-life example.
3. What does the discount factor γ do, and what happens as γ → 0 vs γ → 1?
4. In the corridor example, explain why Q-values decrease as states get farther from the
   goal.
5. Give two real applications of reinforcement learning.

## Mini-Project

**Project: Solve a small gridworld with Q-learning.**

1. Build a small grid environment (e.g. 4×4 with a goal cell giving +1 and maybe a pit
   giving −1).
2. Implement Q-learning from scratch (Q-table, ε-greedy, the update rule).
3. Plot the total reward per episode to see learning improve over time (Chapter 14).
4. Visualise the learned policy (best action arrow in each cell).
5. Experiment with ε and γ and report their effects. Save in `my-ml-journey/`.

## Assignments

1. **Coding:** Extend the corridor agent to add a penalty (−1) at one end and show the
   policy changes to avoid it.
2. **Coding:** (Optional `pip install gymnasium`) Train a Q-learning agent on `FrozenLake`
   and report the success rate before and after training.
3. **Conceptual:** Write one page on reward design: why it's hard, with an example of reward
   hacking and how you'd fix it.

::: tip
**Part V complete!** You now understand learning without full labels — clustering,
dimensionality reduction, association rules, semi-supervised, and reinforcement learning.
**Part VI** opens the most transformative area of modern AI: **Deep Learning** — neural
networks that power vision, language, and generative AI.
:::
