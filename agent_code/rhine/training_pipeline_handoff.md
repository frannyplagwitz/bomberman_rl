# Training Pipeline Handoff — model.py / ppo.py / train.py

These three files (`model.py`, `ppo.py`, `train.py`) are the training
infrastructure I built for my PPO agent, factored out so you can plug in your
own feature engineering, reward function, and action mask without touching
the training logic itself.

`ppo.py` is the PPO algorithm proper — rollout buffer, GAE, and the
clipped-objective update — and can be used as-is, with no changes needed, as
long as your model exposes a `forward(features) -> (action_logits, value)`
interface (see `model.py`'s `ActorCriticMLP` for the expected shape).

`train.py` is the training-loop orchestration (when to store a transition,
when to trigger a PPO update, when to save a checkpoint); it no longer makes
any assumption about how reward is computed — you just need to provide your
own `compute_reward(old_game_state, self_action, new_game_state, events,
reward_config) -> float`, plus your own feature/mask/state-extraction
modules (see the table below for the full list of what `train.py` expects
from each).

`model.py` is the small MLP I used for my feature vector; if you're going
with a CNN or a lookup-table model instead, treat this as a reference only
(the `masked_logits()` helper is architecture-agnostic and reusable
regardless), not something to reuse directly — and note that a
non-differentiable lookup table won't be compatible with `ppo.py`'s
gradient-based update at all, only a differentiable model (MLP, CNN, or an
embedding-based lookup) will work with this training loop.

## What `train.py` needs from you

`train.py` imports five things by name via relative imports (`from . import
config`, `from .features import ...`, etc.), so it expects to live inside its
own `agent_code/<your_agent_name>/` folder alongside a `config.py`,
`features.py`, `action_mask.py`, `rewards.py`, and `state_processing.py` that
you write yourself. `model.py` and `ppo.py` can just sit in that same folder
unchanged (or even be imported straight from my agent's folder — up to you).

| You implement | Called as | Contract |
|---|---|---|
| `config.PPO_CONFIG` | `cfg.PPO_CONFIG` | needs `.learning_rate`, `.gamma`, `.gae_lambda`, `.clip_range`, `.entropy_coef`, `.value_loss_coef`, `.max_grad_norm`, `.rollout_steps`, `.minibatch_size`, `.update_epochs`, `.normalize_advantage` — these are the fields `ppo.py` reads |
| `config.REWARD_CONFIG` | `cfg.REWARD_CONFIG` | whatever shape you want; it's just passed straight through to your own `compute_reward` |
| `config.ACTIONS` | `cfg.ACTIONS.index(self_action)` | list of the 6 action name strings, used to turn an action string into an index |
| `config.MODELS_DIR` / a checkpoint-path constant | used when saving checkpoints | just needs to be a valid path |
| `state_processing.extract_semantic_state(game_state) -> anything` | called once per step | the return value gets passed straight into both of the next two functions — it can be literally any object (a dataclass, a dict, or even `game_state` itself if you don't want an intermediate representation) |
| `features.features_from_semantic(semantic) -> np.ndarray` | feeds the model | shape must match whatever `n_features` your model expects |
| `action_mask.mask_from_semantic(semantic) -> np.ndarray[bool]` | length 6 | one bool per action, in the same order as `config.ACTIONS`; this is fixed by the game itself, not something you need to redesign |
| `rewards.compute_reward(old_game_state, self_action, new_game_state, events, reward_config) -> float` | called once per step | `new_game_state` can be `None` on a defensive terminal-transition fallback path — handle that case |

One thing worth knowing: `train.py` calls `extract_semantic_state` once and
feeds the *same* result into both `features_from_semantic` and
`mask_from_semantic`. If you'd rather keep feature extraction and mask
computation completely separate (e.g. a CNN reading straight off the raw
board, no shared intermediate representation), `extract_semantic_state` can
just be a pass-through that returns `game_state` unchanged — Python doesn't
care what shape that object is.

## About the Lookup idea

If your "Lookup" model means a classic, non-differentiable tabular Q-table
(a plain dict/array keyed by discretized state, updated via Q-learning-style
rules), then **`ppo.py` doesn't apply to it at all**, not just `model.py`.
PPO's update step needs `model.parameters()` and `loss.backward()`, which a
plain lookup table doesn't have. In that case, only `train.py`'s scheduling
logic (when to save checkpoints, the overall step-by-step structure) is worth
skimming as a reference, and you'd write your own update rule instead of
using `ppo.py`.

If instead you wrap the lookup table in something differentiable (e.g. an
`nn.Embedding`-based table that still exposes a `forward(features) ->
(logits, value)` interface), then it's just another kind of "model" and the
whole `ppo.py` + `train.py` pipeline works with it unchanged — same as the
CNN case.
