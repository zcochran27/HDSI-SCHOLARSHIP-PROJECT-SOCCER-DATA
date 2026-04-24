# Defensive Duel Value

A location-aware metric that assigns each defensive duel a scalar value expressed in units of **goal probability averted (or conceded) relative to expectation**. Positive means the defender outperformed the prior at that pitch location; negative means they underperformed.

## 1. Outcomes

Every defensive duel is labelled with exactly one of three mutually exclusive ordinal outcomes:

| Outcome      | Meaning                                             | Ordinal rank |
|--------------|-----------------------------------------------------|--------------|
| `recovered`  | Defender wins possession                             | best         |
| `stopped`    | Defender halts attacker progress without recovery    | middle       |
| `neither`    | Defender is beat — attacker retains possession and progress | worst        |

The raw data provides two overlapping booleans (`groundDuel_recoveredPossession`, `groundDuel_stoppedProgress`). They are collapsed to a single categorical with priority `recovered > stopped > neither`:

```python
duel_outcome = np.select(
    [recovered, stopped],
    ["recovered", "stopped"],
    default="neither",
)
```

## 2. Two models

Both are LightGBM, fit on defensive duels starting in the defensive half (`start_x < 60` in StatsBomb coordinates).

### 2.1 Outcome model

$$p(c \mid x, y) \quad \text{for } c \in \{\text{recovered}, \text{stopped}, \text{neither}\}$$

Multiclass classifier with features `(start_x, start_y)`. Produces a probability simplex per duel.

### 2.2 Conditional goal model

$$p(\text{goal} \mid x, y, c)$$

Binary classifier where `goal = 1` iff `outcome == -1` (the VAEP-style label for "goal conceded within the action window"). Features are `(start_x, start_y, c)` with `c` flagged as a LightGBM categorical feature so the split is on class identity, not on ordinal encoding. The positive class is extremely rare, so `scale_pos_weight = n_neg / n_pos` is applied; this distorts calibration but not ranking, which is fine for our purposes because the value formula uses *differences* of predicted probabilities.

## 3. The value formula

### 3.1 Starting proposal

Per-duel value was initially proposed as a sum over outcomes of *surprise* times *outcome-conditional danger*:

$$v(\text{duel}) = \sum_{c} \big(\mathbb{1}[c^\star = c] - p(c \mid x, y)\big) \cdot p(\text{goal} \mid c, x, y)$$

where $c^\star$ is the observed outcome of this particular duel. The indicator $\mathbb{1}[c^\star = c]$ equals 1 if the observed class matches, 0 otherwise. Since $\sum_c \mathbb{1}[c^\star = c] = 1$ and $\sum_c p(c \mid x, y) = 1$, the surprise terms sum to zero across classes — the formula is a centred weighting.

### 3.2 Algebraic simplification

Distribute the sum:

$$v = \sum_c \mathbb{1}[c^\star = c]\, p(\text{goal} \mid c, x, y) \;-\; \sum_c p(c \mid x, y)\, p(\text{goal} \mid c, x, y)$$

The first sum collapses to the single term where $c = c^\star$:

$$\sum_c \mathbb{1}[c^\star = c]\, p(\text{goal} \mid c, x, y) = p(\text{goal} \mid c^\star, x, y)$$

The second sum is exactly the marginalization of the conditional goal model over the outcome prior — by the law of total probability it equals the unconditional goal probability at that location:

$$\sum_c p(c \mid x, y)\, p(\text{goal} \mid c, x, y) = p(\text{goal} \mid x, y)$$

So:

$$\boxed{\; v(\text{duel}) = p(\text{goal} \mid c^\star, x, y) \;-\; p(\text{goal} \mid x, y) \;}$$

### 3.3 Defender sign convention

In the form above, a *larger* $v$ means the observed outcome was *more* dangerous than expected — bad for the defender. We flip the sign so positive means good:

$$v_{\text{def}}(\text{duel}) = \underbrace{p(\text{goal} \mid x, y)}_{\text{expected danger}} \;-\; \underbrace{p(\text{goal} \mid c^\star, x, y)}_{\text{realized danger}}$$

Interpretation: *how much goal probability did this defender avert relative to what a league-average outcome at this pitch location would have produced*.

## 4. One model, not three

The expected-danger term can be computed two different ways:

1. Train a separate unconditional `goal_model` with features `(x, y)` that does not see the outcome.
2. Use the conditional `goal_cond_model` and marginalize:

$$p(\text{goal} \mid x, y) = \sum_c p(c \mid x, y)\, p(\text{goal} \mid c, x, y)$$

Option 2 is preferred. The algebraic identity $\sum_c (\mathbb{1}[c^\star = c] - p(c)) \cdot p(\text{goal} \mid c) = p(\text{goal} \mid c^\star) - p(\text{goal} \mid x, y)$ only holds exactly when both terms are derived from the *same* model. Mixing a separately trained unconditional model introduces residuals that aren't meaningful — differences in model bias rather than differences in defender performance.

Implementation per duel:
```
p_outcomes[i]         = model.predict_proba([x_i, y_i])               # shape (3,)
p_goal_per_outcome[i] = [goal_cond_model.predict_proba([x_i, y_i, c])
                         for c in {0, 1, 2}]                          # shape (3,)
expected_danger[i]    = p_outcomes[i] @ p_goal_per_outcome[i]          # scalar
realized_danger[i]    = p_goal_per_outcome[i, outcome_code_i]          # scalar
duel_value[i]         = expected_danger[i] - realized_danger[i]
```

## 5. Aggregation to final metrics

The per-duel value `v_def` is the leaf unit. Common aggregations:

- **Player leaderboard**: `df.groupby("player_id")["duel_value"].agg(total="sum", per_duel="mean", n="count")`. `total` rewards volume, `per_duel` rewards efficiency, `n` is opportunity. Filter by `n` or minutes played when ranking by `per_duel` to avoid small-sample noise.
- **Per match / per team**: same shape, swap the grouping key.
- **Per zone per player**: identifies players who are uniquely good in specific pitch regions.

## 6. Interpretation of the pitch heatmap

Plotting mean `duel_value` per pitch bin over all duels in the dataset tends to produce a map dominated by the *outcome mix* at each location, not defender skill: zones where many recoveries happen look red, zones where defenders get beat look blue. This is a genuine reflection of what the league collectively accomplishes at each spot — it is not an artifact — but it should not be read as a skill map. Skill-focused maps should group by player first, then plot.

## 7. Limitations

- **Features are only `(x, y)`.** Pressure, defender speed, attacker identity, game state, and score differential all affect both outcome probability and subsequent danger. The current value metric attributes all deviation from the spatial prior to the defender.
- **Danger horizon is fixed to the goal-concession window built into the `outcome == -1` label.** If the VAEP labelling uses, for example, a 10-action look-ahead, a duel's value only captures what happens in that window.
- **`scale_pos_weight` trades calibration for AUC.** The numbers are on the right *scale* only up to an affine transformation. Relative rankings and differences are meaningful; raw absolute values should be interpreted with caution.
- **Ordinality of outcomes is used implicitly, not encoded.** The three outcomes are treated as categories; any monotonicity between "recovered → stopped → neither" in danger emerges from data, not from a structural constraint. In low-data regions this can produce non-monotone estimates.
