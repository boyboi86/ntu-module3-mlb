# Engineered Features

This file documents the 25 features added in `notebooks/02_feature_engineering.ipynb` (function `add_features`). For each one it gives the calculation, the reason it should relate to wins, and how it behaves in the training data.

All features are computed from a team-season's own row, so the same function applies to `predict.csv` with no information taken from other rows or from the target.

## How to read the evidence columns

Each table reports three numbers from `Data/train_features.csv` (1,412 team-seasons):

| Column | Meaning |
|---|---|
| **r(W)** | Pearson correlation with actual wins |
| **r(resid)** | Correlation with `W − pythpat_W`, the wins that Pythagorean expectation fails to predict. A feature with a large value here adds information beyond runs scored and allowed. |
| **Ridge coef** | Coefficient in the final Ridge model (α = 2.51), in wins per one standard deviation of the feature |

Ridge coefficients are shared across overlapping features, so read them by group. A single coefficient can have a counter-intuitive sign when a correlated feature next to it carries the effect.

For reference, the raw columns `R` and `RA` have r(W) of +0.585 and −0.521.

## Shared building blocks

| Symbol | Calculation | Meaning |
|---|---|---|
| `IP` | `IPouts / 3` | Innings pitched (3 outs per inning) |
| `TB` | `H + 2B + 2·3B + 3·HR` | Total bases. A single is 1, a double 2, a triple 3, a home run 4. `H` already counts each hit once, so the extra terms add the additional bases. |

## Summary

| Group | Features | Purpose |
|---|---|---|
| Run-based | `pythpat_W`, `pyth_W`, `RD`, `RD_pg`, `log_R_RA`, `R_pg`, `RA_pg` | Wins come from outscoring opponents |
| Close-game | `SV_pg`, `SHO_pg`, `CG_pg`, `IP_pg` | Capture what run totals miss: results in close games |
| Era-relative | `R_rel`, `RA_rel` | Judge scoring against the season's league average |
| Offense rates | `OBP`, `SLG`, `OPS`, `ISO`, `RC_pg` | Standard measures of batting quality |
| Pitching and defense rates | `WHIP`, `K9`, `BB9`, `HR9`, `FIP_core`, `UER`, `E_pg` | Standard measures of run prevention |

---

## 1. Run-based features

A team wins a game by scoring more runs than it allows, so a season's win total depends mainly on runs scored (`R`) and allowed (`RA`). These features express that relationship in forms a model can use directly.

| Feature | Calculation | r(W) | r(resid) | Ridge coef |
|---|---|---|---|---|
| `pythpat_W` | `G × R^x / (R^x + RA^x)`, where `x = ((R + RA) / G)^0.287` | +0.954 | −0.005 | +3.03 |
| `pyth_W` | `G × R^1.83 / (R^1.83 + RA^1.83)` | +0.953 | −0.006 | +2.89 |
| `RD` | `R − RA` | +0.937 | −0.011 | +1.03 |
| `RD_pg` | `(R − RA) / G` | +0.936 | −0.012 | +0.37 |
| `log_R_RA` | `ln(R / RA)` | +0.939 | −0.017 | −0.39 |
| `R_pg` | `R / G` | +0.562 | +0.038 | +0.32 |
| `RA_pg` | `RA / G` | −0.550 | +0.052 | −0.12 |

### `pyth_W`: Pythagorean wins

Bill James found that a team's winning percentage is well approximated by

```
win% ≈ R^k / (R^k + RA^k)
```

James originally used k = 2, which resembles the Pythagorean theorem, hence the name. Later work found k ≈ 1.83 fits better across MLB history. Multiplying by `G` turns the percentage into a win count, which also adjusts for season length (150 to 165 games here).

**Why it's meaningful:**
- It's the most widely used single estimate of team quality in baseball analysis.
- The relationship is curved, not linear: each extra run is worth slightly less to a team that already outscores opponents by a lot. A linear model can't produce that curve from raw `R` and `RA`, so supplying it directly helps.
- On its own it predicts wins with an RMSE of about 4.0. That matches the error commonly reported for Pythagorean expectation.

### `pythpat_W`: Pythagenpat wins

This is the same formula, but the exponent changes with the scoring environment:

```
x = ((R + RA) / G) ^ 0.287
```

The exponent formula is from the "Pythagenpat" method, developed independently by David Smyth and Patriot.

**Why it's meaningful:** in a low-scoring environment, one run decides more games, so the exponent should be smaller. In high-scoring seasons it should be larger. This dataset runs from the dead-ball era to the steroid era, with league scoring from about 3.4 to 5.5 runs per game, so a fixed exponent fits some eras worse than others.

It is the strongest single feature in the data (r = +0.954), slightly ahead of `pyth_W` (RMSE 4.00 against 4.04).

Its residual correlation is near zero by construction. Its residual (`W − pythpat_W`) is the baseline the r(resid) column measures every other feature against.

### `RD` and `RD_pg`: run differential

`RD` is runs scored minus runs allowed. `RD_pg` divides it by games played.

**Why they're meaningful:**
- Run differential is the linear version of the Pythagorean idea. A common rule of thumb is that about 10 runs of differential equal 1 win.
- It gives linear models a simple, near-linear signal.
- The per-game version removes the effect of season length, so a 154-game season and a 162-game season are measured on the same scale.

**Caveat:** `RD = R − RA` and `RD_pg = R_pg − RA_pg` exactly. Both are exact linear combinations of other columns, so the plain-OLS preprocessor drops them. Ridge and trees can keep them.

### `log_R_RA`: log run ratio

`ln(R / RA)` is 0 for a team that scores exactly as many runs as it allows, positive for one that outscores its opponents, and negative otherwise.

**Why it's meaningful:** taking logs of the Pythagorean formula shows that winning percentage depends on the *ratio* R/RA. The log puts that ratio on a symmetric, roughly linear scale: scoring 10% more than you allow has the same size effect as allowing 10% more than you score, with the opposite sign.

Its Ridge coefficient is slightly negative only because `pythpat_W` and `pyth_W` already carry the same information.

### `R_pg` and `RA_pg`: runs per game

**Why they're meaningful:**
- They express offense and defense as rates, so they don't depend on season length.
- They let the model weight offense and defense separately, which a single differential can't do.
- They're also the inputs to the era-relative features below.

---

## 2. Close-game features

Pythagorean expectation predicts how many games a team *should* have won given its runs. Teams that do especially well or badly in one-run games beat or miss that estimate. Run totals alone can't capture this, so these features target it directly.

Their r(resid) values are the highest in the feature set.

| Feature | Calculation | r(W) | r(resid) | Ridge coef |
|---|---|---|---|---|
| `SV_pg` | `SV / G` | +0.388 | **+0.241** | +1.51 |
| `IP_pg` | `IPouts / G` | +0.424 | **+0.131** | +1.10 |
| `CG_pg` | `CG / G` | −0.044 | −0.112 | +1.68 |
| `SHO_pg` | `SHO / G` | +0.442 | +0.024 | −0.39 |

### `SV_pg`: saves per game

A save is credited to a relief pitcher who finishes a win while protecting a lead, typically of 3 runs or fewer.

**Why it's meaningful:**
- A save can only happen in a **close win**, so saves count close games the team won.
- It's the best single explanation of the wins Pythagorean expectation misses (r = +0.24 with the residual). The raw `SV` count also has one of the largest Ridge coefficients (+2.57).
- Dividing by games played puts 154- and 162-game seasons on the same scale.

**Caveat:** saves became an official statistic in 1969. Earlier totals were credited retroactively and are lower (median 15 before 1969 against 39 after). The era and decade flags let the model adjust for this.

### `IP_pg`: pitching outs per game

The number of outs the team's pitchers recorded, divided by games played. A full 9-inning game is 27 outs.

**Why it's meaningful:** when the home team leads after the top of the 9th inning, the game ends without the bottom of the 9th. So:
- A winning home team bats in only 8 innings, but its pitchers still record all 27 outs.
- A losing road team's pitchers record only 24 outs in those games.

Winning teams therefore record slightly more outs per game. Extra-inning games and rainouts add noise, but the effect is consistent (r = +0.13 with the residual). The raw columns show the same effect: `AB` has a large negative coefficient (winners bat less) and `IPouts` a positive one.

### `CG_pg`: complete games per game

How often the starting pitcher finished the game.

**Why it's meaningful:**
- Within an era, starters who go the distance are usually pitching well.
- Across eras, the rate fell from 74% of games in the 1900s to 3% in the 2010s (team averages in this data), so it also records how pitchers were used in each era.

That mix explains why its overall correlation with wins is near zero (−0.04), while its coefficient is positive once the era flags are in the model.

### `SHO_pg`: shutouts per game

**Why it's meaningful:** every shutout is a win, so this counts guaranteed wins and reflects dominant pitching.

Its correlation with wins is strong (+0.44), but it adds little beyond runs allowed (r(resid) = +0.02). Shutouts are already reflected in a low `RA`. Its small negative Ridge coefficient sits alongside a positive coefficient on raw `SHO`, so the two largely cancel.

---

## 3. Era-relative features

| Feature | Calculation | r(W) | r(resid) | Ridge coef |
|---|---|---|---|---|
| `R_rel` | `(R / G) / mlb_rpg` | +0.703 | −0.033 | +0.96 |
| `RA_rel` | `(RA / G) / mlb_rpg` | −0.730 | −0.012 | −0.02 |

`mlb_rpg` is the league-wide average runs per game for that season. A value of 1.10 means the team scored 10% more runs per game than the league average that year.

**Why they're meaningful:**
- The same raw scoring means different things in different eras. 4.5 runs per game was excellent in 1968 and average in 2000.
- Wins come from beating the *other teams in the same season*, so scoring relative to the league is a fairer measure of quality across eras.
- It also matches the competition's goal of models that work across eras without learning the year.

The evidence is clear: `R_rel` correlates with wins at +0.70 against +0.56 for `R_pg`, and `RA_rel` at −0.73 against −0.55 for `RA_pg`.

---

## 4. Offense rates

These are standard batting statistics. They measure how a team generates runs, which helps most for tree models: trees can't easily calculate ratios themselves.

| Feature | Calculation | r(W) | r(resid) | Ridge coef |
|---|---|---|---|---|
| `OBP` | `(H + BB) / (AB + BB)` | +0.477 | +0.055 | +1.33 |
| `SLG` | `TB / AB` | +0.466 | +0.101 | +1.34 |
| `OPS` | `OBP + SLG` | +0.504 | +0.093 | +1.44 |
| `ISO` | `SLG − H / AB` | +0.390 | +0.099 | −0.25 |
| `RC_pg` | `(H + BB) × TB / (AB + BB) / G` | +0.485 | +0.082 | −1.22 |

### `OBP`: on-base percentage

The share of plate appearances in which the batter reached base.

**Why it's meaningful:** avoiding outs is the most valuable thing a batter can do, since each team gets only 27 outs per game. It was central to the "Moneyball" approach.

**Caveat:** this is an approximation. The official formula also includes hit-by-pitches and sacrifice flies, which this dataset doesn't have.

### `SLG`: slugging percentage

Total bases per at-bat.

**Why it's meaningful:** it measures hitting for power. Extra-base hits move runners further and score runs more efficiently than singles.

### `OPS`: on-base plus slugging

**Why it's meaningful:** it's a simple combination of getting on base and hitting for power, and it tracks runs scored better than either part alone (r = +0.50).

**Caveat:** `OPS = OBP + SLG` exactly, so the OLS preprocessor drops it.

### `ISO`: isolated power

Slugging minus batting average: extra bases per at-bat, without singles.

**Why it's meaningful:** it separates power from contact hitting. Two teams with the same batting average can differ a lot in power.

### `RC_pg`: runs created per game

Bill James's basic runs-created formula is

```
RC = (times on base) × (total bases) / (opportunities)
```

Here that's `(H + BB) × TB / (AB + BB)`, divided by games played.

**Why it's meaningful:** it estimates the runs a lineup *should* score from its hits and walks. A team whose actual runs differ from its runs created scored more or fewer runs than its hitting suggests, for example through clutch hitting or baserunning.

Its negative Ridge coefficient means that, given actual runs, a team that scored more than its hitting suggests also tends to win more than expected.

---

## 5. Pitching and defense rates

These mirror the offense rates for run prevention. Expressing them per inning or per game makes them comparable across seasons of different lengths.

| Feature | Calculation | r(W) | r(resid) | Ridge coef |
|---|---|---|---|---|
| `WHIP` | `(HA + BBA) / IP` | −0.527 | +0.021 | −0.89 |
| `K9` | `9 × SOA / IP` | +0.248 | +0.091 | +0.07 |
| `BB9` | `9 × BBA / IP` | −0.379 | −0.010 | +0.47 |
| `HR9` | `9 × HRA / IP` | +0.001 | +0.093 | −0.55 |
| `FIP_core` | `(13 × HRA + 3 × BBA − 2 × SOA) / IP` | −0.343 | +0.023 | −0.44 |
| `UER` | `RA − ER` | −0.370 | −0.085 | −0.19 |
| `E_pg` | `E / G` | −0.312 | −0.109 | +0.25 |

### `WHIP`: walks plus hits per inning

**Why it's meaningful:** it counts the baserunners a pitching staff allows per inning, which is the most direct measure of how often opponents get chances to score. It has the strongest correlation with wins in this group (−0.53).

### `K9`, `BB9`, `HR9`: strikeouts, walks and home runs per 9 innings

**Why they're meaningful:** these are the three outcomes a pitcher controls almost entirely, without help or harm from the fielders.
- Strikeouts end at-bats with no chance of a hit.
- Walks give free baserunners.
- Home runs score at least one run immediately.

Scaling to 9 innings, one full game, makes them comparable across staffs and seasons.

Their raw correlations with wins are weak or mixed, because all three rose over the century: strikeouts and home runs went up across eras. They matter more within an era, which the model handles together with the era flags.

### `FIP_core`: fielding-independent pitching

FIP (Tom Tango) estimates how many runs a pitching staff deserved to allow, using only strikeouts, walks and home runs. The weights 13, 3 and −2 reflect the average run value of each outcome.

**Why it's meaningful:** runs allowed mix pitching quality with fielding and luck on balls in play. FIP isolates the part due to the pitchers, which tends to predict future run prevention better than ERA does.

**Notes:**
- The official FIP adds a yearly constant so it reads on the ERA scale. The constant only shifts the value, which makes no difference to a model, so it's left out.
- `FIP_core` equals `(13 × HR9 + 3 × BB9 − 2 × K9) / 9` exactly, so the OLS preprocessor drops it.

### `UER`: unearned runs

Runs allowed minus earned runs: runs that scored only because of a fielding error.

**Why it's meaningful:** it measures runs lost to poor defense rather than poor pitching, so it separates the fielding part of `RA`.

**Caveat:** `UER = RA − ER` exactly, so the OLS preprocessor drops it.

### `E_pg`: errors per game

**Why it's meaningful:** errors give opponents extra outs and baserunners. The per-game rate removes the effect of season length.

Errors fell sharply over the century as gloves and field conditions improved, so this also reflects the era. Within an era, fewer errors means better defense. Its correlation with the Pythagorean residual is −0.11: error-prone teams also win fewer games than their runs suggest.

---

## Exact linear combinations

These five features are exactly equal to a linear combination of other columns. They're harmless for Ridge and tree models but cause perfect collinearity in ordinary least squares, so the `ols` preprocessor in `03_data_processing.ipynb` drops them.

| Feature | Identity |
|---|---|
| `RD` | `R − RA` |
| `RD_pg` | `R_pg − RA_pg` |
| `UER` | `RA − ER` |
| `OPS` | `OBP + SLG` |
| `FIP_core` | `(13 × HR9 + 3 × BB9 − 2 × K9) / 9` |

## Did the features help?

Cross-validated RMSE in wins, from `02_feature_engineering.ipynb` (lower is better):

| Model | Raw features | Raw + engineered |
|---|---|---|
| Ridge, random 5-fold | 3.485 | **3.413** |
| Ridge, leave one decade out | 3.853 | **3.709** |
| Gradient boosting, random 5-fold | 4.117 | **3.881** |
| Gradient boosting, leave one decade out | 4.438 | **4.063** |

- **Gradient boosting gains the most.** Trees can't easily compute ratios such as R/RA, so supplying them directly helps.
- **Ridge gains less** because it can already combine raw `R` and `RA` linearly. The improvement comes mainly from the curved Pythagorean features and the close-game features.
- The final tuned Ridge model scores a 5-fold CV MAE of 2.707 wins and an R² of 0.934.
