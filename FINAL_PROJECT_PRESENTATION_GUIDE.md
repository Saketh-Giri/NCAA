# Final Project Presentation Guide

Project: Predicting NCAA men's basketball tournament wins from regular-season metrics

This guide is for the 10-12 minute Final Project Presentation only. It does not complete the separate Data Visualization Presentation.

## Core Project Framing

Use this as the main story for the presentation:

We studied whether regular-season team statistics can predict NCAA tournament success. The prediction task is: given a tournament team's seed and regular-season performance metrics before tournament games begin, how many tournament games will that team win?

The client or beneficiary can be described as bracket analysts, college basketball fans, athletic departments, and media groups that want an evidence-based way to identify teams likely to advance deeper than expected.

Main hypothesis:

Teams with stronger regular-season efficiency, more wins, better overall power ratings, and better tournament seeds should win more tournament games, but the model will still have unavoidable error because the tournament is single-elimination and high-variance.

## Files Created

Use these files for the final presentation work:

- `final_project_models.py`: complete Python workflow for cleaning data, training models, evaluating forecasts, saving model outputs, and creating figures.
- `requirements.txt`: Python dependencies needed to run the workflow.
- `outputs/model_metrics.csv`: model performance table.
- `outputs/random_test_predictions.csv`: team-level forecasts for the latest randomized test split.
- `outputs/feature_importance_best_model.csv`: most important variables for the best model in the latest run.
- `outputs/ridge_coefficients.csv`: coefficients for the interpretable Ridge Regression model.
- `outputs/descriptive_statistics_tournament_teams.csv`: descriptive statistics for tournament teams.
- `outputs/tour_wins_by_seed.csv`: average tournament wins by seed.
- `outputs/tour_wins_distribution.png`: target variable chart.
- `outputs/seed_vs_average_tour_wins.png`: seed relationship chart.
- `outputs/model_performance.png`: model comparison chart.
- `outputs/feature_importance.png`: best-model feature importance chart.
- `outputs/top_predictions_vs_actual.png`: forecast examples from the latest randomized test split.

To rerun the code:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python final_project_models.py
```

## Modeling Decisions

The full dataset has 2,455 team-seasons from 2013 through 2019. The modeling sample uses the 476 team-seasons that made the NCAA tournament. This is the fairest framing because teams that do not make the tournament automatically have zero tournament wins, which would make the prediction task too easy and less useful.

The target variable is `Tour_Wins`, the number of NCAA tournament games won by a team in a given season.

The features include conference, games played, wins, win rate, adjusted offensive efficiency, adjusted defensive efficiency, overall power rating, shooting efficiency, turnover rates, rebounding rates, free throw rates, tempo, wins above bubble, and tournament seed. `POSTSEASON` is intentionally excluded because it reveals the answer after the tournament has already happened.

The presentation should focus on two models:

- Ridge Regression: an interpretable baseline that estimates tournament wins as a weighted combination of standardized features.
- Gradient Boosting Regression: a nonlinear model able to capture interactions between seed, efficiency, wins, and tournament success.

Training and evaluation use a fresh randomized train/test split each time the code runs. The modeling sample is randomly divided into training rows and test rows, so the model metrics and forecast examples will change slightly each run.

Key final results:

- Use the latest values in `outputs/model_metrics.csv` for model MAE, RMSE, and R-squared.
- Because the split is randomized, these values may change each time `final_project_models.py` is rerun.
- The best model is whichever model has the lower mean absolute error in the latest randomized run.
- The strongest best-model features are games played, overall power rating (`BARTHAG`), wins, defensive efficiency (`ADJDE`), offensive efficiency (`ADJOE`), and seed.

## Recommended Deck Structure

Use 11 or 12 slides for a 10-12 minute talk. Aim for about 45-70 seconds per slide. Keep slide text short and put most of the explanation in what you say.

### Slide 1: Title and Hook

Slide title:

Predicting NCAA Tournament Success from Regular-Season Team Metrics

Put on slide:

- Group member names
- Course name
- Dataset: NCAA men's college basketball team-season statistics, 2013-2019
- One-sentence hook: "Can regular-season statistics identify which tournament teams are most likely to advance?"

Visual:

- Use a clean basketball/tournament background or a simple bracket image.

Speaker notes:

Open by saying that March Madness is difficult to predict because every game is single-elimination, but teams enter the tournament with a large amount of measurable regular-season information. The goal is to test how much that information can explain tournament wins.

### Slide 2: Purpose, Client, and Prediction Question

Slide title:

Project Purpose

Put on slide:

- Purpose: predict NCAA tournament wins using pre-tournament information.
- Prediction question: "How many tournament games will each tournament team win?"
- Beneficiaries: fans, bracket analysts, media, and athletic departments.
- Hypothesis: stronger efficiency, better seed, and more wins should predict deeper tournament runs.

Visual:

- A simple flow: Regular-season metrics -> model -> predicted tournament wins.

Speaker notes:

Explain that the project is not trying to perfectly predict every upset. Instead, it asks whether a statistical model can give a reasonable forecast before the tournament begins. This directly addresses the rubric's context, question, and hypothesis category.

### Slide 3: Dataset Overview

Slide title:

Data Overview

Put on slide:

- Source: course-provided `cbb.csv` college basketball dataset.
- Years: 2013-2019.
- Full sample: 2,455 team-seasons.
- Modeling sample: 476 tournament team-seasons.
- Target variable: `Tour_Wins`, from 0 to 6.

Visual:

- Use a compact data snapshot or icon list.
- Optionally include a small note: "Modeling only tournament teams avoids rewarding the model for automatic zero-win non-tournament teams."

Speaker notes:

Define a team-season: one row represents one team's statistics in one season. Explain that the final model focuses only on tournament teams because the practical question is about advancing in the tournament once the field is known.

### Slide 4: Variables and Preprocessing

Slide title:

Variables and Preparation

Put on slide:

- Response variable: tournament wins.
- Key predictors: seed, wins, win rate, offensive efficiency, defensive efficiency, shooting, rebounding, turnovers, tempo, and wins above bubble.
- Categorical predictor: conference, one-hot encoded.
- Excluded leakage variable: `POSTSEASON`.
- Evaluation design: randomized train/test split of tournament-team seasons.

Visual:

- Use a three-part process diagram: clean data -> random split -> train and test models.

Speaker notes:

Define the most important terms clearly. Adjusted offensive efficiency measures scoring efficiency adjusted for opponent and pace; adjusted defensive efficiency measures points allowed adjusted for opponent and pace; `BARTHAG` is an overall team strength rating. Say that `POSTSEASON` was removed because it is only known after the tournament and would make the model invalid.

### Slide 5: Descriptive Statistics - Target Variable

Slide title:

Tournament Wins Are Hard to Predict

Put on slide:

- Most tournament teams win zero or one game.
- Winning four, five, or six games is rare.
- This makes the task challenging because deep runs are unusual.

Visual:

- Insert `outputs/tour_wins_distribution.png`.

Speaker notes:

Explain that the target variable is imbalanced. Most tournament teams exit early, while only one team each year can win six games. This means the model must perform well on common outcomes while still identifying teams with deep-run potential.

### Slide 6: Descriptive Statistics - Seed and Tournament Wins

Slide title:

Seed Matters, But It Is Not Everything

Put on slide:

- 1-seeds averaged 3.25 tournament wins in this dataset.
- 2-seeds averaged 2.18 wins.
- 3-seeds averaged 1.90 wins.
- Double-digit seeds can still win games, but deep runs are less common.

Visual:

- Insert `outputs/seed_vs_average_tour_wins.png`.

Speaker notes:

Use this slide to show an intuitive relationship: better seeds generally win more. Then emphasize that seed alone is not enough. The model also uses efficiency, wins, and other performance indicators to distinguish teams with similar seeds.

### Slide 7: Modeling Approach

Slide title:

Two Model Approach

Put on slide:

- Model 1: Ridge Regression.
- Model 2: Gradient Boosting Regression.
- Both predict the same target: number of tournament wins.
- Metric: average prediction error in tournament wins.
- Test set: randomized tournament-team seasons.

Visual:

- Simple side-by-side boxes for the two models.

Speaker notes:

Say that Ridge Regression is useful because it is easier to interpret and acts as a baseline. Gradient Boosting is useful because tournament performance may depend on nonlinear combinations of features, such as a team having both a strong seed and strong efficiency metrics. Mention that the presentation focuses on two models to keep the results clear.

### Slide 8: Model Performance

Slide title:

Forecast Accuracy on the Randomized Test Set

Put on slide:

- Copy the latest average-error values from `outputs/model_metrics.csv`.
- The exact values change because the train/test split is randomized.
- Both models explain a substantial share of variation in tournament wins.
- Select the model with the lower mean absolute error as the final model for the latest run.

Visual:

- Insert `outputs/model_performance.png`.

Speaker notes:

Define mean absolute error in plain terms: on average, the best model's prediction was off by about half a tournament win. Explain that this is strong performance given the single-elimination format, but it does not mean the model can perfectly predict the champion.

### Slide 9: Most Important Variables

Slide title:

What Drove the Best Model?

Put on slide:

- Most important features: games played, `BARTHAG`, wins, defensive efficiency, offensive efficiency, and seed.
- Interpretation: overall team strength and efficiency matter more than any one box-score statistic.
- Defensive and offensive efficiency both contribute to deep-run forecasts.

Visual:

- Insert `outputs/feature_importance.png`.

Speaker notes:

Explain feature importance as follows: each feature was shuffled to see how much model error increased. If shuffling a feature makes predictions much worse, that feature is important. The finding is intuitive: tournament success is associated with a combination of team quality, resume strength, and seed.

### Slide 10: Forecast Examples

Slide title:

Random Test-Set Forecast Examples

Put on slide:

- Use the top rows from `outputs/random_test_predictions.csv`.
- Include team, year, predicted tournament wins, actual tournament wins, and absolute error.
- Choose three close predictions and one miss to make the model's strengths and limits clear.

Visual:

- Insert `outputs/top_predictions_vs_actual.png`.

Speaker notes:

Use this slide to show that the model identified several strong tournament teams well. Then discuss one larger miss from the latest randomized test split. This is reasonable because winning several straight tournament games requires both quality and favorable game-level outcomes.

### Slide 11: Error, Limitations, and Interpretation

Slide title:

What the Model Gets Right and Misses

Put on slide:

- Strength: identifies high-quality teams likely to make deep runs.
- Limitation: cannot model injuries, matchups, late-season changes, or game-level randomness.
- Limitation: one row per team-season, not one row per game.
- Largest misses can be found by sorting `outputs/random_test_predictions.csv` by `Absolute_Error`.
- Practical use: a decision-support tool, not a perfect bracket generator.

Visual:

- Use a small callout with the largest misses from `outputs/random_test_predictions.csv`.

Speaker notes:

This slide is important for the rubric because it shows that the group understands model error. Explain that the model is useful because it gives a disciplined baseline prediction, but the NCAA tournament has randomness that regular-season summaries cannot fully capture.

### Slide 12: Conclusion and Implications

Slide title:

Conclusions

Put on slide:

- Regular-season metrics can meaningfully predict tournament wins.
- The best model predicts tournament wins with measurable test-set error; use the latest MAE from `outputs/model_metrics.csv`.
- Strong overall team quality, wins, efficiency, and seed were the clearest drivers.
- The model is most useful for ranking likely deep-run teams, not guaranteeing exact outcomes.
- Additional game-level data could improve future forecasts.

Visual:

- End with a clean summary graphic: "Metrics + seed + efficiency -> useful tournament forecast."

Speaker notes:

Restate the purpose and answer the main question directly. Say that the project shows meaningful predictive value, but the best interpretation is probabilistic: the model improves judgment, but does not remove uncertainty. End with one lesson learned: good modeling requires both predictive accuracy and careful thinking about what information would actually be known at prediction time.

## Optional Appendix Slides

Use these only if your instructor expects more technical detail or if you want backup material for Q&A.

Appendix A: Ridge coefficients

- Use `outputs/ridge_coefficients.csv`.
- Explain which standardized features received the largest positive or negative weights.
- Keep this out of the main deck unless asked because coefficients with correlated basketball statistics can be hard to explain quickly.

Appendix B: Data dictionary

- `ADJOE`: adjusted offensive efficiency.
- `ADJDE`: adjusted defensive efficiency.
- `BARTHAG`: overall team strength rating.
- `EFG_O`, `EFG_D`: offensive and defensive effective field goal percentage.
- `TOR`, `TORD`: offensive and defensive turnover rates.
- `ORB`, `DRB`: offensive and defensive rebounding rates.
- `FTR`, `FTRD`: offensive and defensive free throw rates.
- `ADJ_T`: adjusted tempo.
- `WAB`: wins above bubble.

Appendix C: Code workflow

- Show the high-level steps from `final_project_models.py`.
- Load data.
- Filter to tournament teams.
- Create win rate and numeric seed.
- Train Ridge Regression and Gradient Boosting Regression.
- Evaluate on a randomized test split.
- Save metrics, predictions, feature importance, and figures.

## Rubric Checklist

Before presenting, make sure the final deck does the following:

- Context, question, and hypothesis: clearly states why tournament prediction matters, who benefits, and what the model predicts.
- Research method and descriptive statistics: explains dataset size, target variable, key predictors, preprocessing, and train/test split.
- Analytical analysis and results: presents two models, model performance, and important variables in plain English.
- Forecasts and implications: gives at least one forecast with error and explains what the results mean for users.
- Presentation delivery and media: uses readable charts, minimal text, consistent formatting, and avoids reading paragraphs from slides.

## Suggested Speaking Time

Use this pacing:

- Slides 1-2: about 1.5 minutes.
- Slides 3-4: about 2 minutes.
- Slides 5-6: about 2 minutes.
- Slides 7-9: about 3 minutes.
- Slides 10-12: about 3 minutes.

Total: about 11-12 minutes.

## Final Delivery Tips

Keep the presentation focused on the story, not the code. The audience should leave understanding that the model uses pre-tournament information to make a practical forecast, that it performs well but imperfectly, and that its mistakes are explainable because tournament basketball is noisy.

Use simple language for metrics. Instead of saying only "MAE," say "average error." Instead of saying only "feature importance," say "which variables the model relied on most."

Do not overload slides with raw tables. Put the charts on the slides and use the CSV outputs only to copy selected numbers.
