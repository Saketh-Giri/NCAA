# Interactive Bracket Site

Run `final_project_models.py` first so `outputs/official_bracket_predictions.csv` exists. Then run this from the project root so the site can load that file:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/bracket_site/
```

Use the season selector to load one official NCAA bracket year at a time. The bracket uses official region seed slots and standard seed pairings, with First Four slots resolved by the model before the Round of 64. Click teams to advance them manually, or use "Auto-Fill by Model" to advance the team with the higher predicted tournament wins in each matchup.
