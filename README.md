# Interactive Bracket Site

Run `final_project_models.py` first so `outputs/random_test_predictions.csv` exists. Then run this from the project root so the site can load that file:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/bracket_site/
```

Use the bracket by clicking teams to advance them. The "Randomize Bracket" button creates a new random bracket from the latest randomized test predictions. The "Auto-fill by Model" button advances the team with the higher predicted tournament wins in each matchup. The "Random-fill Winners" button simulates random winners through the whole bracket.
