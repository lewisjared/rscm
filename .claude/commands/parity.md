Read the parity results CSV at `tests/regression/parity_results.csv` and provide a prioritised summary of regression test status.

## Instructions

1. Read the CSV file. If it doesn't exist, tell the user to run `uv run pytest tests/regression/` first.

2. Produce a summary with these sections:

### Overview
- Total tests (unique suite+test+variable combinations)
- Pass / fail counts
- Overall pass rate

### Failing Tests (prioritised)
Rank failing tests by how far they are from passing (actual / threshold ratio, highest first).
For phased tests, report the worst phase.

Present as a markdown table:
| Priority | Suite | Test | Variable | Worst Phase | Threshold | Actual | Ratio | Bias |
With ratio = actual / threshold, showing how many x over the tolerance they are.

### Passing Tests Near Threshold
Show any passing tests where actual > 0.5 * threshold (at risk of regressing).
Same table format.

### Recommendations
For the top 3-5 failing tests, briefly suggest what kind of work is needed based on:
- **Bias direction**: consistent warm/cool bias suggests a systematic offset
- **Ratio magnitude**: >10x suggests fundamental model difference; 1-3x suggests tuning needed
- **Phase pattern**: shock-only failures suggest initial condition issues; final-phase failures suggest equilibrium drift

Keep the summary concise. Focus on actionable insights.
