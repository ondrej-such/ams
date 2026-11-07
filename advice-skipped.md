# Advice skipped

Review suggestions we decided not to apply, kept so they aren't raised again
and can be revisited before submission. Line numbers refer to `index.qmd` at
the time of the note and may have shifted since.

## Section 3: Logistic models

### "will not converge" (line 204)
- **Advice:** say that the maximum likelihood estimate does not exist, rather
  than that the fitting procedure will not converge.
- **Why it was raised:** non-convergence is how software reports the problem
  (R's `glm` stops after 25 iterations with a warning); the mathematical fact is
  that the likelihood has supremum 1 but no maximizer.
- **Decision:** wording kept (2026-11-07).
