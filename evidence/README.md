# Evidence Summary

RAGAS results show that both prompt versions meet the faithfulness target. Prompt V2 performs best overall in this run with faithfulness 1.0000, context recall 1.0000, and context precision 1.0000. Prompt V1 is also strong, with faithfulness 0.9714 and the same context recall/context precision scores.

Answer relevancy is 0.0000 for both prompts because the evaluator returned no valid answer relevancy rows in this free-tier run, so the script safely reports 0.0000 instead of propagating NaN. The screenshot in `03_ragas_scores.png` shows the valid-row counts used for each metric.
