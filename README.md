# Push-optimal Sokoban solver

Entry for the make.fun bounty `sokoban-solver-01-run-20261121`.

`main.ts` (Node 22, no dependencies) answers every instance of a set:
reads `level.txt` (XSB) per instance under the input directory and writes
`solution.txt` (player moves) per instance under the output directory.

Design: push-level A* with the player position normalised by reachable
region, an admissible minimum-cost box-to-goal matching heuristic (exact
subset DP up to 8 boxes), sound deadlock pruning (dead squares, frozen
2x2 blocks, freeze fixpoint), and room decomposition along the hallway
row with a whole-level A* fallback. Deterministic: budgets are counted
in created states, never in time.

Verified on the kit practice set with the kit's official scorer:
all 5 instances solved in exactly the optimal push counts
(13 / 16 / 29 / 47 / 60), set score 1,000,000.
