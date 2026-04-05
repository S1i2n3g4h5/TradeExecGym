#!/bin/bash
# Commands to create a new branch and push to GitHub

# 1. Create and switch to a new branch
git checkout -b feature/planning-docs

# 2. Add all the new files (planning docs + modified files)
git add .

# 3. Commit with a descriptive message
git commit -m "Add comprehensive planning and strategy documentation

- Add MASTER_ACTION_PLAN.md: Complete 3-week roadmap to 9.8/10
- Add XFACTOR_UPGRADES.md: Advanced quant features (order book, alpha decay, TCA)
- Add LLM_JUDGE_STRATEGY.md: Optimization strategy for LLM evaluators
- Add ROBUSTNESS_VALIDATION_GUIDE.txt: 4-layer validation proof
- Add QUICK_START_GUIDE.md: Immediate next steps guide
- Add validation scripts: baseline/ablation/robustness testing
- Update implementation_plan.md: Enhanced with feedback
- Update ui/app.py: Minor improvements"

# 4. Push to GitHub (assuming 'gh' is your GitHub remote)
git push -u gh feature/planning-docs

echo "✅ Done! Branch 'feature/planning-docs' pushed to GitHub"
