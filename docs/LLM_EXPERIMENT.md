# Optional LLM experiment

The capstone report describes Gemma-2B + LoRA and includes appendix material using Phi-2 for a dengue dataset. Trained checkpoints/adapters are not included here, so the LLM path is documented rather than silently represented as a ready-to-run model.

For a reproducible extension, obtain the source data, create temporally ordered prompts using only past observations, split data chronologically, fine-tune with PEFT/LoRA in a GPU environment, save the adapter/config/package versions, reject malformed numeric generations, and evaluate MAE/RMSE on future weeks against seasonal-naive, SARIMA, Random Forest and XGBoost baselines. Do not download large weights during API startup.
