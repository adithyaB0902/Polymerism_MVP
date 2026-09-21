"""User-facing labels for the POLYMEMSIM workflow shell."""

APP_NAME = "POLYMEMSIM"
APP_TAGLINE = "Design better lead-removing membranes, faster"
APP_SUMMARY = (
    "Plan a membrane recipe, enter lab measurements, compare predictions, "
    "and prepare paper-ready analysis in one workspace."
)
STAGES = ["Start", "Plan and Enter", "Analyze", "Optimize and Confirm", "More"]
PROGRESS_STEPS = ["Plan", "Enter", "Analyze", "Optimize", "Confirm", "Export"]
DATA_BADGES = ["Simulated", "Predicted", "Measured in the lab"]
RELIABILITY_BADGES = ["Reliable", "Use with caution", "Not reliable"]
PAPER_PATH = "Plan and Enter -> Analyze -> Optimize and Confirm -> Export"

# Each paper-facing term has a plain label and a technical caption.  Keep
# these paired so UI copy can be reviewed without searching the app code.
LABELS = {
    "planner": {"plain": "Experiment Planner", "technical": "Box-Behnken design, 29 runs"},
    "rsm": {"plain": "What Affects Lead Removal?", "technical": "RSM + ANOVA"},
    "removal": {"plain": "% of lead removed", "technical": "Removal efficiency R(%)"},
    "uptake": {"plain": "Lead absorbed per gram of membrane (mg/g)", "technical": "Uptake qe"},
    "starting_lead": {"plain": "Starting lead level (mg/L)", "technical": "C0"},
    "ph": {"plain": "Acidity (pH)", "technical": "pH"},
    "biochar": {"plain": "Biochar amount (% of polymer)", "technical": "Biochar loading"},
    "chitosan": {"plain": "Chitosan share (% of polymer)", "technical": "Chitosan fraction"},
    "fit": {"plain": "How well the model fits", "technical": "R2"},
    "adjusted_fit": {"plain": "How well the model fits after adjusting for size", "technical": "Adjusted R2"},
    "predicted_fit": {"plain": "How well the model fits on new data", "technical": "Predicted R2"},
    "lack_of_fit": {"plain": "Does the model miss any pattern?", "technical": "Lack of fit"},
    "precision": {"plain": "Signal strength vs. noise", "technical": "Adequate precision"},
    "cross_validation": {"plain": "Fair test on unseen data", "technical": "Cross-validation"},
    "importance": {"plain": "Which factors matter most?", "technical": "SHAP / permutation importance"},
    "best_recipe": {"plain": "Find the Best Recipe", "technical": "Differential evolution / desirability"},
    "prediction_interval": {"plain": "Likely range for the result", "technical": "Prediction interval"},
    "domain": {"plain": "Safe range for predictions", "technical": "Applicability domain"},
    "isotherm": {"plain": "How much it absorbs", "technical": "Isotherm"},
    "kinetics": {"plain": "How fast it absorbs", "technical": "Kinetics"},
    "regeneration": {"plain": "Reuse test", "technical": "Regeneration"},
    "swelling": {"plain": "Water uptake", "technical": "Swelling degree"},
    "porosity": {"plain": "Open space in the membrane", "technical": "Porosity"},
}
