from finetune.tasks.classification import main as classification_main
from finetune.tasks.imputation import main as imputation_main
from finetune.tasks.niche_prediction import main as niche_main

TASK_REGISTRY = {
    'classification': classification_main,
    'imputation': imputation_main,
    'niche': niche_main,
}
