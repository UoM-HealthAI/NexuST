from finetune.tasks.cell_annotation import main as cell_annotation_main
from finetune.tasks.gene_recovery import main as gene_recovery_main
from finetune.tasks.niche_prediction import main as niche_main

TASK_REGISTRY = {
    'cell_annotation': cell_annotation_main,
    'gene_recovery': gene_recovery_main,
    'niche': niche_main,
}
