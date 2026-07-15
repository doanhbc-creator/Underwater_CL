import numpy as np
import json

result_path = "/home/ubuntu/doanhbc/Underwater_CL_project/checkpoints/task_iou_history_naive_training.json"
results = json.load(open(result_path))


def build_result_matrix(results):
    """
    Convert nested continual-learning dictionary into matrix R.

    R[i, j] = performance on task j after training task i.
    Missing values are set as np.nan.
    """
    tasks = list(results.keys())
    T = len(tasks)

    R = np.full((T, T), np.nan)

    for i, train_task in enumerate(tasks):
        for j, test_task in enumerate(tasks):
            if test_task in results[train_task]:
                R[i, j] = results[train_task][test_task]

    return R, tasks


def compute_forgetting(R):
    """
    Forgetting for each previous task:
    max performance after learning the task - final performance.

    Average forgetting excludes the final task.
    """
    T = R.shape[0]
    forgetting_per_task = []

    for j in range(T - 1):
        best_previous = np.nanmax(R[j:T-1, j])
        final_score = R[T-1, j]
        forgetting = best_previous - final_score
        forgetting_per_task.append(forgetting)

    avg_forgetting = np.mean(forgetting_per_task)

    return avg_forgetting, forgetting_per_task


def compute_bwt(R):
    """
    Backward Transfer:
    final performance on previous tasks - performance right after learning them.

    Average BWT excludes the final task.
    """
    T = R.shape[0]
    bwt_per_task = []

    for j in range(T - 1):
        initial_score = R[j, j]
        final_score = R[T-1, j]
        bwt = final_score - initial_score
        bwt_per_task.append(bwt)

    avg_bwt = np.mean(bwt_per_task)

    return avg_bwt, bwt_per_task


def compute_fwt(R, baseline=None):
    """
    Forward Transfer:
    performance on a task before learning it.

    This requires upper-triangular values, e.g. R[0,1], R[1,2], etc.

    If baseline is provided, FWT is:
        R[i-1, i] - baseline[i]

    If baseline is None, FWT is:
        R[i-1, i]

    With your current lower-triangular matrix, FWT cannot be computed.
    """
    T = R.shape[0]
    fwt_per_task = []

    for i in range(1, T):
        before_learning_score = R[i-1, i]

        if np.isnan(before_learning_score):
            return None, None

        if baseline is not None:
            fwt = before_learning_score - baseline[i]
        else:
            fwt = before_learning_score

        fwt_per_task.append(fwt)

    avg_fwt = np.mean(fwt_per_task)

    return avg_fwt, fwt_per_task


R, tasks = build_result_matrix(results)

avg_forgetting, forgetting_per_task = compute_forgetting(R)
avg_bwt, bwt_per_task = compute_bwt(R)
avg_fwt, fwt_per_task = compute_fwt(R)

print("Tasks:")
for i, task in enumerate(tasks):
    print(f"{i+1}. {task}")

print("\nResult matrix R:")
print(R)

print("\nPer-task results:")
for i, task in enumerate(tasks[:-1]):
    print(
        f"{task}: "
        f"Forgetting = {forgetting_per_task[i]:.4f}, "
        f"BWT = {bwt_per_task[i]:.4f}"
    )

print("\nAverage metrics:")
print(f"Average Forgetting: {avg_forgetting:.4f}")
print(f"Average BWT: {avg_bwt:.4f}")

if avg_fwt is None:
    print("Average FWT: Not computable because upper-triangular values are missing.")
else:
    print(f"Average FWT: {avg_fwt:.4f}")