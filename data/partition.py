from collections import Counter
import numpy as np


def partition_report(targets,
                     data_indices,
                     class_num=None,
                     verbose=True,
                     file=None):
    # https://github.com/SMILELab-FL/FedLab/blob/882fe40b0f2e1d0d8486b3d2af25090d43de3dae/fedlab/utils/functional.py
    if not verbose and file is None:
        print("No partition report generated")
        return

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    if not class_num:
        class_num = max(targets) + 1

    sorted_cid = sorted(
        data_indices.keys())  # sort client id in ascending order

    header_line = "Class frequencies:"
    col_name = "client," + ','.join([f"class{i}"
                                     for i in range(class_num)]) + ",Amount"

    if verbose:
        print(header_line)
        print(col_name)
    if file is not None:
        reports = [header_line, col_name]
    else:
        reports = None

    for client_id in sorted_cid:
        indices = data_indices[client_id].astype('int64')

        client_targets = targets[indices]
        client_sample_num = len(
            indices)  # total number of samples of current client

        client_target_cnt = Counter(
            client_targets)  # { cls1: num1, cls2: num2, ... }

        report_line = f"Client {client_id:3d}," + \
                      ','.join([
                          f"{client_target_cnt[cls] / client_sample_num:.3f}" if cls in client_target_cnt else "0.00"
                          for cls in range(class_num)]) + \
                      f",{client_sample_num}"
        if verbose:
            print(report_line)
        if file is not None:
            reports.append(report_line)

    if file is not None:
        fh = open(file, "w")
        fh.write("\n".join(reports))
        fh.close()
