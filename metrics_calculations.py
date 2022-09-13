import statistics
import math
import matplotlib.pyplot as plt


def load_data(file_path):
    output = []

    with open(file_path) as f:
        for line in f:
            content = line.strip().split(',')
            if len(content) > 0 and content[0][0] != '#':
                gt = int(content[0])
                score = float(content[1])

                output.append((gt, score))

    return output


def compute_fmr(observations, threshold):
    impostor_count = 0
    false_match_count = 0

    for obs in observations:
        if obs[0] == 0:  # impostor sample
            impostor_count = impostor_count + 1

            if obs[1] > threshold:
                false_match_count = false_match_count + 1

    if impostor_count > 0:
        return float(false_match_count) / impostor_count
    else:
        return float('inf')


def compute_fnmr(observations, threshold):
    genuine_count = 0
    false_non_match_count = 0

    for obs in observations:
        if obs[0] == 1:  # genuine sample
            genuine_count = genuine_count + 1

            if obs[1] < threshold:
                false_non_match_count = false_non_match_count + 1

    if genuine_count > 0:
        return float(false_non_match_count) / genuine_count
    else:
        return float('inf')


def compute_fmr_fnmr_eer(observations):
    output_fmr = 0
    output_fnmr = 0
    output_threshold = 0
    fmr_fnmr_diff = float('inf')

    scores = []
    for obs in observations:
        scores.append(obs[1])
    scores = sorted(scores)

    for threshold in scores:
        current_fmr = compute_fmr(observations, threshold)
        current_fnmr = compute_fnmr(observations, threshold)

        current_diff = abs(current_fmr - current_fnmr)
        if current_diff <= fmr_fnmr_diff:
            output_fmr = current_fmr
            output_fnmr = current_fnmr
            output_threshold = threshold
            fmr_fnmr_diff = current_diff

    return output_fmr, output_fnmr, output_threshold


def compute_fmr_fnmr_auc(observations):
    fmr = []
    fnmr = []
    auc_parts = []
    auc = 0

    scores = []
    for obs in observations:
        scores.append(obs[1])
    scores = sorted(scores)

    for threshold in scores:
        current_fmr = compute_fmr(observations, threshold)
        current_fnmr = compute_fnmr(observations, threshold)

        fmr.append(current_fmr)
        fnmr.append(current_fnmr)

    for i in range(len(fmr) - 1):
        auc_parts.append((1 / 2) * (fnmr[i] + fnmr[i + 1]) * abs(fmr[i] - fmr[i + 1]))

    auc = sum(auc_parts)

    return auc, fmr, fnmr


def compute_d_prime(observations):
    genuine_scores = []
    impostor_scores = []
    d_prime = 0

    for obs in observations:
        if obs[0] == 1:
            genuine_scores.append(obs[1])
        else:
            impostor_scores.append(obs[1])

    genuine_mean = sum(genuine_scores) / len(genuine_scores)
    impostor_mean = sum(impostor_scores) / len(impostor_scores)

    genuine_variance = statistics.variance(genuine_scores)
    impostor_variance = statistics.variance(impostor_scores)

    d_prime = math.sqrt(2.0) * abs(genuine_mean - impostor_mean) / math.sqrt(genuine_variance + impostor_variance)

    return d_prime


def compute_tmr(observations, threshold):
    fmr = compute_fmr(observations, threshold)
    if fmr == float('inf'):
        return float('inf')
    else:
        return 1.0 - fmr


def compute_tnmr(observations, threshold):
    fnmr = compute_fnmr(observations, threshold)
    if fnmr == float('inf'):
        return float('inf')
    else:
        return 1.0 - fnmr


def plot_fmr_fnmr_roc(observations):
    auc, fmr, fnmr = compute_fmr_fnmr_auc(observations)

    plt.xlabel('FMR')
    plt.ylabel('FNMR')

    plt.plot(fmr, fnmr, label='AUC:' + '{:.5f}'.format(auc))

    plt.legend()
    plt.show()


def process_data(observations):
    data = load_data(observations)
    fmr = compute_fmr(data, 0.5)
    fnmr = compute_fnmr(data, 0.5)
    print('FMR:', fmr, 'FNMR:', fnmr)

    tmr = compute_tmr(data, 0.5)
    tnmr = compute_tnmr(data, 0.5)
    print('TMR:', tmr, 'TNMR:', tnmr)

    fmr, fnmr, threshold = compute_fmr_fnmr_eer(data)
    auc, _, _ = compute_fmr_fnmr_auc(data)
    d_prime = compute_d_prime(data)
    print('FMR:', fmr, 'FNMR:', fnmr, 'Threshold:', threshold, 'AUC:', auc, 'D-Prime:', d_prime)

    plot_fmr_fnmr_roc(data)


if __name__ == "__main__":
    print("---------------------")
    print("x Metrics Evaluations:")
    print("---------------------")
    process_data("observations_x.csv")

    print("---------------------")
    print("y Metrics Evaluations:")
    print("---------------------")
    process_data("observations_y.csv")
