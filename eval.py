

def B3(gold_cluster, reverse_gold_cluster, system_cluster, reverse_system_cluster):
    precision = 0
    recall = 0
    for mention in reverse_gold_cluster:
        G = gold_cluster[reverse_gold_cluster[mention]]
        if mention in reverse_system_cluster:
            S = system_cluster[reverse_system_cluster[mention]]
        else:
            S = [mention]
        precision += len(set(S).intersection(set(G))) / len(set(S))
        recall += len(set(S).intersection(set(G))) / len(set(G))
    precision /= len(reverse_gold_cluster)
    recall /= len(reverse_gold_cluster)
    f_score = 2 * precision * recall / (precision + recall)
    return recall, precision, f_score
