import torch


def get_gt_ranks(ranks, ans_ind):
    batch_size, num_rounds = ans_ind.size()

    # collapse batch dimension
    ranks = ranks.view(batch_size * num_rounds, -1)
    ans_ind = ans_ind.view(batch_size * num_rounds)
    gt_ranks = torch.zeros_like(ans_ind)

    for i in range(ans_ind.size(0)):
        gt_ranks[i] = int(ranks[i, ans_ind[i]])

    # add batch dimension again
    gt_ranks = gt_ranks.view(batch_size, num_rounds)
    return gt_ranks


def process_ranks(ranks):
    # collapse batch dimension
    ranks = ranks.view(-1)
    num_ques = ranks.size(0)

    ranks = ranks.float()
    num_r1 = float(torch.sum(torch.le(ranks, 1)))
    num_r5 = float(torch.sum(torch.le(ranks, 5)))
    num_r10 = float(torch.sum(torch.le(ranks, 10)))
    print("\tNo. questions: {}".format(num_ques))
    print("\tr@1: {}".format(num_r1 / num_ques))
    print("\tr@5: {}".format(num_r5 / num_ques))
    print("\tr@10: {}".format(num_r10 / num_ques))
    print("\tmeanR: {}".format(torch.mean(ranks)))
    print("\tmeanRR: {}".format(torch.mean(ranks.reciprocal())))


def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this position
    # but we want i-th position to have rank of score at that position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks
