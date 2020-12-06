from utils.misc import calculate_iou, xywh_to_xyxy


__all__ = ['NewHitRateEvaluator', 'CtxHitRateEvaluator']


class NewHitRateEvaluator:

    def __init__(self, refer, top_N=None, threshold=0.5):
        """Evaluate refexp-based hit rate.

        Args:
            refdb: `refdb` dict.
            split: Dataset split to evaluate on.
            top_N: Select top-N scoring proposals to evaluate. `None` means no selection. Default `None`.

        """
        self.refer = refer
        self.top_N = top_N
        self.threshold = threshold

    def eval_hit_rate(self, split, proposal_dict, image_as_key=False):
        """Evaluate refexp-based hit rate.

        Args:
            proposal_dict: {exp_id or image_id: [{box: [4,], score: float}]}.
            image_as_key: Use image_id instead of exp_id as key, default `False`.

        Returns:
            proposal_per_ref: Number of proposals per refexp.
            hit_rate: Refexp-based hit rate of proposals.

        """
        # Initialize counters
        num_hit = 0
        num_proposal = 0
        num_ref = 0  # NOTE: this is the number of refexp, not ref
        for ref_id in self.refer.getRefIds(split=split):
            ref = self.refer.Refs[ref_id]
            image_id = ref['image_id']
            ann_id = ref['ann_id']
            ann = self.refer.Anns[ann_id]
            gt_box = xywh_to_xyxy(ann['bbox'])
            for exp_id in ref['sent_ids']:
                # Get proposals
                if image_as_key:
                    proposals = proposal_dict[image_id]
                else:
                    proposals = proposal_dict[exp_id]
                # Rank and select proposals
                ranked_proposals = sorted(proposals, key=lambda p: p['score'], reverse=True)[:self.top_N]
                for proposal in ranked_proposals:
                    if calculate_iou(gt_box, proposal['box']) > self.threshold:
                        num_hit += 1
                        break
                num_proposal += len(ranked_proposals)
                num_ref += 1
        proposal_per_ref = num_proposal / num_ref
        hit_rate = num_hit / num_ref
        return proposal_per_ref, hit_rate


class CtxHitRateEvaluator:

    def __init__(self, refer, ctxdb, top_N=None, threshold=0.5):
        self.refer = refer
        self.ctxdb = ctxdb
        self.top_N = top_N
        self.threshold = threshold

    def eval_hit_rate(self, split, proposal_dict, image_as_key=False):
        """Evaluate refexp-based hit rate.

        Args:
            proposal_dict: {exp_id or image_id: [{box: [4,], score: float}]}.
            image_as_key: Use image_id instead of exp_id as key, default `False`.

        Returns:
            proposal_per_ref: Number of proposals per refexp.
            hit_rate: Refexp-based hit rate of proposals.

        """
        # Initialize counters
        recall_list = []
        avg_num_list = []
        for exp_id, ctx in self.ctxdb[split].items():
            exp_id = int(exp_id)
            if len(ctx['ctx']) == 0:
                continue
            # Get proposals
            if image_as_key:
                image_id = self.refer.sentToRef[exp_id]['image_id']
                proposals = proposal_dict[image_id]
            else:
                proposals = proposal_dict[exp_id]
            # Rank and select proposals
            ranked_proposals = sorted(proposals, key=lambda p: p['score'], reverse=True)[:self.top_N]
            hit_num, ctx_num = 0, 0
            for ctx_item in ctx['ctx']:
                ctx_num += 1
                ctx_box = ctx_item['box']
                for proposal in ranked_proposals:
                    if calculate_iou(ctx_box, proposal['box']) > self.threshold:
                        hit_num += 1
                        break
            recall_list.append(hit_num / ctx_num)
            avg_num_list.append(len(ranked_proposals))
        return sum(avg_num_list) / len(avg_num_list), sum(recall_list) / len(recall_list)
