import numpy as np
from collections import defaultdict
from typing import Optional

class BaseEvaluator:
    def __init__(self, gt_scores, rm_scores):
        self.rm_scores = rm_scores
        self.gt_scores = gt_scores

        # Key: int, num samples. Value: list of predicted win rates
        self.eval_result = defaultdict(list)

    def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
        self._record_result(num_gt_samples, 0)
        return 0
    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        self._record_result_batch(num_gt_samples, [0] * num_repeat)
        return np.asarray([0] * num_repeat)

    def _record_result(self, num_gt_samples, pred_win_rate):
        self.eval_result[num_gt_samples].append(pred_win_rate)

    def _record_result_batch(self, num_gt_samples, pred_win_rate_list):
        self.eval_result[num_gt_samples] += pred_win_rate_list

    def clear_result(self):
        self.eval_result = defaultdict(list)

    def get_eval_result(self):
        return self.eval_result

class HumanOnlyEvaluator(BaseEvaluator):
    '''
    Baseline. Use ground truth scores to predict the win rate.
    '''
    def __init__(self, gt_scores):
        super().__init__(gt_scores, None)
    def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.gt_scores), size=(num_gt_samples,), replace=True)
        pred_win_rate = np.mean(self.gt_scores[idx])
        self._record_result(num_gt_samples, pred_win_rate)
        return pred_win_rate
    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples), replace=True)
        pred_win_rate = np.mean(self.gt_scores[idx], axis = 1)
        self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
        return pred_win_rate

# class HumanOnlyEvaluator(BaseEvaluator):
#     '''
#     Baseline. Use ground truth scores to predict the win rate.
#     '''
#     def __init__(self, gt_scores):
#         super().__init__(gt_scores, None)
#     def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
#         if seed is not None:
#             np.random.seed(seed)
#         idx = np.random.choice(len(self.gt_scores), size=(num_gt_samples,), replace=True)
#         pred_win_rate = np.mean(self.gt_scores[idx])
#         self._record_result(num_gt_samples, pred_win_rate)
#         return pred_win_rate
#     def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
#         if seed is not None:
#             np.random.seed(seed)
#         idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples), replace=True)
#         pred_win_rate = np.mean(self.gt_scores[idx], axis = 1)
#         self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
#         return pred_win_rate

class RmOnlyEvaluator(BaseEvaluator):
    '''
    Use reward modelling scores to predict the win rate.
    '''
    def __init__(self, rm_scores):
        super().__init__(None, rm_scores)
    def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.rm_scores), size=(num_gt_samples,), replace=True)
        pred_win_rate = np.mean(self.rm_scores[idx])
        self._record_result(num_gt_samples, pred_win_rate)
        return pred_win_rate
        
    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.rm_scores), size=(num_repeat, num_gt_samples), replace=True)
        # print(type(self.rm_scores), idx)
        pred_win_rate = np.mean(self.rm_scores[idx], axis = 1)
        self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
        return pred_win_rate

class ControlVariatesEvaluator(BaseEvaluator):
    '''
    Use both ground truth scores and reward modelling scores to predict the win rate.
    '''
    def __init__(self, gt_scores, rm_scores, num_corr_samples: int = 100):
        '''
        num_corr_samples: number of samples to estimate the correlation 
            between the ground truth scores and the reward modelling scores.
        '''
        assert(len(gt_scores) == len(rm_scores))
        assert num_corr_samples <= len(gt_scores), f"num_corr_samples {num_corr_samples} should be less than the number of samples {len(gt_scores)}"
        super().__init__(gt_scores, rm_scores)
        self.num_corr_samples = num_corr_samples
        # alpha should be fixed in all runs
        self.alpha = self._estimate_alpha()
        
    def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.gt_scores), size=(num_gt_samples,), replace=True)
        pred_win_rate = np.mean(self.gt_scores[idx]) - self.alpha * (np.mean(self.rm_scores[idx]) - np.mean(self.rm_scores))
        self._record_result(num_gt_samples, pred_win_rate)
        return pred_win_rate

    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples), replace=True)
        rm_global_mean = np.mean(self.rm_scores)
        gt_mean = np.mean(self.gt_scores[idx], axis = 1)
        rm_mean = np.mean(self.rm_scores[idx], axis = 1)
        pred_win_rate = gt_mean - self.alpha * (rm_mean - rm_global_mean)
        self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
        return pred_win_rate

    def _estimate_alpha(self):
        corr_id = np.random.choice(len(self.gt_scores), size=(self.num_corr_samples,), replace=False)
        corr_gt_scores = self.gt_scores[corr_id]
        corr_rm_scores = self.rm_scores[corr_id]
        cov = np.cov(corr_gt_scores, corr_rm_scores)[0,1]
        rm_var = np.var(corr_rm_scores)
        return cov / rm_var

class ControlVariatesEvaluatorV2(ControlVariatesEvaluator):
    '''
    Reuse samples for alpha estimation to evaluate. So we can save human samples.
    '''
    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        if num_gt_samples <= self.num_corr_samples:
            idx_single = self.corr_id[:num_gt_samples]
            idx = np.tile(idx_single, (num_repeat, 1))
        else:
            new_idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples - self.num_corr_samples), replace=True)
            reuse_idx = np.tile(self.corr_id, (num_repeat, 1))
            idx = np.concatenate([reuse_idx, new_idx], axis=1)
        rm_global_mean = np.mean(self.rm_scores)
        gt_mean = np.mean(self.gt_scores[idx], axis = 1)
        rm_mean = np.mean(self.rm_scores[idx], axis = 1)
        pred_win_rate = gt_mean - self.alpha * (rm_mean - rm_global_mean)
        self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
        return pred_win_rate

    def _estimate_alpha(self):
        self.corr_id = np.random.choice(len(self.gt_scores), size=(self.num_corr_samples,), replace=False)
        corr_gt_scores = self.gt_scores[self.corr_id]
        corr_rm_scores = self.rm_scores[self.corr_id]
        cov = np.cov(corr_gt_scores, corr_rm_scores)[0,1]
        rm_var = np.var(corr_rm_scores)
        return cov / rm_var

class ControlVariatesEvaluatorV3(BaseEvaluator):
    '''
    Sample without replacement. Do not fix alpha in each run 
    '''
    def __init__(self, gt_scores, rm_scores, num_corr_samples: int = 100):
        '''
        num_corr_samples: number of samples to estimate the correlation 
            between the ground truth scores and the reward modelling scores.
        '''
        assert(len(gt_scores) == len(rm_scores))
        assert num_corr_samples <= len(gt_scores), f"num_corr_samples {num_corr_samples} should be less than the number of samples {len(gt_scores)}"
        super().__init__(gt_scores.astype(np.float32), rm_scores.astype(np.float32))
        self.num_corr_samples = num_corr_samples
        # alpha should be fixed in all runs
        # self.alpha = self._estimate_alpha()
        
    # def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
    #     if seed is not None:
    #         np.random.seed(seed)
    #     idx = np.random.choice(len(self.gt_scores), size=(num_gt_samples,), replace=True)
    #     pred_win_rate = np.mean(self.gt_scores[idx]) - self.alpha * (np.mean(self.rm_scores[idx]) - np.mean(self.rm_scores))
    #     self._record_result(num_gt_samples, pred_win_rate)
    #     return pred_win_rate

    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        if num_gt_samples <= len(self.gt_scores):
            idx = self._sample(len(self.gt_scores), num_repeat, num_gt_samples)
            # Get corr_idx
            if num_gt_samples <= self.num_corr_samples:
                corr_other_idx = self._sample(len(self.gt_scores), num_repeat, self.num_corr_samples - num_gt_samples)
                corr_idx = np.concatenate([idx, corr_other_idx], axis=1)
            else:
                corr_idx = idx[0, :self.num_corr_samples]
        else:
            residual_idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples - len(self.gt_scores)), replace=True)
            base_idx = np.tile(np.arange(len(self.gt_scores)), (num_repeat, 1))
            idx = np.concatenate([base_idx, residual_idx], axis = 1)
            corr_idx = np.random.choice(len(self.gt_scores), size=(self.num_corr_samples), replace=False)
        # idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples), replace=True)
        alpha = self._estimate_alpha(corr_idx)
        rm_global_mean = np.mean(self.rm_scores)
        gt_mean = np.mean(self.gt_scores[idx], axis = 1)
        rm_mean = np.mean(self.rm_scores[idx], axis = 1)
        pred_win_rate = gt_mean - alpha * (rm_mean - rm_global_mean)
        self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
        return pred_win_rate

    def _sample(self, max_entry, repeat, n):
        idx_list = [np.random.choice(max_entry, size=(1,n), replace=False) for _ in range(repeat)]
        return np.concatenate(idx_list, axis=0)


    def _estimate_alpha(self, corr_idx):
        # corr_id = np.random.choice(len(self.gt_scores), size=(self.num_corr_samples,), replace=False)
        corr_gt_scores = self.gt_scores[corr_idx]
        corr_rm_scores = self.rm_scores[corr_idx]
        cov = np.cov(corr_gt_scores, corr_rm_scores)[0,1]
        rm_var = np.var(corr_rm_scores)
        return cov / rm_var

class ControlVariatesEvaluatorV4(BaseEvaluator):
    '''
    Use the eval samples to estimate correlation
    '''
    def __init__(self, gt_scores, rm_scores, num_corr_samples: int = 100):
        '''
        num_corr_samples: number of samples to estimate the correlation 
            between the ground truth scores and the reward modelling scores.
        '''
        assert(len(gt_scores) == len(rm_scores))
        assert num_corr_samples <= len(gt_scores), f"num_corr_samples {num_corr_samples} should be less than the number of samples {len(gt_scores)}"
        super().__init__(gt_scores.astype(np.float32), rm_scores.astype(np.float32))
        self.num_corr_samples = num_corr_samples
        # alpha should be fixed in all runs
        # self.alpha = self._estimate_alpha()
        
    # def evaluate(self, num_gt_samples: int, seed: Optional[int] = None):
    #     if seed is not None:
    #         np.random.seed(seed)
    #     idx = np.random.choice(len(self.gt_scores), size=(num_gt_samples,), replace=True)
    #     pred_win_rate = np.mean(self.gt_scores[idx]) - self.alpha * (np.mean(self.rm_scores[idx]) - np.mean(self.rm_scores))
    #     self._record_result(num_gt_samples, pred_win_rate)
    #     return pred_win_rate

    def evaluate_batch(self, num_gt_samples: int, num_repeat: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(len(self.gt_scores), size=(num_repeat, num_gt_samples), replace=True)
        alpha_batch = self._estimate_alpha_batch(idx)
        rm_global_mean = np.mean(self.rm_scores)
        gt_mean = np.mean(self.gt_scores[idx], axis = 1)
        rm_mean = np.mean(self.rm_scores[idx], axis = 1)
        pred_win_rate = gt_mean - alpha_batch * (rm_mean - rm_global_mean)
        self._record_result_batch(num_gt_samples, pred_win_rate.tolist())
        return pred_win_rate

    def _estimate_alpha_batch(self, corr_idx):
        '''
        (B,n)
        '''
        # corr_id = np.random.choice(len(self.gt_scores), size=(self.num_corr_samples,), replace=False)
        assert corr_idx.ndim == 2
        alpha_batch = []
        for row_corr_idx in corr_idx:
            corr_gt_scores = self.gt_scores[row_corr_idx]
            corr_rm_scores = self.rm_scores[row_corr_idx]
            cov = np.cov(corr_gt_scores, corr_rm_scores)[0,1]
            rm_var = np.var(corr_rm_scores)
            alpha_batch.append(cov / rm_var)
        return np.asarray(alpha_batch)

