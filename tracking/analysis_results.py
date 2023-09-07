import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128', dataset_name=dataset_name,
#                             run_ids=None, display_name='dist128'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_2', dataset_name=dataset_name,
#                             run_ids=None, display_name='dist128_2'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_h128', dataset_name=dataset_name,
#                             run_ids=None, display_name='dist128_head128'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ost_dist_128_h128_noce', dataset_name=dataset_name,
#                             run_ids=None, display_name='noce'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_h64', dataset_name=dataset_name,
#                             run_ids=None, display_name='128_64'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_128_h32', dataset_name=dataset_name,
#                             run_ids=None, display_name='128_32'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ostrack_distillation_123_64_h32', dataset_name=dataset_name,
#                             run_ids=None, display_name='64_32'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='ost128_h128', dataset_name=dataset_name,
#                             run_ids=None, display_name='new128_128'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae128'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae128_3'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mtr_128_h128_3', dataset_name=dataset_name,
#                             run_ids=None, display_name='mtr128_3'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_9', dataset_name=dataset_name,
#                             run_ids=None, display_name='mae128_9'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3_Trblk', dataset_name=dataset_name,
#                             run_ids=None, display_name='TrBlk'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3_pe', dataset_name=dataset_name,
#                             run_ids=None, display_name='pe'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_pe_clip_1out', dataset_name=dataset_name,
#                             run_ids=None, display_name='clip'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_pe_clip_3out', dataset_name=dataset_name,
#                             run_ids=None, display_name='clip'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_64_h32_3_pe', dataset_name=dataset_name,
#                             run_ids=None, display_name='64_h32'))
# trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_3_pe_KL', dataset_name=dataset_name,
#                             run_ids=None, display_name='KL'))
# trackers.extend(trackerlist(name='efficienttrack', parameter_name='para_base_4_BN', dataset_name=dataset_name,
#                             run_ids=None, display_name='EV'))
trackers.extend(trackerlist(name='vit_dist', parameter_name='vit_128_h128_noKD', dataset_name=dataset_name,
                            run_ids=None, display_name='base'))
trackers.extend(trackerlist(name='vit_dist', parameter_name='mae_128_h128_noKD', dataset_name=dataset_name,
                            run_ids=None, display_name='mae'))
trackers.extend(trackerlist(name='vit_dist', parameter_name='vit_48_h32_noKD', dataset_name=dataset_name,
                            run_ids=None, display_name='48'))
trackers.extend(trackerlist(name='vit_dist', parameter_name='vit_256_h128_noKD', dataset_name=dataset_name,
                            run_ids=None, display_name='256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
