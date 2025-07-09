

# def Acrophase(
#     GOI: Union[List[str], np.ndarray], 
#     GOI_Ideal: Union[List[str], np.ndarray], 
#     Ideal_Acrophases: np.ndarray, 
#     R_Squared: np.ndarray, 
#     Estimated_Acrophases: np.ndarray, 
#     P_Values: np.ndarray, 
#     P_Cutoff: float, 
#     space_factor: float = math.pi/35.5, 
#     create_fig: bool = True, 
#     subplot_space_factor_ratio: float = 1.5, 
#     subplot_fontsize: int = 8
# ):
#     if create_fig:
#         fig = plt.figure(figsize=(10, 11))
#         ax = plt.axes(polar=True)
#     else:
#         ax = plt.gca()
    
#     ax.spines["polar"].set_visible(False)
#     plt.axis([0, 2 * math.pi, 0, 1.1])
#     plt.xticks([math.pi/2, math.pi, 3*math.pi/2, 2*math.pi],
#            [r'$0$', r'$\frac{3\pi}{2}$', r'$\pi$', r'$\frac{\pi}{2}$'],
#            fontsize=22)
#     plt.yticks([0, 0.5, 1], ["", "", ""])
#     ax.yaxis.grid(True)
#     ax.xaxis.grid(False)
    
#     comparable_indices = P_Values < P_Cutoff

#     GOI = np.array(GOI)
#     GOI_Ideal = np.array(GOI_Ideal)

#     Comparable_GOI = GOI[comparable_indices]
#     Comparable_GOI_Ideal = GOI_Ideal[comparable_indices]
#     comparable_core_clock_acrophases = Estimated_Acrophases[comparable_indices]
#     mouse_comparable_acrophases = Ideal_Acrophases[comparable_indices]

#     plot_acrophases = np.mod(-(comparable_core_clock_acrophases - math.pi/2), 2 * math.pi)
#     plot_ideals = np.mod(-(mouse_comparable_acrophases - math.pi/2), 2 * math.pi)

#     plt.scatter(plot_acrophases, np.ones(len(plot_acrophases)), alpha=0.7, s=R_Squared[comparable_indices] * (1000 if create_fig else 75), c="b", label="Estimated Acrophases")
#     plt.scatter(plot_ideals, np.ones(len(plot_ideals)) * 0.5, alpha=0.7, s=(75 if create_fig else 50), c="orange", label="Ideal Acrophases")
    
#     if create_fig:
#         plt.legend(loc=(0.45, -0.1875))
    
#     significant_acrophase_mean = Circular_Mean(plot_acrophases)
#     range_upper, range_lower = plus_minus(significant_acrophase_mean, math.pi/2)
#     range_upper = range_upper % (2 * math.pi)
#     range_lower = range_lower % (2 * math.pi)

#     significant_ideal_acrophase_mean = Circular_Mean(plot_ideals)
#     range_upper_ideal, range_lower_ideal = plus_minus(significant_ideal_acrophase_mean, math.pi/2)
#     range_upper_ideal = range_upper_ideal % (2 * math.pi)
#     range_lower_ideal = range_lower_ideal % (2 * math.pi)

#     if range_upper > range_lower:
#         acrophases_in_range_logical = (range_lower < plot_acrophases) & (plot_acrophases <= range_upper)
#     else:
#         acrophases_in_range_logical = ~((range_upper < plot_acrophases) & (plot_acrophases <= range_lower))
    
#     if range_upper_ideal > range_lower_ideal:
#         ideal_acrophases_in_range_logical = (range_lower_ideal < plot_ideals) & (plot_ideals <= range_upper_ideal)
#     else:
#         ideal_acrophases_in_range_logical = ~((range_upper_ideal < plot_ideals) & (plot_ideals <= range_lower_ideal))
    
#     in_range_sig_acrophases_logical = acrophases_in_range_logical
#     in_range_sig_ideal_acrophases_logical = ideal_acrophases_in_range_logical
    
#     acrophase_in_range_mean = Circular_Mean(plot_acrophases[in_range_sig_acrophases_logical]) if np.any(in_range_sig_acrophases_logical) else 0.0
#     ideal_acrophase_in_range_mean = Circular_Mean(plot_ideals[in_range_sig_ideal_acrophases_logical]) if np.any(in_range_sig_ideal_acrophases_logical) else 0.0

#     closest_gene_index = np.argmin(np.arccos(np.cos(acrophase_in_range_mean - plot_acrophases)))
#     closest_ideal_gene_index = np.argmin(np.arccos(np.cos(ideal_acrophase_in_range_mean - plot_ideals)))
    
#     middle_gene_acrophase = plot_acrophases[closest_gene_index]
#     ideal_middle_gene_acrophase = plot_ideals[closest_ideal_gene_index]

#     plt.annotate(
#         Comparable_GOI[closest_gene_index], 
#         xy=[middle_gene_acrophase, 1], 
#         xytext=[middle_gene_acrophase, (1.35 if create_fig else 1.85)],
#         arrowprops=dict(arrowstyle="->", facecolor="grey"), 
#         fontsize=(12 if create_fig else subplot_fontsize)
#     )
#     plt.annotate(
#         Comparable_GOI_Ideal[closest_ideal_gene_index], 
#         xy=[ideal_middle_gene_acrophase, 0.5], 
#         xytext=[ideal_middle_gene_acrophase, (0.75 if create_fig else 0.85)],
#         arrowprops=dict(arrowstyle="->", facecolor="grey"), 
#         fontsize=(12 if create_fig else subplot_fontsize)
#     )

#     distance_from_middle_annotation = middle_gene_acrophase - plot_acrophases
#     sig_phases_larger_middle_logical = distance_from_middle_annotation < 0
#     sig_phases_smaller_middle_logical = distance_from_middle_annotation > 0

#     distance_from_ideal_middle_annotation = ideal_middle_gene_acrophase - plot_ideals
#     sig_phases_larger_ideal_middle_logical = distance_from_ideal_middle_annotation < 0
#     sig_phases_smaller_ideal_middle_logical = distance_from_ideal_middle_annotation > 0

#     sig_larger_phases = plot_acrophases[sig_phases_larger_middle_logical]
#     sig_smaller_phases = plot_acrophases[sig_phases_smaller_middle_logical]
#     sig_larger_ideal_phases = plot_ideals[sig_phases_larger_ideal_middle_logical]
#     sig_smaller_ideal_phases = plot_ideals[sig_phases_smaller_ideal_middle_logical]

#     sorted_sig_larger_phases = np.sort(sig_larger_phases)
#     sorted_sig_smaller_phases = np.sort(sig_smaller_phases)[::-1]
#     sorted_sig_larger_ideal_phases = np.sort(sig_larger_ideal_phases)
#     sorted_sig_smaller_ideal_phases = np.sort(sig_smaller_ideal_phases)[::-1]

#     annotation_x_vals_larger = np.copy(sorted_sig_larger_phases)
    
#     if len(annotation_x_vals_larger) > 0:
#         temp_arr = np.concatenate(([middle_gene_acrophase], annotation_x_vals_larger))
#         nearest_neighbor = np.diff(temp_arr)
#         nearest_neighbor_too_close_logical = nearest_neighbor < space_factor
#         too_close = np.any(nearest_neighbor_too_close_logical)
        
#         while too_close:
#             annotation_x_vals_larger[nearest_neighbor_too_close_logical[1:]] += space_factor * nearest_neighbor_too_close_logical[1:]
            
#             temp_arr = np.concatenate(([middle_gene_acrophase], annotation_x_vals_larger))
#             nearest_neighbor = np.diff(temp_arr)
#             nearest_neighbor_too_close_logical = nearest_neighbor < space_factor
#             too_close = np.any(nearest_neighbor_too_close_logical)

#     annotation_x_vals_smaller = np.copy(sorted_sig_smaller_phases)
#     if len(annotation_x_vals_smaller) > 0:
#         temp_arr = np.concatenate(([middle_gene_acrophase], annotation_x_vals_smaller))
#         nearest_neighbor = np.diff(temp_arr)

#         nearest_neighbor_too_close_logical = nearest_neighbor > -space_factor
#         too_close = np.any(nearest_neighbor_too_close_logical)
        
#         while too_close:
#             annotation_x_vals_smaller[nearest_neighbor_too_close_logical[1:]] -= space_factor * nearest_neighbor_too_close_logical[1:]
            
#             temp_arr = np.concatenate(([middle_gene_acrophase], annotation_x_vals_smaller))
#             nearest_neighbor = np.diff(temp_arr)
#             nearest_neighbor_too_close_logical = nearest_neighbor > -space_factor
#             too_close = np.any(nearest_neighbor_too_close_logical)

#     for mwm in range(len(sorted_sig_larger_phases)):
#         desired_phases = sorted_sig_larger_phases[mwm]
#         desired_annotation_phases = annotation_x_vals_larger[mwm]
#         desired_gene_index = np.where(plot_acrophases == desired_phases)[0][0]
#         plt.annotate(
#             Comparable_GOI[desired_gene_index], 
#             xy=[desired_phases, 1], 
#             xytext=[desired_annotation_phases, (1.35 if create_fig else 1.85)],
#             arrowprops=dict(arrowstyle="->"), 
#             fontsize=(12 if create_fig else subplot_fontsize)
#         )

#     for wmw in range(len(sorted_sig_smaller_phases)):
#         desired_phases = sorted_sig_smaller_phases[wmw]
#         desired_annotation_phases = annotation_x_vals_smaller[wmw]
#         desired_gene_index = np.where(plot_acrophases == desired_phases)[0][0]
#         plt.annotate(
#             Comparable_GOI[desired_gene_index], 
#             xy=[desired_phases, 1], 
#             xytext=[desired_annotation_phases, (1.35 if create_fig else 1.85)],
#             arrowprops=dict(arrowstyle="->"), 
#             fontsize=(12 if create_fig else subplot_fontsize)
#         )

#     ideal_annotation_x_vals_larger = np.copy(sorted_sig_larger_ideal_phases)
#     scaled_space_factor = space_factor * subplot_space_factor_ratio
    
#     if len(ideal_annotation_x_vals_larger) > 0:
#         temp_arr = np.concatenate(([ideal_middle_gene_acrophase], ideal_annotation_x_vals_larger))
#         nearest_neighbor = np.diff(temp_arr)
#         nearest_neighbor_too_close_logical = nearest_neighbor < scaled_space_factor
#         too_close = np.any(nearest_neighbor_too_close_logical)
        
#         while too_close:
#             ideal_annotation_x_vals_larger[nearest_neighbor_too_close_logical[1:]] += scaled_space_factor * nearest_neighbor_too_close_logical[1:]
            
#             temp_arr = np.concatenate(([ideal_middle_gene_acrophase], ideal_annotation_x_vals_larger))
#             nearest_neighbor = np.diff(temp_arr)
#             nearest_neighbor_too_close_logical = nearest_neighbor < scaled_space_factor
#             too_close = np.any(nearest_neighbor_too_close_logical)

#     ideal_annotation_x_vals_smaller = np.copy(sorted_sig_smaller_ideal_phases)
#     if len(ideal_annotation_x_vals_smaller) > 0:
#         temp_arr = np.concatenate(([ideal_middle_gene_acrophase], ideal_annotation_x_vals_smaller))
#         nearest_neighbor = np.diff(temp_arr)
#         nearest_neighbor_too_close_logical = nearest_neighbor > -scaled_space_factor
#         too_close = np.any(nearest_neighbor_too_close_logical)
        
#         while too_close:
#             ideal_annotation_x_vals_smaller[nearest_neighbor_too_close_logical[1:]] -= scaled_space_factor * nearest_neighbor_too_close_logical[1:]
            
#             temp_arr = np.concatenate(([ideal_middle_gene_acrophase], ideal_annotation_x_vals_smaller))
#             nearest_neighbor = np.diff(temp_arr)
#             nearest_neighbor_too_close_logical = nearest_neighbor > -scaled_space_factor
#             too_close = np.any(nearest_neighbor_too_close_logical)
#     c = 0
#     while c < len(sorted_sig_larger_ideal_phases):
#         desired_phases = sorted_sig_larger_ideal_phases[c]
#         desired_gene_indices = np.where(plot_ideals == desired_phases)[0]
#         sig_desired_gene_indices = desired_gene_indices
        
#         d = 0
#         while d < len(sig_desired_gene_indices):
#             if (c + d) < len(ideal_annotation_x_vals_larger):
#                 desired_annotation_phases = ideal_annotation_x_vals_larger[c + d]
#                 plt.annotate(
#                     Comparable_GOI_Ideal[sig_desired_gene_indices[d]], 
#                     xy=[desired_phases, 0.5], 
#                     xytext=[desired_annotation_phases, (0.75 if create_fig else 0.85)],
#                     arrowprops=dict(arrowstyle="->"), 
#                     fontsize=(12 if create_fig else subplot_fontsize)
#                 )
#             d += 1
#         c += (d if d > 0 else 1)

#     c = 0
#     while c < len(sorted_sig_smaller_ideal_phases):
#         desired_phases = sorted_sig_smaller_ideal_phases[c]
#         desired_gene_indices = np.where(plot_ideals == desired_phases)[0]
#         sig_desired_gene_indices = desired_gene_indices
        
#         d = 0
#         while d < len(sig_desired_gene_indices):
#             if (c + d) < len(ideal_annotation_x_vals_smaller):
#                 desired_annotation_phases = ideal_annotation_x_vals_smaller[c + d]
#                 plt.annotate(
#                     Comparable_GOI_Ideal[sig_desired_gene_indices[d]], 
#                     xy=[desired_phases, 0.5], 
#                     xytext=[desired_annotation_phases, (0.75 if create_fig else 0.85)],
#                     arrowprops=dict(arrowstyle="->"), 
#                     fontsize=(12 if create_fig else subplot_fontsize)
#                 )
#             d += 1
#         c += (d if d > 0 else 1)