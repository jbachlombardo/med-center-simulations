import numpy as np
import pandas as pd
from scipy import stats
import math
import time

t0 = time.clock()

# ===== FUNCTIONS =====

def generate_service_distribution_type_condition(data, condition_type, minimum = 2, skew = 0.05, n_samples = 10000) :
    # Average service time for condition
    avg = data.loc[data['Type'] == condition_type, 'Time_Mean'].item()
    # Worst case upper limit for condition
    worstcase = data.loc[data['Type'] == condition_type, 'Time_WorstCase'].item()
    # % of time worst case for condition
    perc_worst_case = data.loc[data['Type'] == condition_type, 'Perc_WorstCase'].item()
    # St Dev based on % of time worst case occurs
    std = worstcase / stats.skewnorm.ppf(1 - perc_worst_case, avg)
    # Create distribution of 1000 samples based on condition type parameters
    dist = stats.skewnorm.rvs(skew, loc = avg, scale = std, size = n_samples)
    # Remove negative / too low results
    dist[dist < avg / minimum] = avg / minimum
    return dist

def generate_service_distribution_process(data, process, minimum = 2, skew = 0.05, n_samples = 10000) :
    # Average service time for condition
    avg = data.loc[data['Process'] == process, 'Time_Mean'].item()
    # Worst case upper limit for condition
    worstcase = data.loc[data['Process'] == process, 'Time_WorstCase'].item()
    # % of time worst case for condition
    perc_worst_case = data.loc[data['Process'] == process, 'Perc_WorstCase'].item()
    # St Dev based on % of time worst case occurs
    std = worstcase / stats.skewnorm.ppf(1 - perc_worst_case, avg)
    # Create distribution of 1000 samples based on condition type parameters
    dist = stats.skewnorm.rvs(skew, loc = avg, scale = std, size = n_samples)
    # Remove negative / too low results
    dist[dist < avg / minimum] = avg / minimum
    return dist

def generate_service_time_type_condition(dist_dict) :
    """Get randomized service time based on type of condition
    To be used only if step in model is Exam by provider"""
    # Randomly choose type of condition, based on preexisting known probabilities
    cond_type = ['Chronic', 'Preventative', 'Acute']
    p_cond_type = [0.6, 0.2, 0.2]
    type = np.random.choice(cond_type, p = p_cond_type)
    # Return randomly chosen new service type from previously generated distributions
    serve_time = np.random.choice(dist_dict[type])
    return round(serve_time)

def generate_service_time_process(process, process_dict) :
    """Get randomized service time based on process
    To be used only all steps in model except Exam by provider"""
    serve_time = np.random.choice(process_dict[process])
    return round(serve_time)

def mark_service_time(dictionary_service, count_of_completed_service, service_time_tracker_dict, list_service_times_completed) :
    """Reduce the service time left for each patient to complete step by 1
    If service time  reduced to zero, remove patient from  dictionary, move to service completed, free up server
    Return modified service time dictionary
    In parallel, track completed service times to generate avg service time estimate post-simulation"""
    # Reduce service time left to complete step
    for k, v in dictionary_service.items() :
        if np.isnan(v) :
            continue
        else :
            dictionary_service[k] -= 1
            # Count patient as completed step and free up server if service time is 0
            if dictionary_service[k] == 0 :
                count_of_completed_service += 1
                dictionary_service[k] = np.nan
    # For patients who are marked as completed, track the actual service time
    # completion for noting after simulation
    keys_for_removal = list()
    for k in service_time_tracker_dict.keys() :
        service_time_tracker_dict[k][0] += 1
        if service_time_tracker_dict[k][0] == service_time_tracker_dict[k][1] :
            list_service_times_completed.append(service_time_tracker_dict[k][1])
            keys_for_removal.append(k)
    for k in keys_for_removal :
        del service_time_tracker_dict[k]
    return dictionary_service, count_of_completed_service, list_service_times_completed

def check_servers_free(dictionary_service) :
    """Verify if servers are free (denoted by np.nan as value for server key)
    And, if free servers exist, how many"""
    count_servers_free = 0
    for k, v in dictionary_service.items() :
        if np.isnan(v) :
            count_servers_free += 1
    return count_servers_free

def how_many_to_move_from_where(dictionary_waiting, count_servers_free, count_n_arrivals) :
    """Determine how many people to move from wait list to service / from new arrivals to service"""
    count_from_wait_list = min(len(dictionary_waiting.keys()), count_servers_free)
    count_servers_free = count_servers_free - count_from_wait_list
    count_from_new_arrivals = min(count_n_arrivals, count_servers_free)
    return count_from_wait_list, count_from_new_arrivals

def move_from_wait_list_to_service(dictionary_waiting, dictionary_service, count_from_wait_list, list_waiting_time, service_time_tracker_dict, process, empty_servers = True) :
    """Move patients from wait list to free servers
    To be preceded by: if count_from_wait_list > 0"""
    if empty_servers == True : # empty_servers = True for steps not using flow staff otherwise empty_servers pre-defined
        empty_servers = [k for k, v in dictionary_service.items() if np.isnan(v)]
    n_free_servers = len(empty_servers)
    patients_on_wait_list = list(dictionary_waiting.keys()) # Remove sort to fix bug with sorting 9 / 10
    patients_move_to_serve = patients_on_wait_list[:n_free_servers]
    if (len(patients_move_to_serve) > 0) and (len(empty_servers) > 0) :
        for i, m in enumerate(patients_move_to_serve) :
            server_for_m = empty_servers[i]
            list_waiting_time.append(dictionary_waiting[m])
            del dictionary_waiting[m]
            if process == 'Exam' :
                serve_time = generate_service_time_type_condition(distribs_dict_type)
            else :
                serve_time = generate_service_time_process(process, distribs_dict_process)
                if process == 'Refine_complaint' :
                    serve_time += pass_through_steps['Time_Mean'].sum()
            dictionary_service[server_for_m] = serve_time
            unique_key = 'Service_' + str(serve_time) + '_' + str(p)
            service_time_tracker_dict[unique_key] = [0, serve_time]
    return dictionary_waiting, dictionary_service, list_waiting_time, service_time_tracker_dict

def move_from_arrival_to_service(dictionary_service, count_from_new_arrivals, service_time_tracker_dict, process) :
    """Move patients from arrivals to free servers
    To be preceded by: if count_from_new_arrivals > 0 """
    empty_servers = [k for k, v in dictionary_service.items() if np.isnan(v)]
    arrivals_to_place = min(len(empty_servers), count_from_new_arrivals)
    if (arrivals_to_place > 0) and (len(empty_servers) > 0) :
        for i in range(arrivals_to_place) :
            server_for_m = empty_servers[i]
            if process == 'Exam' :
                serve_time = generate_service_time_type_condition(distribs_dict_type)
            else :
                serve_time = generate_service_time_process(process, distribs_dict_process)
                if process == 'Refine_complaint' :
                    serve_time += pass_through_steps['Time_Mean'].sum()
            dictionary_service[server_for_m] = serve_time
            unique_key = 'Service_' + str(serve_time) + '_' + str(p)
            service_time_tracker_dict[unique_key] = [0, serve_time]
    return dictionary_service, arrivals_to_place, service_time_tracker_dict

def add_to_wait_list(dictionary_waiting, count_n_arrivals, count_arrivals_placed) :
    """Move patients from arrivals to wait list
    To be preceded by: if arrivals_placed < n_arrivals"""
    count_diff = count_n_arrivals - count_arrivals_placed
    wait_keys = list(dictionary_waiting.keys()) # Remove sort to fix bug with sorting 9 / 10
    for i in range(count_diff) :
        if len(wait_keys) < 1 :
            new_waitname = 'Waiting' + str(i)
            dictionary_waiting[new_waitname] = 1
        else :
            count_lastwaiter = int(wait_keys[-1][7:])
            new_waitname = 'Waiting' + str(count_lastwaiter + i + 1)
            dictionary_waiting[new_waitname] = 1
    return dictionary_waiting

def generate_step_objects(n_servers, n_periods) :
    """Function to generate step tracking objects"""
    # Treat servers as dictionary to keep track of who is busy
    # NaN means empty server
    # If busy, dict will take form of {'Server#': n_minutes_left_service}
    dictionary_servers = {}
    for i in range(n_servers) :
        servname = 'Server' + str(i)
        dictionary_servers[servname] = np.nan
    # Treat waiting as dictionary
    # If someone waits, will be added to dictionary with form of {'Waiting#': n_minutes_waiting}
    dictionary_waiting = {}
    # Temporary tracker dictionary for service times
    dictionary_track_serve_time = {}
    # Holding lists for completed service times and completed waiting times (for measurement post-simulation)
    list_waiting_times = list()
    list_service_completed_times = list()
    # Set counter for completed service to 0
    count_service_completed = 0
    # Array for holding onto step-by-step process
    # Shape: number_of_periods x 4 -> [n_arrivals, n_being_served, n_waiting, n_completed]
    tracker = np.zeros(shape = (n_periods, 4))
    return dictionary_servers, dictionary_waiting, dictionary_track_serve_time, list_waiting_times, list_service_completed_times, count_service_completed, tracker

def max_try_except(wait_time_list) :
    """Function to calculate summary stat of maximum wait times
    Requires try/except block for np.max returning error when no wait times in list"""
    try :
        return np.max(wait_time_list)
    except :
        return 0

# Modified functions specific to v11 (flow staff tied to patient with offload from provider time)

def mark_service_time_flow_staff_steps(dictionary_service, count_of_completed_service, service_time_tracker_dict, list_service_times_completed, flowstaff = True) :
    """Reduce the service time left for each patient to complete step by 1
    If service time  reduced to zero, remove patient from  dictionary, move to service completed, free up server
    Return modified service time dictionary
    In parallel, track completed service times to generate avg service time estimate post-simulation"""
    # List of servers to move over (if patient completes step)
    move_to_next_step = list()
    # Reduce service time left to complete step
    for k, v in dictionary_service.items() :
        if np.isnan(v) :
            continue
        else :
            dictionary_service[k] -= 1
            # Count patient as completed step and free up server if service time is 0
            if dictionary_service[k] == 0 :
                count_of_completed_service += 1
                move_to_next_step.append(k)
                if flowstaff == True :
                    dictionary_service[k] = np.inf # Hold as np.inf until reset -- ensure server not double used
                else :
                    dictionary_service[k] = np.nan # For if function being used on provider service dict
    # For patients who are marked as completed, track the actual service time
    # completion for noting after simulation
    keys_for_removal = list()
    for k in service_time_tracker_dict.keys() :
        service_time_tracker_dict[k][0] += 1
        if service_time_tracker_dict[k][0] == service_time_tracker_dict[k][1] :
            list_service_times_completed.append(service_time_tracker_dict[k][1])
            keys_for_removal.append(k)
    for k in keys_for_removal :
        del service_time_tracker_dict[k]
    return dictionary_service, count_of_completed_service, move_to_next_step, list_service_times_completed

def move_from_wait_list_to_service_exam(dictionary_waiting, dictionary_service_provider, dictionary_service_flow_staff, count_from_wait_list, list_waiting_time, service_time_tracker_dict_flow_staff, service_time_tracker_dict_provider, offload) :
    """Move patients from wait list to free servers
    To be preceded by: if count_from_wait_list > 0"""
    empty_providers = [k for k, v in dictionary_service_provider.items() if np.isnan(v)]
    n_free_servers = len(empty_providers)
    patients_on_wait_list = sorted(dictionary_waiting, key = dictionary_waiting.get, reverse = True)
    patients_move_to_serve = patients_on_wait_list[:n_free_servers]
    if (len(patients_move_to_serve) > 0) and (len(empty_providers) > 0) :
        for i, m in enumerate(patients_move_to_serve) :
            # Identify servers to be used
            server_provider = empty_providers[i]
            server_flow_staff = m
            # Track waiting time and remove waiting patient from waiting dictionary
            list_waiting_time.append(dictionary_waiting[m])
            del dictionary_waiting[m]
            # Generate service time for flow_staff and provider
            total_serve_time = generate_service_time_type_condition(distribs_dict_type)
            serve_time_flow_staff = total_serve_time
            serve_time_provider = math.ceil(total_serve_time * (1 - offload))
            # Place into service for both provider and flow_staff
            dictionary_service_flow_staff[server_flow_staff] = serve_time_flow_staff
            dictionary_service_provider[server_provider] = serve_time_provider
            unique_key = 'Service_' + str(total_serve_time) + '_' + str(p)
            service_time_tracker_dict_flow_staff[unique_key] = [0, serve_time_flow_staff]
            service_time_tracker_dict_provider[unique_key] = [0, serve_time_provider]
    return dictionary_waiting, dictionary_service_provider, dictionary_service_flow_staff, list_waiting_time, service_time_tracker_dict_flow_staff, service_time_tracker_dict_provider

def move_from_arrival_to_service_exam(dictionary_service_provider, dictionary_service_flow_staff, count_from_new_arrivals, list_flow_staff, service_time_tracker_dict_flow_staff, service_time_tracker_dict_provider, offload) :
    """Move patients from arrivals to free servers
    To be preceded by: if count_from_new_arrivals > 0 """
    empty_providers = [k for k, v in dictionary_service_provider.items() if np.isnan(v)]
    arrivals_to_place = min(len(empty_providers), count_from_new_arrivals)
    if (arrivals_to_place > 0) and (len(empty_providers) > 0) :
        for i in range(arrivals_to_place) :
            # Identify servers to be used -- provider from service dict, flow_staff from transition list created in previous setp
            server_flow_staff = list_flow_staff[0]
            list_flow_staff.remove(server_flow_staff) # Remove flow_staff server from list to be transitioned
            server_provider = empty_providers[i]
            # Generate service time for flow_staff and provider
            total_serve_time = generate_service_time_type_condition(distribs_dict_type)
            serve_time_flow_staff = total_serve_time
            serve_time_provider = math.ceil(total_serve_time * (1 - offload))
            # Place into service for both provider and flow_staff
            dictionary_service_flow_staff[server_flow_staff] = serve_time_flow_staff
            dictionary_service_provider[server_provider] = serve_time_provider
            unique_key = 'Service_' + str(total_serve_time) + '_' + str(p)
            service_time_tracker_dict_flow_staff[unique_key] = [0, serve_time_flow_staff]
            service_time_tracker_dict_provider[unique_key] = [0, serve_time_provider]
    return dictionary_service_provider, dictionary_service_flow_staff, arrivals_to_place, list_flow_staff, service_time_tracker_dict_flow_staff, service_time_tracker_dict_provider

def move_from_arrival_to_service_follow_up(dictionary_service, count_from_new_arrivals, list_flow_staff, service_time_tracker_dict, process) :
    """Move patients from arrivals to free servers
    To be preceded by: if count_from_new_arrivals > 0 """
    for i in range(count_from_new_arrivals) :
        # Identify servers to be used -- provider from service dict, flow_staff from transition list created in previous setp
        server = list_flow_staff[0]
        list_flow_staff.remove(server) # Remove flow_staff server from list to be transitioned
        # Generate service time for flow_staff and provider
        serve_time = generate_service_time_process(process, distribs_dict_process)
        # Place into service for both provider and flow_staff
        dictionary_service[server] = serve_time
        unique_key = 'Service_' + str(serve_time) + '_' + str(p)
        service_time_tracker_dict[unique_key] = [0, serve_time]
    return dictionary_service, service_time_tracker_dict

def add_to_wait_list_exam(dictionary_waiting, server_list) :
    """Move patients from arrivals to wait list
    To be preceded by: if arrivals_placed < n_arrivals"""
    for s in server_list :
        dictionary_waiting[s] = 1
    return dictionary_waiting

def reset_flowstaff_server(server_list, dict1, dict2, dict3) :
    """Reset flowstaff to all np.nan so can pick up new patient in refine_complaint"""
    for s in server_list :
        dict1[s] = np.nan
        dict2[s] = np.nan
        dict3[s] = np.nan
    return dict1, dict2, dict3

# ===== MODEL PARAMETERS =====

# Estimates for how often the worst case scenario in patient service times occurs
low = 0.05
medium = 0.1
high = 0.2

# Providers available
providers = {'Doctor': 9, 'Nurse': 5, 'FlowStaff': 21, 'CSR': 10}

# Time estimates for each step in the patient journey
# Time estimates for Exam by provider (currently NaN's) to come from times by condition type (below)
# Consolidated by Step and Server to simplify
# -key assumption1 : only preserving variability on steps that have variability built-in, rather than waiting time
# -key assumption2 : steps that have no WorstCase are straight pass throughs and added to overall service time,
#                    so only steps to be considered for queueing variability are check_in, refine_complaint,
#                    exam, and checkout
# -key assumption3 : CSR are split between check_in and check_out
cols = ['Step', 'Process', 'Staff', 'Time_Mean', 'Time_WorstCase', 'Perc_WorstCase']
checkin = ['Arrive', 'Check_in', 'CSR', 2, 3, low]
wait = ['Arrive', 'Waiting_room', 'FlowStaff', 3, 0, low]
to_room = ['Arrive', 'To_exam_room', 'FlowStaff', 1, 0, low]
vitals = ['Exam_prep', 'Vitals_check', 'FlowStaff', 2, 0, low]
refine_complaint = ['Exam_prep', 'Refine_complaint', 'FlowStaff', 15, 15, high] # Low for covid test, default high
start_note = ['Exam_prep', 'Start_note', 'FlowStaff', 1, 0, low]
exam = ['Exam_provider', 'Exam', 'Doctor', np.nan, np.nan, np.nan]
follow_up = ['Exam_follow_up', 'Follow_up', 'FlowStaff', 5, 5, low]
checkout = ['Conclude', 'Checkout', 'CSR', 5, 5, medium]
process_flow = pd.DataFrame([checkin, wait, to_room, vitals, refine_complaint, start_note, exam, follow_up, checkout], columns = cols)
process_flow['Servers'] = process_flow['Staff'].map(providers)
process_flow.loc[process_flow['Staff'] == 'CSR', 'Servers'] = (process_flow.loc[process_flow['Staff'] == 'CSR', 'Servers'] / 2).astype(int)
pass_through_steps = process_flow.loc[process_flow['Time_WorstCase'] == 0]
variable_steps = process_flow.loc[process_flow['Time_WorstCase'] != 0]

# Data table for drawing service times for patients
cols = ['Type', 'Frequency', 'Time_Mean', 'Time_WorstCase', 'Perc_WorstCase']
preventative = ['Preventative', 0.2, 30, 30, low]
chronic = ['Chronic', 0.6, 30, 30, medium]
acute = ['Acute', 0.2, 15, 5, low]
base_case_types = pd.DataFrame([preventative, chronic, acute], columns = cols)

# Arrivals to be modeled as poisson, scaling down to minute arrivals
arrivals_day = 130
arrivals_hour = arrivals_day / 7 # Lunch break modeling
arrivals_minute = arrivals_hour / 60
arrivals_quarterhour = arrivals_hour / 4

# Model full day of continuous operations
# Minutes are periods for analysis
hours = 10
n_periods = int(60 * hours)

# Offload percentage
offload = 0.25

# List of processes to be cycled through
processes_with_variability = variable_steps['Process'].to_list()

# ===== LOOPED MODEL =====

# Set up sim repositories

n_sims = 10000

sims_arrivals_check_in = np.empty(shape = n_sims)
sims_served_check_in = np.empty(shape = n_sims)
sims_mean_serve_time_check_in = np.empty(shape = n_sims)
sims_mean_waiting_time_check_in = np.empty(shape = n_sims)
sims_max_waiting_time_check_in = np.empty(shape = n_sims)

sims_arrivals_refine_complaint = np.empty(shape = n_sims)
sims_served_refine_complaint = np.empty(shape = n_sims)
sims_mean_serve_time_refine_complaint = np.empty(shape = n_sims)
sims_mean_waiting_time_refine_complaint = np.empty(shape = n_sims)
sims_max_waiting_time_refine_complaint = np.empty(shape = n_sims)

sims_arrivals_exam = np.empty(shape = n_sims)
sims_served_exam = np.empty(shape = n_sims)
sims_mean_serve_time_exam = np.empty(shape = n_sims)
sims_mean_serve_time_exam_flow_staff = np.empty(shape = n_sims)
sims_mean_waiting_time_exam = np.empty(shape = n_sims)
sims_max_waiting_time_exam = np.empty(shape = n_sims)

sims_arrivals_follow_up = np.empty(shape = n_sims)
sims_served_follow_up = np.empty(shape = n_sims)
sims_mean_serve_time_follow_up = np.empty(shape = n_sims)
sims_mean_waiting_time_follow_up = np.empty(shape = n_sims)
sims_max_waiting_time_follow_up = np.empty(shape = n_sims)

sims_arrivals_checkout = np.empty(shape = n_sims)
sims_served_checkout = np.empty(shape = n_sims)
sims_mean_serve_time_checkout = np.empty(shape = n_sims)
sims_mean_waiting_time_checkout = np.empty(shape = n_sims)
sims_max_waiting_time_checkout = np.empty(shape = n_sims)

sims_fs_utilization = np.empty(shape = n_sims)
sims_fs_waiting = np.empty(shape = n_sims)

for sim in range(n_sims) :

    if sim % 100 == 0 :
        print ('{:.1f}%'.format(sim/n_sims*100))

    # ===== HOLDERS CREATION =====

    # Holders for check_in step
    servers_check_in = variable_steps.loc[variable_steps['Process'] == processes_with_variability[0], 'Servers'].item()

    server_dict_check_in, waiting_dict_check_in, serve_time_track_dict_check_in, waiting_time_list_check_in, service_times_completed_list_check_in, service_completed_check_in, tracker_check_in = generate_step_objects(servers_check_in, n_periods)

    # Holders for refine_complaint step
    servers_refine_complaint = variable_steps.loc[variable_steps['Process'] == processes_with_variability[1], 'Servers'].item()

    server_dict_refine_complaint, waiting_dict_refine_complaint, serve_time_track_dict_refine_complaint, waiting_time_list_refine_complaint, service_times_completed_list_refine_complaint, service_completed_refine_complaint, tracker_refine_complaint = generate_step_objects(servers_refine_complaint, n_periods)

    # Holders for exam step
    servers_exam_provider = variable_steps.loc[variable_steps['Process'] == processes_with_variability[2], 'Servers'].item()

    server_dict_exam_provider, waiting_dict_exam_provider, serve_time_track_dict_exam_provider, waiting_time_list_exam_provider, service_times_completed_list_exam_provider, service_completed_exam_provider, tracker_exam = generate_step_objects(servers_exam_provider, n_periods)

    servers_exam_flow_staff = variable_steps.loc[variable_steps['Process'] == processes_with_variability[1], 'Servers'].item()

    server_dict_exam_flow_staff, waiting_dict_exam_flow_staff, serve_time_track_dict_exam_flow_staff, waiting_time_list_exam_flow_staff, service_times_completed_list_exam_flow_staff, service_completed_exam_flow_staff, _ = generate_step_objects(servers_exam_flow_staff, n_periods)

    # Holders for follow_up step
    servers_follow_up = variable_steps.loc[variable_steps['Process'] == processes_with_variability[3], 'Servers'].item()

    server_dict_follow_up, waiting_dict_follow_up, serve_time_track_dict_follow_up, waiting_time_list_follow_up, service_times_completed_list_follow_up, service_completed_follow_up, tracker_follow_up = generate_step_objects(servers_follow_up, n_periods)

    # Holders for checkout step
    servers_checkout = variable_steps.loc[variable_steps['Process'] == processes_with_variability[4], 'Servers'].item()

    server_dict_checkout, waiting_dict_checkout, serve_time_track_dict_checkout, waiting_time_list_checkout, service_times_completed_list_checkout, service_completed_checkout, tracker_checkout = generate_step_objects(servers_checkout, n_periods)

    # ===== DISTRIBUTION CREATION =====

    # For processes with variability
    check_in_distribution = generate_service_distribution_process(process_flow, processes_with_variability[0])
    refine_complaint_distribution = generate_service_distribution_process(process_flow, processes_with_variability[1])
    follow_up_distribution = generate_service_distribution_process(process_flow, processes_with_variability[3])
    checkout_distribution = generate_service_distribution_process(process_flow, processes_with_variability[4])
    distribs_dict_process = {}
    for k, v in zip(processes_with_variability, [check_in_distribution, refine_complaint_distribution, np.nan, follow_up_distribution, checkout_distribution]) :
        distribs_dict_process[k] = v

    # For condition types
    acute_distribution = generate_service_distribution_type_condition(base_case_types, 'Acute', skew = 0.05, n_samples = 10000)
    chronic_distribution = generate_service_distribution_type_condition(base_case_types, 'Chronic', skew = 0.05, n_samples = 10000)
    preventative_distribution = generate_service_distribution_type_condition(base_case_types, 'Preventative', skew = 0.05, n_samples = 10000)
    distribs_dict_type = {}
    for k, v in zip(['Acute', 'Chronic', 'Preventative'], [acute_distribution, chronic_distribution, preventative_distribution]) :
        distribs_dict_type[k] = v

    # ===== RUN SIMULATION BY MINUTE =====

    flowstaff_usage = list()

    for p in range(n_periods) :

        # ARRIVALS INTO SYSTEM

        # ===== Modeled as set number for testing =====
        if p % 15 == 0 :
            if (p >= 0 and p < 180) or (p >= 240 and p < 479) :
                n_arrivals_check_in = int(round(arrivals_quarterhour))
        else :
            n_arrivals_check_in = 0

        # # ===== Modeled as poisson =====
        # if p % 15 == 0 :
        #     if (p >= 0 and p < 180) or (p >= 240 and p < 479) :
        #         n_arrivals_check_in = np.random.poisson(arrivals_quarterhour)
        # else :
        #     n_arrivals_check_in = 0

        # CHECK-IN
        server_dict_check_in, service_completed_check_in, service_times_completed_list_check_in =  mark_service_time(server_dict_check_in, service_completed_check_in, serve_time_track_dict_check_in, service_times_completed_list_check_in)
        n_servers_free_check_in = check_servers_free(server_dict_check_in)
        from_wait_list_check_in, from_new_arrivals_check_in = how_many_to_move_from_where(waiting_dict_check_in, n_servers_free_check_in, n_arrivals_check_in)
        if from_wait_list_check_in > 0 :
            waiting_dict_check_in, server_dict_check_in, waiting_time_list_check_in, serve_time_track_dict_check_in = move_from_wait_list_to_service(waiting_dict_check_in, server_dict_check_in, from_wait_list_check_in, waiting_time_list_check_in, serve_time_track_dict_check_in, processes_with_variability[0])
        if from_new_arrivals_check_in > 0 :
            server_dict_check_in, n_arrivals_placed_check_in, serve_time_track_dict_check_in =  move_from_arrival_to_service(server_dict_check_in, from_new_arrivals_check_in, serve_time_track_dict_check_in, processes_with_variability[0])
        else :
            n_arrivals_placed_check_in = 0
        waiting_dict_check_in = {k:v + 1 for k, v in waiting_dict_check_in.items()}
        waiting_dict_check_in = add_to_wait_list(waiting_dict_check_in, n_arrivals_check_in, n_arrivals_placed_check_in)
        tracker_check_in[p] = [n_arrivals_check_in, len([v for v in server_dict_check_in.values() if np.isfinite(v)]), len(waiting_dict_check_in.keys()), service_completed_check_in]

        # REFINE COMPLAINT STEP
        n_arrivals_refine_complaint = service_completed_check_in
        service_completed_check_in = 0
        server_dict_refine_complaint, service_completed_refine_complaint, servers_move_next_step_refine_complaint, service_times_completed_list_refine_complaint = mark_service_time_flow_staff_steps(server_dict_refine_complaint, service_completed_refine_complaint, serve_time_track_dict_refine_complaint, service_times_completed_list_refine_complaint)
        n_servers_free_refine_complaint = check_servers_free(server_dict_refine_complaint)
        from_wait_list_refine_complaint, from_new_arrivals_refine_complaint = how_many_to_move_from_where( waiting_dict_refine_complaint, n_servers_free_refine_complaint, n_arrivals_refine_complaint)
        if from_wait_list_refine_complaint > 0 :
            waiting_dict_refine_complaint, server_dict_refine_complaint, waiting_time_list_refine_complaint, serve_time_track_dict_refine_complaint = move_from_wait_list_to_service(waiting_dict_refine_complaint, server_dict_refine_complaint, from_wait_list_refine_complaint, waiting_time_list_refine_complaint, serve_time_track_dict_refine_complaint, processes_with_variability[1])
        if from_new_arrivals_refine_complaint > 0 :
            server_dict_refine_complaint, n_arrivals_placed_refine_complaint, serve_time_track_dict_refine_complaint = move_from_arrival_to_service(server_dict_refine_complaint, from_new_arrivals_refine_complaint, serve_time_track_dict_refine_complaint, processes_with_variability[1])
        else :
            n_arrivals_placed_refine_complaint = 0
        waiting_dict_refine_complaint = {k:v + 1 for k, v in waiting_dict_refine_complaint.items()}
        waiting_dict_refine_complaint = add_to_wait_list(waiting_dict_refine_complaint, n_arrivals_refine_complaint, n_arrivals_placed_refine_complaint)
        tracker_refine_complaint[p] = [n_arrivals_refine_complaint, len([v for v in server_dict_refine_complaint.values() if np.isfinite(v)]), len(waiting_dict_refine_complaint.keys()), service_completed_refine_complaint]

        # EXAM STEP == FLOWSTAFF & PROVIDERS DICTS
        n_arrivals_exam = service_completed_refine_complaint
        service_completed_refine_complaint = 0 # Reset counter to 0 for next transition
        server_dict_exam_provider, _, _, service_times_completed_list_exam_provider = mark_service_time_flow_staff_steps(server_dict_exam_provider, service_completed_exam_provider, serve_time_track_dict_exam_provider, service_times_completed_list_exam_provider, flowstaff = False)
        server_dict_exam_flow_staff, service_completed_exam_flow_staff, servers_move_next_step_exam_flow_staff, service_times_completed_list_exam_flow_staff = mark_service_time_flow_staff_steps(server_dict_exam_flow_staff, service_completed_exam_flow_staff, serve_time_track_dict_exam_flow_staff, service_times_completed_list_exam_flow_staff)
        n_servers_free_exam_provider = check_servers_free(server_dict_exam_provider)
        from_wait_list_exam, from_new_arrivals_exam = how_many_to_move_from_where(waiting_dict_exam_provider, n_servers_free_exam_provider, n_arrivals_exam)
        if from_wait_list_exam > 0 :
            waiting_dict_exam_provider, server_dict_exam_provider, server_dict_exam_flow_staff, waiting_time_list_exam_provider, serve_time_track_dict_exam_flow_staff, serve_time_track_dict_exam_provider = move_from_wait_list_to_service_exam(waiting_dict_exam_provider, server_dict_exam_provider, server_dict_exam_flow_staff, from_wait_list_exam, waiting_time_list_exam_provider, serve_time_track_dict_exam_flow_staff, serve_time_track_dict_exam_provider, offload)
        if from_new_arrivals_exam > 0 :
            server_dict_exam_provider, server_dict_exam_flow_staff, n_arrivals_placed_exam, servers_move_next_step_refine_complaint, serve_time_track_dict_exam_flow_staff, serve_time_track_dict_exam_provider = move_from_arrival_to_service_exam(server_dict_exam_provider, server_dict_exam_flow_staff, from_new_arrivals_exam, servers_move_next_step_refine_complaint, serve_time_track_dict_exam_flow_staff, serve_time_track_dict_exam_provider, offload)
        else :
            n_arrivals_placed_exam = 0
        waiting_dict_exam_provider = {k:v + 1 for k, v in waiting_dict_exam_provider.items()}
        waiting_dict_exam_provider = add_to_wait_list_exam(waiting_dict_exam_provider, servers_move_next_step_refine_complaint)
        tracker_exam[p] = [n_arrivals_exam, len([v for v in server_dict_exam_provider.values() if np.isfinite(v)]), len(waiting_dict_exam_provider.keys()), service_completed_exam_flow_staff]

        # FOLLOW UP == DIRECT FROM EXAM TO FLOW STAFF MATCHING SERVERS
        n_arrivals_follow_up = service_completed_exam_flow_staff
        service_completed_exam_flow_staff = 0
        server_dict_follow_up, service_completed_follow_up, servers_move_next_step_follow_up, service_times_completed_list_follow_up = mark_service_time_flow_staff_steps(server_dict_follow_up, service_completed_follow_up, serve_time_track_dict_follow_up, service_times_completed_list_follow_up)
        if n_arrivals_follow_up > 0 :
            server_dict_follow_up, serve_time_track_dict_follow_up = move_from_arrival_to_service_follow_up(server_dict_follow_up, n_arrivals_follow_up, servers_move_next_step_exam_flow_staff, serve_time_track_dict_follow_up, processes_with_variability[3])
        # =====
        # Track flowstaff usage
        n_fs_rc_in_use = len([k for k, v in server_dict_refine_complaint.items() if np.isfinite(v)])
        n_fs_ex_in_use = len([k for k, v in server_dict_exam_flow_staff.items() if np.isfinite(v)])
        n_fs_fu_in_use = len([k for k, v in server_dict_follow_up.items() if np.isfinite(v)])
        flowstaff_usage.append([n_fs_rc_in_use, n_fs_ex_in_use, n_fs_fu_in_use])
        # =====
        if len(servers_move_next_step_follow_up) > 0 :
            server_dict_refine_complaint, server_dict_exam_flow_staff, server_dict_follow_up = reset_flowstaff_server(servers_move_next_step_follow_up, server_dict_refine_complaint, server_dict_exam_flow_staff, server_dict_follow_up)
        tracker_follow_up[p] = [n_arrivals_follow_up, len([v for v in server_dict_follow_up.values() if np.isfinite(v)]), len(waiting_dict_follow_up.keys()), service_completed_follow_up]

        # CHECK OUT
        n_arrivals_checkout = service_completed_follow_up
        service_completed_follow_up = 0 # Reset counter to 0 for next transition
        server_dict_checkout, service_completed_checkout, service_times_completed_list_checkout = mark_service_time(server_dict_checkout, service_completed_checkout, serve_time_track_dict_checkout, service_times_completed_list_checkout)
        n_servers_free_checkout = check_servers_free(server_dict_checkout)
        from_wait_list_checkout, from_new_arrivals_checkout = how_many_to_move_from_where(waiting_dict_checkout, n_servers_free_checkout, n_arrivals_checkout)
        if from_wait_list_checkout > 0 :
            waiting_dict_checkout, server_dict_checkout, waiting_time_list_checkout, serve_time_track_dict_checkout = move_from_wait_list_to_service(waiting_dict_checkout, server_dict_checkout, from_wait_list_checkout, waiting_time_list_checkout, serve_time_track_dict_checkout, processes_with_variability[4])
        if from_new_arrivals_checkout > 0 :
            server_dict_checkout, n_arrivals_placed_checkout, serve_time_track_dict_checkout = move_from_arrival_to_service(server_dict_checkout, from_new_arrivals_checkout, serve_time_track_dict_checkout, processes_with_variability[4])
        else :
            n_arrivals_placed_checkout = 0
        waiting_dict_checkout = {k:v + 1 for k, v in waiting_dict_checkout.items()}
        waiting_dict_checkout = add_to_wait_list(waiting_dict_checkout, n_arrivals_checkout, n_arrivals_placed_checkout)
        tracker_checkout[p] = [n_arrivals_checkout, len([v for v in server_dict_checkout.values() if np.isfinite(v)]), len(waiting_dict_checkout.keys()), service_completed_checkout]
        service_completed_checkout = 0

    # Summary stats for check_in
    arrived_check_in_final = tracker_check_in[:, 0].sum()
    served_check_in_final = tracker_check_in[:, 3].sum()
    mean_service_time_check_in_final = np.mean(service_times_completed_list_check_in)
    mean_patients_waiting_check_in_final = tracker_check_in[:, 2].sum()
    mean_waiting_time_check_in_final = np.sum(waiting_time_list_check_in) / served_check_in_final
    max_waiting_time_check_in_final = max_try_except(waiting_time_list_check_in)

    # Summary stats for refine_complaint
    arrived_refine_complaint_final = tracker_refine_complaint[:, 0].sum()
    served_refine_complaint_final = tracker_refine_complaint[:, 3].sum()
    mean_service_time_refine_complaint_final = np.mean(service_times_completed_list_refine_complaint)
    mean_patients_waiting_refine_complaint_final = tracker_refine_complaint[:, 2].sum()
    mean_waiting_time_refine_complaint_final = np.sum(waiting_time_list_refine_complaint) / served_refine_complaint_final
    max_waiting_time_refine_complaint_final = max_try_except(waiting_time_list_refine_complaint)

    # Summary stats for exam
    arrived_exam_final = tracker_exam[:, 0].sum()
    served_exam_final = tracker_exam[:, 3].sum()
    mean_service_time_exam_final = np.mean(service_times_completed_list_exam_provider)
    mean_service_time_exam_final_flow_staff = np.mean(service_times_completed_list_exam_flow_staff)
    mean_patients_waiting_exam_final = tracker_exam[:, 2].sum()
    mean_waiting_time_exam_final = np.sum(waiting_time_list_exam_provider) / served_exam_final
    max_waiting_time_exam_final = max_try_except(waiting_time_list_exam_provider)

    # Summary stats for follow_up
    arrived_follow_up_final = tracker_follow_up[:, 0].sum()
    served_follow_up_final = tracker_follow_up[:, 3].sum()
    mean_service_time_follow_up_final = np.mean(service_times_completed_list_follow_up)
    mean_patients_waiting_follow_up_final = tracker_follow_up[:, 2].sum()
    mean_waiting_time_follow_up_final = np.sum(waiting_time_list_follow_up) / served_follow_up_final
    max_waiting_time_follow_up_final = max_try_except(waiting_time_list_follow_up)

    # Summary stats for checkout
    arrived_checkout_final = tracker_checkout[:, 0].sum()
    served_checkout_final = tracker_checkout[:, 3].sum()
    mean_service_time_checkout_final = np.mean(service_times_completed_list_checkout)
    mean_patients_waiting_checkout_final = tracker_checkout[:, 2].sum()
    mean_waiting_time_checkout_final = np.sum(waiting_time_list_checkout) / served_checkout_final
    max_waiting_time_checkout_final = max_try_except(waiting_time_list_checkout)

    # STEP-BY-STEP
    # check_in
    sims_arrivals_check_in[sim] = arrived_check_in_final
    sims_served_check_in[sim] = served_check_in_final
    sims_mean_serve_time_check_in[sim] = mean_service_time_check_in_final
    sims_mean_waiting_time_check_in[sim] = mean_waiting_time_check_in_final
    sims_max_waiting_time_check_in[sim] = max_waiting_time_check_in_final
    # refine_complaint
    sims_arrivals_refine_complaint[sim] = arrived_refine_complaint_final
    sims_served_refine_complaint[sim] = served_refine_complaint_final
    sims_mean_serve_time_refine_complaint[sim] = mean_service_time_refine_complaint_final
    sims_mean_waiting_time_refine_complaint[sim] = mean_waiting_time_refine_complaint_final
    sims_max_waiting_time_refine_complaint[sim] = max_waiting_time_refine_complaint_final
    # exam
    sims_arrivals_exam[sim] = arrived_exam_final
    sims_served_exam[sim] = served_exam_final
    sims_mean_serve_time_exam[sim] = mean_service_time_exam_final
    sims_mean_serve_time_exam_flow_staff[sim] = mean_service_time_exam_final_flow_staff
    sims_mean_waiting_time_exam[sim] = mean_waiting_time_exam_final
    sims_max_waiting_time_exam[sim] = max_waiting_time_exam_final
    # follow_up
    sims_arrivals_follow_up[sim] = arrived_follow_up_final
    sims_served_follow_up[sim] = served_follow_up_final
    sims_mean_serve_time_follow_up[sim] = mean_service_time_follow_up_final
    sims_mean_waiting_time_follow_up[sim] = mean_waiting_time_follow_up_final
    sims_max_waiting_time_follow_up[sim] = max_waiting_time_follow_up_final
    # checkout
    sims_arrivals_checkout[sim] = arrived_checkout_final
    sims_served_checkout[sim] = served_checkout_final
    sims_mean_serve_time_checkout[sim] = mean_service_time_checkout_final
    sims_mean_waiting_time_checkout[sim] = mean_waiting_time_checkout_final
    sims_max_waiting_time_checkout[sim] = max_waiting_time_checkout_final
    # utilization tracking
    mask = [np.sum(x) != 0 for x in flowstaff_usage]
    mean_fs_waiting = tracker_check_in[:, 3][mask].mean()
    mean_fs_utilization = np.mean([np.sum(x) for x in flowstaff_usage if np.sum(x) != 0])
    sims_fs_utilization[sim] = mean_fs_utilization
    sims_fs_waiting[sim] = mean_fs_waiting

columns = ['Arrivals_check_in', 'Served_check_in', 'Mean_serve_time_check_in', 'Mean_wait_time_check_in', 'Max_wait_time_check_in', 'Arrivals_refine_complaint', 'Served_refine_complaint', 'Mean_serve_time_refine_complaint', 'Mean_wait_time_refine_complaint', 'Max_wait_time_refine_complaint', 'Arrivals_exam', 'Served_exam', 'Mean_serve_time_exam', 'Mean_serve_time_exam_flow_staff', 'Mean_wait_time_exam', 'Max_wait_time_exam', 'Arrivals_follow_up', 'Served_follow_up', 'Mean_serve_time_follow_up', 'Mean_wait_time_follow_up', 'Max_wait_time_follow_up', 'Arrivals_checkout', 'Served_checkout', 'Mean_serve_time_checkout', 'Mean_wait_time_checkout', 'Max_wait_time_checkout', 'Mean_flowstaff_utilization', 'Mean_flowstaff_waiting']

data = [sims_arrivals_check_in, sims_served_check_in, sims_mean_serve_time_check_in, sims_mean_waiting_time_check_in, sims_max_waiting_time_check_in, sims_arrivals_refine_complaint, sims_served_refine_complaint, sims_mean_serve_time_refine_complaint, sims_mean_waiting_time_refine_complaint, sims_max_waiting_time_refine_complaint, sims_arrivals_exam, sims_served_exam, sims_mean_serve_time_exam, sims_mean_serve_time_exam_flow_staff, sims_mean_waiting_time_exam, sims_max_waiting_time_exam, sims_arrivals_follow_up, sims_served_follow_up, sims_mean_serve_time_follow_up, sims_mean_waiting_time_follow_up, sims_max_waiting_time_follow_up, sims_arrivals_checkout, sims_served_checkout, sims_mean_serve_time_checkout, sims_mean_waiting_time_checkout, sims_max_waiting_time_checkout, sims_fs_utilization, sims_fs_waiting]

final_results = pd.DataFrame(data, index = columns)
final_results.columns = ['Sim_' + str(s) for s in range(n_sims)]
final_results = final_results.T
# Calculate mean system times
final_results['Mean_system_time_check_in'] = final_results[['Mean_serve_time_check_in', 'Mean_wait_time_check_in']].sum(axis = 1)
final_results['Mean_system_time_refine_complaint'] = final_results[['Mean_serve_time_refine_complaint', 'Mean_wait_time_refine_complaint']].sum(axis = 1)
final_results['Mean_system_time_exam'] = final_results[['Mean_serve_time_exam', 'Mean_wait_time_exam']].sum(axis = 1)
final_results['Mean_system_time_follow_up'] = final_results[['Mean_serve_time_follow_up', 'Mean_wait_time_follow_up']].sum(axis = 1)
final_results['Mean_system_time_checkout'] = final_results[['Mean_serve_time_checkout', 'Mean_wait_time_checkout']].sum(axis = 1)
final_results['Total_service_time'] = final_results[['Mean_serve_time_check_in', 'Mean_serve_time_refine_complaint', 'Mean_serve_time_exam', 'Mean_serve_time_follow_up', 'Mean_serve_time_checkout']].sum(axis = 1)
final_results['Total_wait_time'] = final_results[['Mean_wait_time_check_in', 'Mean_wait_time_refine_complaint', 'Mean_wait_time_exam', 'Mean_wait_time_follow_up', 'Mean_wait_time_checkout']].sum(axis = 1)
final_results['Total_system_time'] = final_results[['Total_service_time', 'Total_wait_time']].sum(axis = 1)
# Calculate throughputs
final_results['Thruput_check_in'] = final_results['Served_check_in'] / final_results['Arrivals_check_in']
final_results['Thruput_refine_complaint'] = final_results['Served_refine_complaint'] / final_results['Arrivals_refine_complaint']
final_results['Thruput_exam'] = final_results['Served_exam'] / final_results['Arrivals_exam']
final_results['Thruput_follow_up'] = final_results['Served_follow_up'] / final_results['Arrivals_follow_up']
final_results['Thruput_checkout'] = final_results['Served_checkout'] / final_results['Arrivals_checkout']
final_results['Thruput_total'] = final_results['Served_checkout'] / final_results['Arrivals_check_in']

final_results['Staffing'] = '10-21-9'
final_results['Offload'] = offload

final_results.to_csv('/Users/jbachlombardo/OneDrive - INSEAD/Coursework/P5/Analytics ISP/Results of sims/Sims/200614_offload_staff10219_offload25.csv') ## COVID NAMING CONVENTION
# final_results.T.to_csv('/Users/jbachlombardo/OneDrive - INSEAD/Coursework/P5/Analytics ISP/Results of sims/Sims/200614_offload_test.csv') ## TEST

t1 = time.clock()

print ('Time', t1 - t0)
print ('{:.2f} sims'.format(n_sims))
print ('{:.2f} s / sim'.format((t1 - t0) / n_sims))
