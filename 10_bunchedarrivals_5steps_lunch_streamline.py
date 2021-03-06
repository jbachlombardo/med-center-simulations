import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import time

t0 = time.clock()

# ===== FUNCTIONS =====

def generate_service_distribution_type_condition(data, condition_type, skew = 0.05, n_samples = 10000) :
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
    dist[dist < avg / 2] = avg / 2
    return dist

def generate_service_distribution_process(data, process, skew = 0.05, n_samples = 10000) :
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
    dist[dist < avg / 2] = avg / 2
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

def move_from_arrival_to_service(dictionary_service, count_n_arrivals, count_from_new_arrivals, service_time_tracker_dict, process, empty_servers = True) :
    """Move patients from arrivals to free servers
    To be preceded by: if count_from_new_arrivals > 0 """
    if empty_servers == True : # empty_servers = True for steps not using flow staff otherwise empty_servers pre-defined
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

def available_flow_staff(dict1, dict2) :
    """Function to maintain an ongoing list of the available flow staff servers
    Developed because flow staff used in both refine_complaint and follow_up steps"""
    available = list(set([k for k, v in dict1.items() if np.isnan(v)]) - (set([k for k, v in\
                dict1.items() if np.isnan(v)]) - set([k for k, v in dict2.items() if np.isnan(v)])))
    return available

def max_try_except(wait_time_list) :
    """Function to calculate summary stat of maximum wait times
    Requires try/except block for np.max returning error when no wait times in list"""
    try :
        return np.max(wait_time_list)
    except :
        return 0

# ===== MODEL PARAMETERS =====

# Estimates for how often the worst case scenario in patient service times occurs
low = 0.05
medium = 0.1
high = 0.2

# Providers available
providers = {'Doctor': 9, 'Nurse': 5, 'FlowStaff': 20, 'CSR': 10} # Base 122
# providers = {'Doctor': 2, 'Nurse': 2, 'FlowStaff': 3, 'CSR': 2} # Covid 122

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

# # ===== COVID VERSION ===== Data table for drawing service times for patients
# cols = ['Type', 'Frequency', 'Time_Mean', 'Time_WorstCase', 'Perc_WorstCase']
# preventative = ['Preventative', 0.0, 30, 30, low]
# chronic = ['Chronic', 0.0, 30, 30, medium]
# acute = ['Acute', 1, 15, 5, high]
# base_case_types = pd.DataFrame([preventative, chronic, acute], columns = cols)

# Arrivals to be modeled as poisson, scaling down to minute arrivals
arrivals_day = 130 #* 0.05 # 5% arrivals = possibly covid
arrivals_hour = arrivals_day / 7 # NOTE: Change for lunch break modeling
arrivals_minute = arrivals_hour / 60
arrivals_quarterhour = arrivals_hour / 4
arrivals_quarterhour_sigma = 0.5

# Model full day of continuous operations
# Minutes are periods for analysis
# NOTE: Might need to adjust for morning / afternoon periods as two separate patient service time windows
hours = 10
n_periods = 60 * hours

# List of processes to be cycled through
processes_with_variability = variable_steps['Process'].to_list()

# ===== LOOPED MODEL =====

# Set up sim repositories

n_sims = 1000

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

for sim in range(n_sims) :

    if sim % 10 == 0 :
        print (sim)

    # ===== HOLDERS CREATION =====

    # Holders for check_in step

    servers_check_in = variable_steps.loc[variable_steps['Process'] == processes_with_variability[0], 'Servers'].item()

    server_dict_check_in, waiting_dict_check_in, serve_time_track_dict_check_in, waiting_time_list_check_in, service_times_completed_list_check_in, service_completed_check_in, tracker_check_in = generate_step_objects(servers_check_in, n_periods)

    # Holders for refine_complaint step
    servers_refine_complaint = variable_steps.loc[variable_steps['Process'] == processes_with_variability[1], 'Servers'].item()

    server_dict_refine_complaint, waiting_dict_refine_complaint, serve_time_track_dict_refine_complaint, waiting_time_list_refine_complaint, service_times_completed_list_refine_complaint, service_completed_refine_complaint, tracker_refine_complaint = generate_step_objects(servers_refine_complaint, n_periods)

    # Holders for exam step
    servers_exam = variable_steps.loc[variable_steps['Process'] == processes_with_variability[2], 'Servers'].item()

    server_dict_exam, waiting_dict_exam, serve_time_track_dict_exam, waiting_time_list_exam, service_times_completed_list_exam, service_completed_exam, tracker_exam = generate_step_objects(servers_exam, n_periods)

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

    for p in range(n_periods) :

        # check_in step
        # # ===== Modeled as set number for testing =====
        # if p % 15 == 0 :
        #     if (p >= 0 and p < 180) or (p >= 240 and p < 479) :
        #         n_arrivals_check_in = 4
        # else :
        #     n_arrivals_check_in = 0

        # # ===== Modeled as normal =====
        # if p % 15 == 0 :
        #     if (p >= 0 and p < 180) or (p >= 240 and p < 479) :
        #         n_arrivals_check_in = round(np.random.normal(arrivals_quarterhour, arrivals_quarterhour_sigma))
        # else :
        #     n_arrivals_check_in = 0

        # ===== Modeled as poisson =====
        if p % 15 == 0 :
            if (p >= 0 and p < 180) or (p >= 240 and p < 479) :
                n_arrivals_check_in = np.random.poisson(arrivals_quarterhour)
        else :
            n_arrivals_check_in = 0

        server_dict_check_in, service_completed_check_in, service_times_completed_list_check_in =  mark_service_time(server_dict_check_in, service_completed_check_in, serve_time_track_dict_check_in, service_times_completed_list_check_in)
        n_servers_free_check_in = check_servers_free(server_dict_check_in)
        from_wait_list_check_in, from_new_arrivals_check_in = how_many_to_move_from_where(waiting_dict_check_in, n_servers_free_check_in, n_arrivals_check_in)
        if from_wait_list_check_in > 0 :
            waiting_dict_check_in, server_dict_check_in, waiting_time_list_check_in, serve_time_track_dict_check_in = move_from_wait_list_to_service(waiting_dict_check_in, server_dict_check_in, from_wait_list_check_in, waiting_time_list_check_in, serve_time_track_dict_check_in, processes_with_variability[0])
        if from_new_arrivals_check_in > 0 :
            server_dict_check_in, n_arrivals_placed_check_in, serve_time_track_dict_check_in =  move_from_arrival_to_service(server_dict_check_in, n_arrivals_check_in, from_new_arrivals_check_in, serve_time_track_dict_check_in, processes_with_variability[0])
        else :
            n_arrivals_placed_check_in = 0
        waiting_dict_check_in = {k:v + 1 for k, v in waiting_dict_check_in.items()}
        waiting_dict_check_in = add_to_wait_list(waiting_dict_check_in, n_arrivals_check_in, n_arrivals_placed_check_in)
        tracker_check_in[p] = [n_arrivals_check_in, servers_check_in -  [v for v in server_dict_check_in.values()].count(np.nan), len(waiting_dict_check_in.keys()), service_completed_check_in]

        # refine_complaint step, plus pass_through steps of waiting_room, to_exam_room, vitals_check, start_note
        n_arrivals_refine_complaint = service_completed_check_in
        service_completed_check_in = 0 # Reset counter to 0 for next transition
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        server_dict_refine_complaint, service_completed_refine_complaint, service_times_completed_list_refine_complaint = mark_service_time(server_dict_refine_complaint, service_completed_refine_complaint, serve_time_track_dict_refine_complaint, service_times_completed_list_refine_complaint)
        n_servers_free_refine_complaint = check_servers_free(server_dict_refine_complaint)
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        from_wait_list_refine_complaint, from_new_arrivals_refine_complaint = how_many_to_move_from_where( waiting_dict_refine_complaint, n_servers_free_refine_complaint, n_arrivals_refine_complaint)
        if from_wait_list_refine_complaint > 0 :
            waiting_dict_refine_complaint, server_dict_refine_complaint, waiting_time_list_refine_complaint, serve_time_track_dict_refine_complaint = move_from_wait_list_to_service( waiting_dict_refine_complaint, server_dict_refine_complaint, from_wait_list_refine_complaint, waiting_time_list_refine_complaint, serve_time_track_dict_refine_complaint, processes_with_variability[1], empty_servers = flowstaff_servers_free)
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        if from_new_arrivals_refine_complaint > 0 :
            server_dict_refine_complaint, n_arrivals_placed_refine_complaint, serve_time_track_dict_refine_complaint = move_from_arrival_to_service(server_dict_refine_complaint, n_arrivals_refine_complaint, from_new_arrivals_refine_complaint, serve_time_track_dict_refine_complaint, processes_with_variability[1], empty_servers = flowstaff_servers_free)
        else :
            n_arrivals_placed_refine_complaint = 0
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        waiting_dict_refine_complaint = {k:v + 1 for k, v in waiting_dict_refine_complaint.items()}
        waiting_dict_refine_complaint = add_to_wait_list(waiting_dict_refine_complaint, n_arrivals_refine_complaint, n_arrivals_placed_refine_complaint)
        tracker_refine_complaint[p] = [n_arrivals_refine_complaint, servers_refine_complaint -  [v for v in server_dict_refine_complaint.values()].count(np.nan), len(waiting_dict_refine_complaint.keys()), service_completed_refine_complaint]

        # exam step
        n_arrivals_exam = service_completed_refine_complaint
        service_completed_refine_complaint = 0 # Reset counter to 0 for next transition
        server_dict_exam, service_completed_exam, service_times_completed_list_exam = mark_service_time(server_dict_exam, service_completed_exam, serve_time_track_dict_exam, service_times_completed_list_exam)
        n_servers_free_exam = check_servers_free(server_dict_exam)
        from_wait_list_exam, from_new_arrivals_exam = how_many_to_move_from_where(waiting_dict_exam, n_servers_free_exam, n_arrivals_exam)
        if from_wait_list_exam > 0 :
            waiting_dict_exam, server_dict_exam, waiting_time_list_exam, serve_time_track_dict_exam = move_from_wait_list_to_service(waiting_dict_exam, server_dict_exam, from_wait_list_exam, waiting_time_list_exam, serve_time_track_dict_exam, processes_with_variability[2])
        if from_new_arrivals_exam > 0 :
            server_dict_exam, n_arrivals_placed_exam, serve_time_track_dict_exam = move_from_arrival_to_service(server_dict_exam, n_arrivals_exam, from_new_arrivals_exam, serve_time_track_dict_exam, processes_with_variability[2])
        else :
            n_arrivals_placed_exam = 0
        waiting_dict_exam = {k:v + 1 for k, v in waiting_dict_exam.items()}
        waiting_dict_exam = add_to_wait_list(waiting_dict_exam, n_arrivals_exam, n_arrivals_placed_exam)
        tracker_exam[p] = [n_arrivals_exam, servers_exam -  [v for v in server_dict_exam.values()].count(np.nan), len(waiting_dict_exam.keys()), service_completed_exam]

        # follow_up step
        n_arrivals_follow_up = service_completed_exam
        service_completed_exam = 0 # Reset counter to 0 for next transition
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        server_dict_follow_up, service_completed_follow_up, service_times_completed_list_follow_up = mark_service_time(server_dict_follow_up, service_completed_follow_up, serve_time_track_dict_follow_up, service_times_completed_list_follow_up)
        n_servers_free_follow_up = check_servers_free(server_dict_follow_up)
        from_wait_list_follow_up, from_new_arrivals_follow_up = how_many_to_move_from_where( waiting_dict_follow_up, n_servers_free_follow_up, n_arrivals_follow_up)
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        if from_wait_list_follow_up > 0 :
            waiting_dict_follow_up, server_dict_follow_up, waiting_time_list_follow_up, serve_time_track_dict_follow_up = move_from_wait_list_to_service(waiting_dict_follow_up, server_dict_follow_up, from_wait_list_follow_up, waiting_time_list_follow_up, serve_time_track_dict_follow_up, processes_with_variability[3], empty_servers = flowstaff_servers_free)
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        if from_new_arrivals_follow_up > 0 :
            server_dict_follow_up, n_arrivals_placed_follow_up, serve_time_track_dict_follow_up = move_from_arrival_to_service(server_dict_follow_up, n_arrivals_follow_up, from_new_arrivals_follow_up, serve_time_track_dict_follow_up, processes_with_variability[3], empty_servers = flowstaff_servers_free)
        else :
            n_arrivals_placed_follow_up = 0
        flowstaff_servers_free = available_flow_staff(server_dict_refine_complaint, server_dict_follow_up)
        waiting_dict_follow_up = {k:v + 1 for k, v in waiting_dict_follow_up.items()}
        waiting_dict_follow_up = add_to_wait_list(waiting_dict_follow_up, n_arrivals_follow_up, n_arrivals_placed_follow_up)
        tracker_follow_up[p] = [n_arrivals_follow_up, servers_follow_up - [v for v in server_dict_follow_up.values()].count(np.nan), len(waiting_dict_follow_up.keys()), service_completed_follow_up]

        # checkout step
        n_arrivals_checkout = service_completed_follow_up
        service_completed_follow_up = 0 # Reset counter to 0 for next transition
        server_dict_checkout, service_completed_checkout, service_times_completed_list_checkout = mark_service_time(server_dict_checkout, service_completed_checkout, serve_time_track_dict_checkout, service_times_completed_list_checkout)
        n_servers_free_checkout = check_servers_free(server_dict_checkout)
        from_wait_list_checkout, from_new_arrivals_checkout = how_many_to_move_from_where(waiting_dict_checkout, n_servers_free_checkout, n_arrivals_checkout)
        if from_wait_list_checkout > 0 :
            waiting_dict_checkout, server_dict_checkout, waiting_time_list_checkout, serve_time_track_dict_checkout = move_from_wait_list_to_service(waiting_dict_checkout, server_dict_checkout, from_wait_list_checkout, waiting_time_list_checkout, serve_time_track_dict_checkout, processes_with_variability[4])
        if from_new_arrivals_checkout > 0 :
            server_dict_checkout, n_arrivals_placed_checkout, serve_time_track_dict_checkout = move_from_arrival_to_service(server_dict_checkout, n_arrivals_checkout, from_new_arrivals_checkout, serve_time_track_dict_checkout, processes_with_variability[4])
        else :
            n_arrivals_placed_checkout = 0
        waiting_dict_checkout = {k:v + 1 for k, v in waiting_dict_checkout.items()}
        waiting_dict_checkout = add_to_wait_list(waiting_dict_checkout, n_arrivals_checkout, n_arrivals_placed_checkout)
        tracker_checkout[p] = [n_arrivals_checkout, servers_checkout - [v for v in server_dict_checkout.values()].count(np.nan), len(waiting_dict_checkout.keys()), service_completed_checkout]
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
    mean_service_time_exam_final = np.mean(service_times_completed_list_exam)
    mean_patients_waiting_exam_final = tracker_exam[:, 2].sum()
    mean_waiting_time_exam_final = np.sum(waiting_time_list_exam) / served_exam_final
    max_waiting_time_exam_final = max_try_except(waiting_time_list_exam)

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

columns = ['Arrivals_check_in', 'Served_check_in', 'Mean_serve_time_check_in', 'Mean_wait_time_check_in', 'Max_wait_time_check_in', 'Arrivals_refine_complaint', 'Served_refine_complaint', 'Mean_serve_time_refine_complaint', 'Mean_wait_time_refine_complaint', 'Max_wait_time_refine_complaint', 'Arrivals_exam', 'Served_exam', 'Mean_serve_time_exam', 'Mean_wait_time_exam', 'Max_wait_time_exam', 'Arrivals_follow_up', 'Served_follow_up', 'Mean_serve_time_follow_up', 'Mean_wait_time_follow_up', 'Max_wait_time_follow_up', 'Arrivals_checkout', 'Served_checkout', 'Mean_serve_time_checkout', 'Mean_wait_time_checkout', 'Max_wait_time_checkout']

data = [sims_arrivals_check_in, sims_served_check_in, sims_mean_serve_time_check_in, sims_mean_waiting_time_check_in, sims_max_waiting_time_check_in, sims_arrivals_refine_complaint, sims_served_refine_complaint, sims_mean_serve_time_refine_complaint, sims_mean_waiting_time_refine_complaint, sims_max_waiting_time_refine_complaint, sims_arrivals_exam, sims_served_exam, sims_mean_serve_time_exam, sims_mean_waiting_time_exam, sims_max_waiting_time_exam, sims_arrivals_follow_up, sims_served_follow_up, sims_mean_serve_time_follow_up, sims_mean_waiting_time_follow_up, sims_max_waiting_time_follow_up, sims_arrivals_checkout, sims_served_checkout, sims_mean_serve_time_checkout, sims_mean_waiting_time_checkout, sims_max_waiting_time_checkout]

thruputs = ['Thruput_check_in', 'Thruput_refine_complaint', 'Thruput_exam', 'Thruput_follow_up', 'Thruput_checkout']

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

# final_results.to_csv('/Users/jbachlombardo/OneDrive - INSEAD/Coursework/P5/Analytics ISP/Results of sims/Sims/200524_5stepsLunch_Base_BunchedPoisson130.csv') ## COVID NAMING CONVENTION
final_results.T.to_csv('/Users/jbachlombardo/OneDrive - INSEAD/Coursework/P5/Analytics ISP/Results of sims/Sims/200528_streamlined_test.csv') ## TEST

t1 = time.clock()

print ('Time', t1 - t0)
print ('{:.2f} sims'.format(n_sims))
print ('{:.2f} s / sim'.format((t1 - t0) / n_sims))
