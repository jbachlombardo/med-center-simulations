import numpy as np
import pandas as pd
from scipy import stats

# ===== FUNCTIONS =====

def generate_service_time_type_condition(data, n_new_service_times = 1, skew = 0.05, n_samples = 1000) :
    """Get randomized service time based on type of condition
    To be used only if step in model is Exam by provider"""
    # Choose random service time, with probabilities given by frequency
    condition_type = np.random.choice(data['Type'], p = data['Frequency'])
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
    # Return number of new service times needed (will always be 1 unless exceptional circumstances)
    serve_times = np.random.choice(dist, n_new_service_times)
    # Round service times to nearest minute
    for i, s in enumerate(serve_times) :
        serve_times[i] = round(s)
    return condition_type, serve_times[0]

def generate_service_time_process(data, process, n_new_service_times = 1, skew = 0.05, n_samples = 1000) :
    """Get randomized service time based on process
    To be used only all steps in model except Exam by provider"""
    # Average service time for process
    avg = data.loc[data['Process'] == process, 'Time_Mean'].item()
    # Worst case upper limit for process
    worstcase = data.loc[data['Process'] == process, 'Time_WorstCase'].item()
    # % of time worst case for process
    perc_worst_case = data.loc[data['Process'] == process, 'Perc_WorstCase'].item()
    # St Dev based on % of time worst case occurs
    std = worstcase / stats.skewnorm.ppf(1 - perc_worst_case, avg)
    # Create distribution of 1000 samples based on condition type parameters
    dist = stats.skewnorm.rvs(skew, loc = avg, scale = std, size = n_samples)
    # Remove negative / too low results
    dist[dist < avg / 2] = avg / 2
    # Return number of new service times needed (will always be 1 unless exceptional circumstances)
    serve_times = np.random.choice(dist, n_new_service_times)
    # Round service times to nearest minute
    for i, s in enumerate(serve_times) :
        serve_times[i] = round(s)
    return serve_times[0]

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

def move_from_wait_list_to_service(dictionary_waiting, dictionary_service, count_from_wait_list, list_waiting_time, service_time_tracker_dict, process) :
    """Move patients from wait list to free servers
    To be preceded by: if count_from_wait_list > 0"""
    patients_on_wait_list = np.sort(list(dictionary_waiting.keys()))
    patients_move_to_serve = patients_on_wait_list[:count_from_wait_list]
    for m in patients_move_to_serve :
        list_waiting_time.append(dictionary_waiting[m])
        del dictionary_waiting[m]
        for k, v in dictionary_service.items() :
            if np.isnan(v) :
                if process == 'Exam' :
                    condition, serve_time = generate_service_time_type_condition(base_case_types)
                    dictionary_service[k] = serve_time
                else :
                    serve_time = generate_service_time_process(process_flow, process)
                    if process == 'Refine_complaint' :
                        serve_time += pass_through_steps['Time_Mean'].sum()
                unique_key = 'Service_' + str(serve_time) + '_' + str(p)
                service_time_tracker_dict[unique_key] = [0, serve_time]
    return dictionary_waiting, dictionary_service, list_waiting_time, service_time_tracker_dict

def move_from_arrival_to_service(dictionary_service, count_n_arrivals, count_from_new_arrivals, service_time_tracker_dict, process) :
    """Move patients from arrivals to free servers
    To be preceded by: if count_from_new_arrivals > 0 """
    count_arrivals_placed = 0
    for k, v in dictionary_service.items() :
        if np.isnan(v) :
            if process == 'Exam' :
                condition, serve_time = generate_service_time_type_condition(base_case_types)
            else :
                serve_time = generate_service_time_process(process_flow, process)
                if process == 'Refine_complaint' :
                    serve_time += pass_through_steps['Time_Mean'].sum()
            dictionary_service[k] = serve_time
            unique_key = 'Service_' + str(serve_time) + '_' + str(p)
            service_time_tracker_dict[unique_key] = [0, serve_time]
            count_arrivals_placed += 1
            if count_arrivals_placed >= count_from_new_arrivals :
                break
    return dictionary_service, count_arrivals_placed, service_time_tracker_dict

def add_to_wait_list(dictionary_waiting, count_n_arrivals, count_arrivals_placed) :
    """Move patients from arrivals to wait list
    To be preceded by: if arrivals_placed < n_arrivals"""
    count_diff = count_n_arrivals - count_arrivals_placed
    wait_keys = np.sort(list(dictionary_waiting.keys()))
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

# ===== MODEL PARAMETERS =====

# Estimates for how often the worst case scenario in patient service times occurs
low = 0.05
medium = 0.1
high = 0.2

# Providers available
providers = {'Doctor': 9, 'Nurse': 5, 'FlowStaff': 20, 'CSR': 10}

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
refine_complaint = ['Exam_prep', 'Refine_complaint', 'FlowStaff', 15, 15, high]
start_note = ['Exam_prep', 'Start_note', 'FlowStaff', 1, 0, low]
exam = ['Exam_provider', 'Exam', 'Doctor', np.nan, np.nan, np.nan]
checkout = ['Conclude', 'Checkout', 'CSR', 5, 5, medium]
process_flow = pd.DataFrame([checkin, wait, to_room, vitals, refine_complaint, start_note, exam, checkout], columns = cols)
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
arrivals_day = 100
arrivals_hour = arrivals_day / 10
arrivals_minute = arrivals_hour / 60

# Model full day of continuous operations
# Minutes are periods for analysis
# NOTE: Might need to adjust for morning / afternoon periods as two separate patient service time windows
hours = 10
n_periods = 60 * hours

# List of processes to be cycled through
processes_with_variability = variable_steps['Process'].to_list()

# ===== LOOPED MODEL =====

# Set up sim repositories

n_sims = 100

sims_total_wait_time_system = np.empty(shape = n_sims)
sims_total_service_time_system = np.empty(shape = n_sims)
sims_total_time_system = np.empty(shape = n_sims)

sims_thruput_checkin = np.empty(shape = n_sims)
sims_thruput_refine_complaint = np.empty(shape = n_sims)
sims_thruput_exam = np.empty(shape = n_sims)
sims_thruput_checkout = np.empty(shape = n_sims)
sims_thruput_checkin_checkout = np.empty(shape = n_sims)

for sim in range(n_sims) :

    # Holders for check_in step
    servers_check_in = variable_steps.loc[variable_steps['Process'] == processes_with_variability[0], 'Servers'].item()

    server_dict_check_in, waiting_dict_check_in, serve_time_track_dict_check_in, waiting_time_list_check_in, service_times_completed_list_check_in, service_completed_check_in, tracker_check_in = generate_step_objects(servers_check_in, n_periods)

    # Holders for refine_complaint step
    servers_refine_complaint = variable_steps.loc[variable_steps['Process'] == processes_with_variability[1], 'Servers'].item()

    server_dict_refine_complaint, waiting_dict_refine_complaint, serve_time_track_dict_refine_complaint, waiting_time_list_refine_complaint, service_times_completed_list_refine_complaint, service_completed_refine_complaint, tracker_refine_complaint = generate_step_objects(servers_refine_complaint, n_periods)

    # Holders for exam step
    servers_exam = variable_steps.loc[variable_steps['Process'] == processes_with_variability[2], 'Servers'].item()

    server_dict_exam, waiting_dict_exam, serve_time_track_dict_exam, waiting_time_list_exam, service_times_completed_list_exam, service_completed_exam, tracker_exam = generate_step_objects(servers_exam, n_periods)

    # Holders for checkout step
    servers_checkout = variable_steps.loc[variable_steps['Process'] == processes_with_variability[3], 'Servers'].item()

    server_dict_checkout, waiting_dict_checkout, serve_time_track_dict_checkout, waiting_time_list_checkout, service_times_completed_list_checkout, service_completed_checkout, tracker_checkout = generate_step_objects(servers_checkout, n_periods)

    for p in range(n_periods) :

        # check_in step
        n_arrivals_check_in = np.random.poisson(arrivals_minute)
        server_dict_check_in, service_completed_check_in, service_times_completed_list_check_in = mark_service_time(server_dict_check_in, service_completed_check_in, serve_time_track_dict_check_in, service_times_completed_list_check_in)
        n_servers_free_check_in = check_servers_free(server_dict_check_in)
        from_wait_list_check_in, from_new_arrivals_check_in = how_many_to_move_from_where(waiting_dict_check_in, n_servers_free_check_in, n_arrivals_check_in)
        if from_wait_list_check_in > 0 :
            waiting_dict_check_in, server_dict_check_in, waiting_time_list_check_in, serve_time_track_dict_check_in = move_from_wait_list_to_service(waiting_dict_check_in, server_dict_check_in, from_wait_list_check_in, waiting_time_list_check_in, serve_time_track_dict_check_in, processes_with_variability[0])
        if from_new_arrivals_check_in > 0 :
            server_dict_check_in, n_arrivals_placed_check_in, serve_time_track_dict_check_in = move_from_arrival_to_service(server_dict_check_in, n_arrivals_check_in, from_new_arrivals_check_in,\
                                             serve_time_track_dict_check_in, processes_with_variability[0])
        else :
            n_arrivals_placed_check_in = 0
        waiting_dict_check_in = {k:v + 1 for k, v in waiting_dict_check_in.items()}
        waiting_dict_check_in = add_to_wait_list(waiting_dict_check_in, n_arrivals_check_in, n_arrivals_placed_check_in)
        tracker_check_in[p] = [n_arrivals_check_in, servers_check_in - [v for v in server_dict_check_in.values()].count(np.nan), len(waiting_dict_check_in.keys()), service_completed_check_in]

        # refine_complaint step, plus pass_through steps of waiting_room, to_exam_room, vitals_check, start_note
        n_arrivals_refine_complaint = service_completed_check_in
        service_completed_check_in = 0 # Reset counter to 0 for next transition
        server_dict_refine_complaint, service_completed_refine_complaint, service_times_completed_list_refine_complaint = mark_service_time(server_dict_refine_complaint, service_completed_refine_complaint, serve_time_track_dict_refine_complaint, service_times_completed_list_refine_complaint)
        n_servers_free_refine_complaint = check_servers_free(server_dict_refine_complaint)
        from_wait_list_refine_complaint, from_new_arrivals_refine_complaint = how_many_to_move_from_where(waiting_dict_refine_complaint, n_servers_free_refine_complaint, n_arrivals_refine_complaint)
        if from_wait_list_refine_complaint > 0 :
            waiting_dict_refine_complaint, server_dict_refine_complaint, waiting_time_list_refine_complaint, serve_time_track_dict_refine_complaint = move_from_wait_list_to_service(waiting_dict_refine_complaint, server_dict_refine_complaint, from_wait_list_refine_complaint, waiting_time_list_refine_complaint, serve_time_track_dict_refine_complaint, processes_with_variability[1])
        if from_new_arrivals_refine_complaint > 0 :
            server_dict_refine_complaint, n_arrivals_placed_refine_complaint, serve_time_track_dict_refine_complaint = move_from_arrival_to_service(server_dict_refine_complaint, n_arrivals_refine_complaint, from_new_arrivals_refine_complaint, serve_time_track_dict_refine_complaint, processes_with_variability[1])
        else :
            n_arrivals_placed_refine_complaint = 0
        waiting_dict_refine_complaint = {k:v + 1 for k, v in waiting_dict_refine_complaint.items()}
        waiting_dict_refine_complaint = add_to_wait_list(waiting_dict_refine_complaint, n_arrivals_refine_complaint, n_arrivals_placed_refine_complaint)
        tracker_refine_complaint[p] = [n_arrivals_refine_complaint, servers_refine_complaint - [v for v in server_dict_refine_complaint.values()].count(np.nan), len(waiting_dict_refine_complaint.keys()), service_completed_refine_complaint]

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
        tracker_exam[p] = [n_arrivals_exam, servers_exam - [v for v in server_dict_exam.values()].count(np.nan), len(waiting_dict_exam.keys()), service_completed_exam]

        # checkout step
        n_arrivals_checkout = service_completed_exam
        service_completed_exam = 0 # Reset counter to 0 for next transition
        server_dict_checkout, service_completed_checkout, service_times_completed_list_checkout = mark_service_time(server_dict_checkout, service_completed_checkout, serve_time_track_dict_checkout, service_times_completed_list_checkout)
        n_servers_free_checkout = check_servers_free(server_dict_checkout)
        from_wait_list_checkout, from_new_arrivals_checkout = how_many_to_move_from_where(waiting_dict_checkout, n_servers_free_checkout, n_arrivals_checkout)
        if from_wait_list_checkout > 0 :
            waiting_dict_checkout, server_dict_checkout, waiting_time_list_checkout, serve_time_track_dict_checkout = move_from_wait_list_to_service(waiting_dict_checkout, server_dict_checkout, from_wait_list_checkout, waiting_time_list_checkout, serve_time_track_dict_checkout, processes_with_variability[3])
        if from_new_arrivals_checkout > 0 :
            server_dict_checkout, n_arrivals_placed_checkout, serve_time_track_dict_checkout = move_from_arrival_to_service(server_dict_checkout, n_arrivals_checkout, from_new_arrivals_checkout, serve_time_track_dict_checkout, processes_with_variability[3])
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
    mean_time_in_system_check_in_final = mean_waiting_time_check_in_final + mean_service_time_check_in_final

    # Summary stats for refine_complaint
    arrived_refine_complaint_final = tracker_refine_complaint[:, 0].sum()
    served_refine_complaint_final = tracker_refine_complaint[:, 3].sum()
    mean_service_time_refine_complaint_final = np.mean(service_times_completed_list_refine_complaint)
    mean_patients_waiting_refine_complaint_final = tracker_refine_complaint[:, 2].sum()
    mean_waiting_time_refine_complaint_final = np.sum(waiting_time_list_refine_complaint) / served_refine_complaint_final
    mean_time_in_system_refine_complaint_final = mean_waiting_time_refine_complaint_final + mean_service_time_refine_complaint_final

    # Summary stats for exam
    arrived_exam_final = tracker_exam[:, 0].sum()
    served_exam_final = tracker_exam[:, 3].sum()
    mean_service_time_exam_final = np.mean(service_times_completed_list_exam)
    mean_patients_waiting_exam_final = tracker_exam[:, 2].sum()
    mean_waiting_time_exam_final = np.sum(waiting_time_list_exam) / served_exam_final
    mean_time_in_system_exam_final = mean_waiting_time_exam_final + mean_service_time_exam_final

    # Summary stats for checkout
    arrived_checkout_final = tracker_checkout[:, 0].sum()
    served_checkout_final = tracker_checkout[:, 3].sum()
    mean_service_time_checkout_final = np.mean(service_times_completed_list_checkout)
    mean_patients_waiting_checkout_final = tracker_checkout[:, 2].sum()
    mean_waiting_time_checkout_final = np.sum(waiting_time_list_checkout) / served_checkout_final
    mean_time_in_system_checkout_final = mean_waiting_time_checkout_final + mean_service_time_checkout_final

    # Summary stats for total
    total_service_time_final = mean_service_time_check_in_final + mean_service_time_refine_complaint_final + mean_service_time_exam_final + mean_service_time_checkout_final
    total_waiting_time_final = mean_waiting_time_check_in_final + mean_waiting_time_refine_complaint_final + mean_waiting_time_exam_final + mean_waiting_time_checkout_final
    total_time_in_system_final =  total_service_time_final + total_waiting_time_final

    sims_total_service_time_system[sim] = total_service_time_final
    sims_total_wait_time_system[sim] =total_waiting_time_final
    sims_total_time_system[sim] = total_time_in_system_final

    sims_thruput_checkin[sim] = served_check_in_final / arrived_check_in_final
    sims_thruput_refine_complaint[sim] = served_refine_complaint_final / arrived_refine_complaint_final
    sims_thruput_exam[sim] = served_exam_final / arrived_exam_final
    sims_thruput_checkout[sim] = served_checkout_final / arrived_checkout_final
    sims_thruput_checkin_checkout[sim] = served_checkout_final / arrived_check_in_final

print ('Total time in system: {:.2f}'.format(np.mean(sims_total_time_system)))
print ('    [0.2, 0.8]:', np.percentile(sims_total_time_system, [20, 80]))
print ('Total service time in system: {:.2f}'.format(np.mean(sims_total_service_time_system)))
print ('    [0.2, 0.8]:', np.percentile(sims_total_service_time_system, [20, 80]))
print ('Total wait time in system: {:.2f}'.format(np.mean(sims_total_wait_time_system)))
print ('    [0.2, 0.8]:', np.percentile(sims_total_wait_time_system, [20, 80]))
print ('Total thruput of system: {:.2f}'.format(np.mean(sims_thruput_checkin_checkout)))
print ('    [0.2, 0.8]:', np.percentile(sims_thruput_checkin_checkout, [20, 80]))
print ('Total thruput of check_in: {:.2f}'.format(np.mean(sims_thruput_checkin)))
print ('    [0.2, 0.8]:', np.percentile(sims_thruput_checkin, [20, 80]))
print ('Total thruput of refine_complaint: {:.2f}'.format(np.mean(sims_thruput_refine_complaint)))
print ('    [0.2, 0.8]:', np.percentile(sims_thruput_refine_complaint, [20, 80]))
print ('Total thruput of exam: {:.2f}'.format(np.mean(sims_thruput_exam)))
print ('    [0.2, 0.8]:', np.percentile(sims_thruput_exam, [20, 80]))
print ('Total thruput of checkout: {:.2f}'.format(np.mean(sims_thruput_checkout)))
print ('    [0.2, 0.8]:', np.percentile(sims_thruput_checkout, [20, 80]))
