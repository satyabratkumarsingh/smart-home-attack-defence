
import csv
import os


def write_header_if_not_exists(filename):
    header = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
        'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
        'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
        'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
        'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
        'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
        'Variance', 'Weight', 'label'
    ]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)


def write_to_csv(filename, row_data):
    write_header_if_not_exists(filename)
    with open(filename, 'a', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(row_data)

            