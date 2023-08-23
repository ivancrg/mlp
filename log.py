import os

PARENT_DIR = './log'


def save_search(loc, data, prototypes, t_proto, criticisms, t_crit, m_proto, m_crit, gamma):
    # Saving data, prototypes and criticisms stats
    with open(os.path.join(loc, 'stats.txt'), 'w') as f:
        f.write(f'mproto={m_proto}\nmcrit={m_crit}\ngamma={gamma}\n\n')
        f.write(f'DATA - {len(data)}\n\n{data}\n\n')
        f.write(f'PROTOTYPES\n{t_proto}s\n\n{prototypes}\n\n')
        f.write(f'CRITICISMS\n{t_crit}s\n\n')
        f.write(
            '\n'.join([f'C: {crit}, w(C)={w}, idx={idx}' for crit, w, idx in criticisms]))
