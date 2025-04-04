import numpy as np

def load_cd_events(filename):
    with open(filename, 'rb') as f:
        header = []
        num_comment_lines = 0
        
        while True:
            pos = f.tell()
            line = f.readline().decode('utf-8', errors='ignore')
            if not line.startswith('%'):
                f.seek(pos)
                break
            words = line.split()
            if len(words) > 2:
                if words[1] == 'Date' and len(words) > 3:
                    header.append((words[1], f"{words[2]} {words[3]}"))
                else:
                    header.append((words[1], words[2]))
            num_comment_lines += 1
        
        ev_type = np.fromfile(f, dtype=np.uint8, count=1)[0] if num_comment_lines > 0 else 0
        ev_size = np.fromfile(f, dtype=np.uint8, count=1)[0] if num_comment_lines > 0 else 8

        bof = f.tell()
        f.seek(0, 2)
        ev_size = np.uint32(ev_size)
        num_events = (f.tell() - bof) // ev_size
        f.seek(bof, 0)
        
        all_data = np.fromfile(f, dtype=np.uint32, count=num_events * 2)
        
    version = 0
    for key, value in header:
        if key == 'Version':
            version = int(value)
            break
        
    if version < 2:
        xmask, ymask, polmask = 0x1FF, 0x1FE00, 0x20000
        xshift, yshift, polshift = 0, 9, 17
    else:
        xmask, ymask, polmask = 0x3FFF, 0xFFFC000, 0xF0000000
        xshift, yshift, polshift = 0, 14, 28
        
    all_ts = all_data[0::2]
    all_addr = np.abs(all_data[1::2])
    
    events = np.zeros((num_events, 4), dtype=np.float64)
    events[:, 0] = (all_addr & xmask) >> xshift
    events[:, 1] = (all_addr & ymask) >> yshift
    events[:, 2] = (all_addr & polmask) >> polshift
    events[:, 3] = all_ts.astype(np.float64)
    
    return events