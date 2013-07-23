import numpy as np

def load_ply(fn):
    f = open(fn, 'rb')
    prop_to_dtype = { 'float' : np.float32, 'int' : np.int32, 'uchar' : np.uint8 }

    header = []
    while True:
        s = f.readline().split()
        header.append(s)
        if s[0] == 'end_header':
            break

    it = iter(header)
    s = it.next()
    elements = {}
    while True:
        if s[0] == 'end_header':
            break
        if s[0] == 'element':
            el_name, el_len = s[1], int(s[2])
            el_props = []
            s = it.next()
            while s[0] == 'property':
                el_props.append( s )
                s = it.next()
            if el_name == 'face':
                el_type = np.dtype( [('count', np.uint8), ('idx', np.int32, 3)] )
                elements[el_name] = np.fromfile(f, el_type, el_len)['idx'].copy()
            else:
                el_type = np.dtype( [(name, np.dtype(prop_to_dtype[tp])) for _, tp, name in el_props] )
                elements[el_name] = np.fromfile(f, el_type, el_len)
            continue
        s = it.next()
    return elements

ply_header_bin = '''ply
format binary_little_endian 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
'''

ply_face_bin = '''element face %(face_num)d
property list uchar int vertex_indices
'''


def write_ply_bin(fn, verts, colors=(192, 192, 192), faces=None):
    verts = verts.reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    
    vert_t = np.dtype([('vert', np.float32, 3), ('color', np.uint8, 3)])
    data = np.zeros(len(verts), vert_t)
    data['vert'] = verts
    data['color'] = colors

    face_data = None
    if faces is not None:
        faces = np.int32(faces).reshape(-1, 3)
        face_t = np.dtype([('vn', np.uint8), ('vs', np.uint32, 3)])
        face_data = np.zeros(len(faces), face_t)
        face_data['vn'][:] = 3
        face_data['vs'] = faces

    with open(fn, 'wb') as f:
        f.write(ply_header_bin % dict(vert_num=len(verts)))
        if face_data is not None:
            f.write(ply_face_bin % dict(face_num=len(face_data)))
        f.write('end_header\n')
        data.tofile(f)
        if face_data is not None:
            face_data.tofile(f)


if __name__ == '__main__':
    print load_ply('zzz5.ply')