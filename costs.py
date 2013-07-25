
bitcounts = np.uint8(unpackbits(arange(256, dtype=uint8)).reshape(-1, 8).sum(-1))

def cost_census5(img_l, img_r, dnum):
    h, w = img_l.shape[:2]
    R = np.zeros((h, w, dnum), np.float32)
    
    def prepare(img):
        return census5(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_l, img_r = map(prepare, [img_l, img_r])

    for i in xrange(dnum):
        print i,
        sub_r = img_r[:, dnum-i-1:][:, :w]
        d = sub_r^img_l
        R[...,i] = bitcounts[d&0xff] + bitcounts[(d>>8)&0xff] + bitcounts[(d>>16)&0xff]
    return R
