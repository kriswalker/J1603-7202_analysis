import glob
import numpy as np


def lin_average(data1, data2):
    lindata1 = 10**(data1 / 10)
    lindata2 = 10**(data2 / 10)
    linavg = (lindata1 + lindata2) / 2
    return 10 * np.log10(linavg)


def average_obserations(mjd_arc, power, nfdop, noise, names_arc, dt):
    dupes_ = []
    for i, t in enumerate(mjd_arc):
        mjd_sub = mjd_arc - t
        inds = np.argwhere((abs(mjd_sub) < dt)).flatten()
        if len(inds) > 1:
            truth = []
            for d in dupes_:
                if len(d) == len(inds):
                    t = d == inds
                    truth.append(t.all())
                else:
                    truth.append(False)
            if True not in truth:
                dupes_.append(inds)
    dupes = np.array(dupes_)
    
    power_avg = []
    noise_avg = []
    mjd_avg = []
    names_avg = []
    for j,d in enumerate(dupes):
        pdupe = power[d]
        ndupe = noise[d]
        mdupe = mjd_arc[d]
        name_dupe = names_arc[d]
        
        mavg = np.mean(mdupe)
        name_avg = name_dupe[0]
        navg = lin_average(ndupe[0], ndupe[1])
        pavg = lin_average(pdupe[0], pdupe[1])
        i = 2
        while i < len(pdupe):
            navg = lin_average(navg, ndupe[i])
            pavg = lin_average(pavg, pdupe[i])
            i += 1
        power_avg.append(pavg)
        noise_avg.append(navg)
        mjd_avg.append(mavg)
        names_avg.append(name_avg)
    
    dupes_flat = np.hstack(dupes)
    power = np.delete(power, dupes_flat, 0)
    noise = np.delete(noise, dupes_flat, 0)
    mjd_arc = np.delete(mjd_arc, dupes_flat, 0)
    names_arc = np.delete(names_arc, dupes_flat, 0)
            
    power = np.concatenate((power, power_avg), axis=0)
    noise = np.concatenate((noise, noise_avg), axis=0)
    mjd_arc = np.concatenate((mjd_arc, mjd_avg), axis=0)
    names_arc = np.concatenate((names_arc, names_avg), axis=0)
    nfdop = nfdop[:len(power)]
    
    print('{} independent observations'.format(len(power)))
    
    return mjd_arc, power, nfdop, noise, names_arc


def selection(sel_dir, power, nfdop, noise, mjd_arc, names_arc):
    leftfile = open(sel_dir + 'left.txt', 'r')
    rightfile = open(sel_dir + 'right.txt', 'r')
    avgfile = open(sel_dir + 'average.txt', 'r')
    discardfile = open(sel_dir + 'discard.txt', 'r')
    left = []
    right = []
    avg = []
    disc = []
    for line in leftfile:
        left.append(line.split()[0])
    for line in rightfile:
        right.append(line.split()[0])
    for line in avgfile:
        avg.append(line.split()[0])
    for line in discardfile:
        disc.append(line.split()[0])
    left = np.array(left)
    right = np.array(right)
    avg = np.array(avg)
    disc = np.array(disc)
    
    ids = np.array([0] * len(names_arc))#np.zeros(np.shape(names_arc))
    left_ind = np.where(names_arc == left[:,None])[1]
    right_ind = np.where(names_arc == right[:,None])[1]
    avg_ind = np.where(names_arc == avg[:,None])[1]
    
    ids[left_ind] = 1
    ids[right_ind] = 2
    ids[avg_ind] = 3
    
    l = len(nfdop[0])
    nfdop = nfdop[:, int(l/2):]
    
    power_temp = np.zeros(np.shape(power[:,:int(l/2)]))
    for i, iden in enumerate(ids):
        if (iden == 1) or (iden == 2):
            power_temp[i] = take_side(power, int(iden)-1)[i]
        elif (iden == 3) or (iden == 0):
            power_temp[i] = lin_average(take_side(power, 0), take_side(power,
                                                                       1))[i]
    power = power_temp
    
    return power, nfdop, noise, mjd_arc, names_arc, ids


def take_side(p, side):
    l = len(p[0])
    p1 = np.flip(p[:, :int(l/2)], axis=1)
    p2 = p[:, int(l/2):]
    
    pp = [p1, p2]
    
    return pp[side]


def split_profile(p, x, sig, mjd):
    p1 = take_side(p, 0)
    p2 = take_side(p, 1)
    
    pp = np.concatenate((p1, p2))
    x = np.concatenate((x, x))
    sig = np.concatenate((sig, sig))
    mjd = np.concatenate((mjd, mjd))
    
    return pp, x, sig, mjd


def peak_power(p):
    p1 = take_side(p, 0)
    p2 = take_side(p, 1)
    
    max_ind1 = np.argmax(p1, axis=1)
    max_ind2 = np.argmax(p2, axis=1)
    max_ind = np.hstack((max_ind1[:, np.newaxis], max_ind2[:, np.newaxis]))
    
    p1_max = np.amax(p1, axis=1)
    p2_max = np.amax(p2, axis=1)
    p_max = np.hstack((p1_max[:, np.newaxis], p2_max[:, np.newaxis]))
    inds = np.argmax(p_max, axis=1)
    pstack = np.hstack((p1[:, np.newaxis], p2[:, np.newaxis]))
    
    pp = np.zeros(np.shape(p1))
    for i, ind in enumerate(inds):
        pp[i] = pstack[i, ind, :]
        
    return pp, inds, max_ind, p_max


def peak_area(p, eta):
    p_left = take_side(p, 0)
    p_right = take_side(p, 1)
    
    integral_left = np.sum(p_left[:,:2000] * (eta[0,1] - eta[0,0]), axis=1)
    integral_right = np.sum(p_right[:,:2000] * (eta[0,1] - eta[0,0]), axis=1)

    integral_stack = np.hstack((integral_left[:, np.newaxis],
                                integral_right[:, np.newaxis]))
    max_inds = np.argmax(integral_stack, axis=1)
    left_max_inds = np.argwhere((max_inds == 0)).flatten()
    right_max_inds = np.argwhere((max_inds == 1)).flatten()
    
    pp = np.zeros(np.shape(p[:,int(len(eta[0])/2):]))
    pp[left_max_inds] = take_side(p, 0)[left_max_inds]
    pp[right_max_inds] = take_side(p, 1)[right_max_inds]
    
    ids = max_inds + 1
    
    return pp, ids, max_inds, integral_stack


def curate(p, eta):
    pp, inds, minds, pmax = peak_power(p)
    
    zinds1 = np.argwhere((minds[:,0]==0.)).flatten()
    zinds2 = np.argwhere((minds[:,1]==0.)).flatten()
    
    peak_zero = np.where(zinds1 == zinds2[:,None])
    zinds = zinds1[peak_zero[1]]
    
    pz = p[zinds]
    pz_left = take_side(pz, 0)
    pz_right = take_side(pz, 1)
    
    integral_left = np.sum(pz_left[:,:2000] * (eta[0,1] - eta[0,0]), axis=1)
    integral_right = np.sum(pz_right[:,:2000] * (eta[0,1] - eta[0,0]), axis=1)

    integral_stack = np.hstack((integral_left[:, np.newaxis],
                                integral_right[:, np.newaxis]))
    max_inds = np.argmax(integral_stack, axis=1)
    left_max_inds = np.argwhere((max_inds == 0)).flatten()
    right_max_inds = np.argwhere((max_inds == 1)).flatten()
    
    zinds_max_left = zinds[left_max_inds]
    zinds_max_right = zinds[right_max_inds]
    
    lefties = take_side(p, 0)[zinds_max_left]
    righties = take_side(p, 1)[zinds_max_right]
    
    pp[zinds_max_left] = lefties
    pp[zinds_max_right] = righties
    
    ids = inds + 1
    ids[zinds_max_left] = 1
    ids[zinds_max_right] = 2
    
    return pp, ids, zinds


def read_data(path, average_obs=True, dt=0.005, sel_dir=None, discard=None,
              average=None):
    datafiles = np.sort(glob.glob(path))
    nfdop = []
    power = []
    noise = []
    mjd_arc = []
    names_arc = []
    for f in datafiles:
        data = np.load(f)
        names_arc.append(data['arr_0'][0])
        mjd_arc.append(float(data['arr_0'][1]))
        
        normsspec_fdop = data['arr_1']
        normsspecavg = data['arr_2']
        nfdop.append(normsspec_fdop)
        power.append(normsspecavg)
        noise.append(data['arr_3'])
        
    nfdop = np.array(nfdop)
    power = np.array(power)
    noise = np.array(noise)
    mjd_arc = np.array(mjd_arc)
    names_arc = np.array(names_arc)
    
    print('Imported {} raw observations'.format(len(power)))
    print('MJD {0} - {1}\n'.format(round(min(mjd_arc), 3), round(max(mjd_arc),
                                                                 3)))
        
    cutoff = 10
    nfdop = nfdop[:, cutoff:-cutoff]
    power = power[:, cutoff:-cutoff]
    
    if average_obs:
        mjd_arc, power, nfdop, noise, names_arc = \
            average_obserations(mjd_arc, power, nfdop, noise, names_arc, dt)
    
    # preserve original data
    power_ = power
    nfdop_ = nfdop
    noise_ = noise
    mjd_arc_ = mjd_arc
    names_arc_ = names_arc
    
    if sel_dir is not None:
        power, nfdop, noise, mjd_arc, names_arc, ids = \
            selection(sel_dir, power, nfdop, noise, mjd_arc, names_arc)
    
    else:
        power, ids, zinds = curate(power, nfdop)        
        nfdop = nfdop[:, int(len(nfdop[0])/2):]

        if average is not None:
            averagefile = open(average, 'r')
            avg = []
            for line in averagefile:
                avg.append(line.split()[0])
            avg = np.array(avg)

            avg_ind = np.where(names_arc == avg[:,None])[1]
            power[avg_ind] = lin_average(take_side(power_, 0),
                                         take_side(power_, 1))[avg_ind]
            ids[avg_ind] = 3

            print('\n{} averaged profiles'.format(len(names_arc[avg_ind])))

        if discard is not None:
            discardfile = open(discard, 'r')
            disc = []
            for line in discardfile:
                disc.append(line.split()[0])
            disc = np.array(disc)
            
            disc_ind = np.where(names_arc == disc[:,None])[1]
            print('\n{} discarded profiles\n'.format(len(names_arc[disc_ind])))
            power = np.delete(power, disc_ind, 0)
            nfdop = np.delete(nfdop, disc_ind, 0)
            noise = np.delete(noise, disc_ind, 0)
            mjd_arc = np.delete(mjd_arc, disc_ind, 0)
            names_arc = np.delete(names_arc, disc_ind, 0)
            
    # sort in chronological order
    nfdop = nfdop[np.argsort(mjd_arc)]
    power = power[np.argsort(mjd_arc)]
    noise = noise[np.argsort(mjd_arc)]
    names_arc = names_arc[np.argsort(mjd_arc)]
    ids = ids[np.argsort(mjd_arc)]
    mjd_arc = mjd_arc[np.argsort(mjd_arc)]
    
    nfdop_ = nfdop_[np.argsort(mjd_arc_)]
    power_ = power_[np.argsort(mjd_arc_)]
    noise_ = noise_[np.argsort(mjd_arc_)]
    names_arc_ = names_arc_[np.argsort(mjd_arc_)]
    mjd_arc_ = mjd_arc_[np.argsort(mjd_arc_)]
    
    data = [nfdop, power, noise, mjd_arc, names_arc]
    data_ = [nfdop_, power_, noise_, mjd_arc_, names_arc_]
    
    return data, data_, ids
