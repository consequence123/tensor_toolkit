from numpy import*
from copy import deepcopy
from collections import Iterable
import pdb


class tensor(ndarray):
    def __new__(cls, Array, label):
        result = Array.view(cls)
        result.label = label
        result.labeldic = {l:i for i, l in enumerate(label)}
        if len(Array.shape) != len(result.label):
            raise Exception('Label wrong in __new__!')
        return result

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def set_label(self, label):
        self.label = label
        self.labeldic = {l:i for i, l in enumerate(label)}

    def getindex(self, label_list):
        labeldic = self.labeldic
        index = [labeldic[l] for l in label_list]
        return index

    def getshape(self, indices):
        shape = self.shape
        s = [shape[i] for i in indices]
        return s
    
    def getshape_label(self, label):
        indices = self.getindex(label)
        s = self.getshape(indices)
        return s

    def reset_label(self, oldlabel, newlabel):
        if newlabel in self.label:
            raise Exception('New label wrong in reset_label!')
        if oldlabel in self.label:
            labeldic = self.labeldic; index = labeldic[oldlabel]
            del labeldic[oldlabel]; labeldic[newlabel] = index
            self.labeldic = labeldic; self.label[index] = newlabel
        else: 
            raise Exception('Old label wrong in reset_label!')

    def reset_labels(self,oldlabels, newlabels):
        label = self.label; labeldic = self.labeldic
        if len(oldlabels) == len(newlabels):
            for i,l in enumerate(oldlabels):
                if l in label:
                    label[labeldic[l]] = newlabels[i]
                else:
                    raise Exception('Old labels wrong in reset_labels!')
            self.set_label(label)
        else:
            raise Exception('New labels wrong in reset_labels!')

    def set_uniclabel(self):
        label = deepcopy(self.label)
        uniclabel = list(set(label))
        uniclabeldic = {l:i for i, l in enumerate(uniclabel)}
        count = zeros(len(uniclabel), dtype = 'int')
        for i, l in label:
            j = uniclabeldic[l]
            label[i] += str(count[j])
            count[j] += 1
        self.set_label(label)

    def checklabeldic(self):
        if len(self.label) != len(self.labeldic):
#            raise Exception('Label wrong in checklabeldic!')
            print ('Label wrong in checklabeldic!')

    def transpose_labels(self,newlabel):
        newindex = self.getindex(newlabel)
        A = deepcopy(self)
        newtensor = super(tensor, A).transpose(newindex)
        newtensor.set_label(newlabel)
        return newtensor
        
    def transpose(self, newindex):
        label = self.label
        A = deepcopy(self)
        newtensor = super(tensor, A).transpose(newindex)
        newlabel = [label[i] for i in newindex]
        newtensor.set_label(newlabel)
        return newtensor
    
    def move2tail_label(self, label):
        '''
            move a label to the last
        '''
        newlabel = deepcopy(self.label)
        newlabel.remove(label); newlabel.append(label)
        newtensor = self.transpose_labels(newlabel)
        return newtensor

    def move2head_label(self, label):
        newlabel = deepcopy(self.label)
        newlabel.remove(label); newlabel.insert(0, label)
        newtensor = self.transpose_labels(newlabel)
        return newtensor

    def move_label(self, label, destination):
        newlabel = deepcopy(self.label)
        newlabel.remove(label); newlabel.insert(destination, label)
        newtensor = self.transpose_labels(newlabel)
        return newtensor

    def move_labels(self, labels, destination):
        newlabel = deepcopy(self.label)
        for l in labels:
            newlabel.remove(l) 
        for l in labels[::-1]:
            newlabel.insert(destination, l)
        newtensor = self.transpose_labels(newlabel)
        return newtensor

    def reshape_uniclabel(self):
        label = self.label
        uniclabel = list(set(label))
        uniclabel.sort()
        # pdb.set_trace()
        new_labels = deepcopy(label)
        oldlabels = [[] for l in uniclabel]
        uniclabel_dict = {l:i for i, l in enumerate(uniclabel)}
        for i, l in enumerate(label):
            j = uniclabel_dict[l]
            new_labels[i] = l + str(i)
            oldlabels[j].append(new_labels[i])
        self.set_label(new_labels)
        # pdb.set_trace()
        A = self.reshape_labels(oldlabels, uniclabel)
        return A

    def reshape_prelabel(self, prelabel):
        lenth = len(prelabel)
        label = self.label
        label1 = [l for l in label if l[:lenth] == prelabel]
        label2 = [l for l in label if l[:lenth] != prelabel]
        newlabel = label2 + [prelabel]
        index1 = self.getindex(label1)
        index2 = self.getindex(label2)
        shape = array(self.shape)
        shape1 = product(shape[index1])
        shape2 = [shape[i] for i in index2]
        shape2.append(shape1)
        A = self.reshape(shape2, newlabel)
        # pdb.set_trace()
        return A

    def reshape_labels(self, oldlabels, newlabels):
        '''
            oldlabels: [[old1, old2],[old3, old4], ...]
            newlabels: [new1, new2, ...]
        '''
        s = array(self.shape)
        # pdb.set_trace()
        new_order = [item for sublist in oldlabels for item in sublist]
        index_list = [self.getindex(label) for label in oldlabels] 
        shape = [product(s[index]) for index in index_list]
        A = self.transpose_labels(new_order)
        # pdb.set_trace()
        A = A.reshape(shape, newlabels)
        return A

    def reshape(self, newshape, newlabel):
        if len(newshape) == len(newlabel):
            newtensor = super(tensor, self).reshape(newshape)
            newtensor.set_label(newlabel)
        else: 
            raise Exception('Shape or label wrong in reshape!')
#        pdb.set_trace()
        return newtensor

    def convert2Matrix(self, shape_row, shape_col):
        return self.reshape([shape_row, shape_col],['row', 'col'])

    def convert2Matrix_index(self, index_row, index_col):
        newindex = index_row + index_col
        newtensor = self.transpose(newindex)
        shape = array(self.shape)
        shape_row = product(shape[index_row])
        shape_col = product(shape[index_col])
        M = newtensor.convert2Matrix(shape_row, shape_col)
        return M

    def convert2Matrix_label(self, label_row, label_col):
        index_row = self.getindex(label_row)
        index_col = self.getindex(label_col)
        return self.convert2Matrix_index(index_row, index_col)

    def svd(self, label_row, label_col):
        index_row = self.getindex(label_row)
        index_col = self.getindex(label_col)
        shape = self.shape
        shape_row = [shape[i] for i in index_row]
        shape_col = [shape[i] for i in index_col]
        newindex = index_row + index_col

        A = self.transpose(newindex)
        M = A.convert2Matrix(product(shape_row), product(shape_col))

        # try:
        #     U,S,VT = linalg.svd(M)
        # except:
        #     pdb.set_trace()

        U,S,VT = linalg.svd(M, full_matrices=False)
        
        nS = S.shape[0]
        U_shape = shape_row + [nS]; U_label = label_row + ['S']
        V_shape = [nS] + shape_col; V_label = ['S'] + label_col
        U = U.reshape(U_shape,U_label)
        VT = VT.reshape(V_shape,V_label)
        return U, S, VT

    def svd_norm(self, label_row, label_col, cut = 100):
        U, S, VT = self.svd(label_row, label_col)
        S = S[:cut]; U = U[...,:cut]; VT = VT[:cut,...]
        Sm = S[0]
        S = S/Sm
        return U, S, VT, Sm

    def svd_cut(self, label_row, label_col, cut = 100):
        U, S, VT = self.svd(label_row, label_col)
        S0 = S.sum()
        S = S[:cut]; U = U[...,:cut]; VT = VT[:cut,...]
        S1 = S.sum()
        error = (S0 - S1)/S0
        return U, S, VT, error

    def svd_UV(self, label_row, label_col, cut = 100):
        U, S, VT = self.svd(label_row, label_col)
        S = S[:cut]; U = U[...,:cut]; VT = VT[:cut,...]
        # pdb.set_trace()
        SS = sqrt(S)
        U = U*SS
        VT = VT.mul_vector(SS, 'S')
        return U, VT

    def QR(self, label_row, label_col, QR_label):
        index_row = self.getindex(label_row)
        index_col = self.getindex(label_col)
        shape = self.shape
        shape_row = [shape[i] for i in index_row]
        shape_col = [shape[i] for i in index_col]
        newindex = index_row + index_col

        A = self.transpose(newindex)
        M = A.convert2Matrix(product(shape_row), product(shape_col))

        Q, R = linalg.qr(M)

        K = min(M.shape)
        Q_shape = shape_row + [K]; Q_label = label_row + [QR_label]
        R_shape = [K] + shape_col; R_label = [QR_label] + label_col
        Q = Q.reshape(Q_shape,Q_label)
        R = R.reshape(R_shape,R_label)
        return Q, R

    def eigh(self, label_row, label_col):
        index_row = self.getindex(label_row)
        index_col = self.getindex(label_col)
        shape = self.shape
        shape_row = [shape[i] for i in index_row]
        shape_col = [shape[i] for i in index_col]
        newindex = index_row + index_col

        A = self.transpose(newindex)
        M = A.convert2Matrix(product(shape_row), product(shape_col))

        E, U0 = linalg.eigh(M)
        K = E.shape[0]
        U_shape = shape_row + [K]; U_label = label_row + ['E']
        U = U0.reshape(U_shape, U_label)
        # pdb.set_trace()
        return E, U

    def eig(self, label_row, label_col):
        index_row = self.getindex(label_row)
        index_col = self.getindex(label_col)
        shape = self.shape
        shape_row = [shape[i] for i in index_row]
        shape_col = [shape[i] for i in index_col]
        newindex = index_row + index_col

        A = self.transpose(newindex)
        M = A.convert2Matrix(product(shape_row), product(shape_col))

        E, U0 = linalg.eig(M)
        K = E.shape[0]
        U_shape = shape_row + [K]; U_label = label_row + ['E']
        U = U0.reshape(U_shape, U_label)
        Ud_shape = shape_col + [K]; Ud_label = label_col + ['E']
        Udag = U0.reshape(Ud_shape, Ud_label)
        Udag = Udag.conj()
        # pdb.set_trace()
        return U, E, Udag
    
    def conj_eigh(self, label_row, label_col):
        '''
            M = U * S * V^dag
            M * M^dag = U * E * U^dag
        '''
        index_row = self.getindex(label_row)
        shape = self.shape
        shape_row = [shape[i] for i in index_row]
        s = product(shape_row)
        label_row_c = [l + '_C' for l in label_row]
        newlabels = label_row + label_row_c
        M_dag = self.conj()
        M_dag.reset_labels(label_row, label_row_c)
        
        H = contract_labels(self, M_dag, label_col)
        H = H.transpose_labels(newlabels)
        H = H.convert2Matrix(s, s)
        E, U = linalg.eigh(H)
        Ushape = shape_row + [s]
        Ulabel = label_row + ['E']
        U = U.reshape(Ushape, Ulabel)
        return U, E
    
    def inv(self, label_row, label_col):
        index_row = self.getindex(label_row)
        index_col = self.getindex(label_col)
        shape = self.shape
        shape_row = [shape[i] for i in index_row]
        shape_col = [shape[i] for i in index_col]
        newindex = index_row + index_col

        A = self.transpose(newindex)
        M = A.convert2Matrix(product(shape_row), product(shape_col))

        M_inv = linalg.inv(M)
        # pdb.set_trace()
        newshape = shape_col + shape_row
        newlabel = label_col + label_row
        M_inv = M_inv.reshape(newshape, newlabel)
        return M_inv

    def __iadd__(self, other):
        ndarray.__iadd__(self, other); return self

    def __isub__(self, other):
        ndarray.__isub__(self, other); return self

    def __imul__(self, other):
        ndarray.__imul__(self, other); return self

    def __add__(self, other):
        C = ndarray.__add__(self, other)
        return tensor(C, self.label) if C.shape else C

    def __radd__(self, other):
        C = ndarray.__radd__(self, other)
        return tensor(C, self.label) if C.shape else C

    def __sub__(self, other):
        C = ndarray.__sub__(self, other)
        return tensor(C, self.label) if C.shape else C

    def __mul__(self, other):
        C = ndarray.__mul__(self, other)
        return tensor(C, self.label) if C.shape else C

    def __rmul__(self, other):
        C = ndarray.__mul__(self, other)
        return tensor(C, self.label) if C.shape else C
    
    def __truediv__(self, other):
        C = ndarray.__truediv__(self, other)
        return tensor(C, self.label) if C.shape else C

    def __pow__(self, other):
        C = ndarray.__pow__(self, other)
        return tensor(C, self.label) if C.shape else C

    def __setitem__(self, item, value):
        label = self.label
        ndarray.__setitem__(self, item, value)

    def __getitem__(self, item):
        # print (item[0])
        if isinstance(item, tuple):
            label = self.label
            newlabel = deepcopy(label)
            lenitem = len(item)
            for i in arange(lenitem):
                # pdb.set_trace()
                if isinstance(item[i], ndarray):
                    continue
                elif isinstance(item[i], int):
                    newlabel.remove(label[i])
                elif item[i] == Ellipsis:
                    for j in arange(-1, -lenitem + i, -1):
                        if isinstance(item[j],int):
                            newlabel.remove(label[j])
                    break

            # pdb.set_trace()
            C = ndarray.__getitem__(self, item)
            if newlabel: C.set_label(newlabel)
            return C
        elif isinstance(item, int):
            C = ndarray.__getitem__(self, item)
            label = deepcopy(self.label); newlabel = label[1:]
            C.set_label(newlabel)
            return C
        else:
            C = ndarray.__getitem__(self, item)
            label = deepcopy(self.label)
            C.set_label(label)
            # print 'in __getitem__'
            # pdb.set_trace()
            return C

    def __getslice__(self, start, end):
        C = ndarray.__getslice__(self, start, end)
        label = deepcopy(self.label)
        C.set_label(label)
        return C
        
    def __repr__(self):
        print ('tensor, '+str(self.shape))
        return (self.view(ndarray)).__str__()

    def __str__(self):
        return (self.view(ndarray)).__str__()

    def absmax(self):
        A = abs(self)
        return super(tensor, A).max()

    def conj(self):
        C = super(tensor, self).conj()
        label = deepcopy(self.label)
        return tensor(C, label)
    
    def dag(self):
        C = super(tensor, self).conj()
        label = self.label
        newlabel = [l + '_dag' for l in label]
        return tensor(C, newlabel)

    def deepcopy(self):
        label = deepcopy(self.label)
        C = deepcopy(self)
        C.set_label(label)
        return C

    def trace_label(self,labelpair):
        '''
            labelpair: [a,b]
        '''
        index = self.getindex(labelpair)
        Trace = self.trace(axis1 = index[0], axis2 = index[1])
        label = deepcopy(self.label)
        label.remove(labelpair[0]); label.remove(labelpair[1])
        # pdb.set_trace()
        return tensor(Trace, label) if label else Trace

    def trace_labels(self, labelpairs):
        '''
           labels pairs: [[a1, a2, ...], [b1, b2, ...]]
        '''
        labelpairs0 = labelpairs[0]; labelpairs1 = labelpairs[1]
        lenth = len(labelpairs[0])
        Trace = self.deepcopy()
        # pdb.set_trace()
        for i in arange(lenth):
            Trace = Trace.trace_label([labelpairs0[i], labelpairs1[i]])
        return Trace

    def mul_vector(self, vector, ilabel):
        '''
            vector: array
        '''
        label = self.label
        newlabel = deepcopy(label)
        newlabel.remove(ilabel); newlabel.append(ilabel)
        newtensor = self.transpose_labels(newlabel)
        newtensor = newtensor*vector
        return newtensor.transpose_labels(label)

    def div_vector(self, vector, ilabel):
        label = self.label
        newlabel = deepcopy(label)
        newlabel.remove(ilabel); newlabel.append(ilabel)
        newtensor = self.transpose_labels(newlabel)
        newtensor = newtensor/vector
        return newtensor.transpose_labels(label)

    def mul_array(self, array, ilabels):
        label_old = deepcopy(self.label)
        label_new = deepcopy(label_old)
        for ilabel in ilabels:
            label_new.remove(ilabel)
            label_new.append(ilabel)
        newtensor = self.transpose_labels(label_new)
        newtensor = newtensor*array
        newtensor = newtensor.transpose_labels(label_old)
        return newtensor
    
    def div_array(self, array, ilabels):
        label_old = deepcopy(self.label)
        label_new = deepcopy(label_old)
        for ilabel in ilabels:
            label_new.remove(ilabel)
            label_new.append(ilabel)
        newtensor = self.transpose_labels(label_new)
        newtensor = newtensor/array
        newtensor = newtensor.transpose_labels(label_old)
        return newtensor

def contract_label(A, B, label):
    '''
        label: string of a label name 
    '''
    label_A = deepcopy(A.label); labeldic_A = A.labeldic; index_A = labeldic_A[label]
    label_B = deepcopy(B.label); labeldic_B = B.labeldic; index_B = labeldic_B[label]
    label_A.remove(label); label_B.remove(label)
    newlabel = label_A + label_B
    C = tensordot(A, B, (index_A, index_B))
#    pdb.set_trace()
    C = tensor(C, newlabel)
    # C.checklabeldic()
    return C

def contract_labels(A, B, labels):
    '''
        labels: a list containing names
    '''
    label_A = deepcopy(A.label); labeldic_A = A.labeldic
    label_B = deepcopy(B.label); labeldic_B = B.labeldic
    contract_index_A = [labeldic_A[l] for l in labels]
    contract_index_B = [labeldic_B[l] for l in labels]
    newlabel_A = [l for l in label_A if l not in labels]
    newlabel_B = [l for l in label_B if l not in labels]
    newlabel = newlabel_A + newlabel_B
    C = tensordot(A, B, [contract_index_A, contract_index_B])
    C = tensor(C, newlabel)
    # C.checklabeldic()
    return C

def contract_index(A,B,index_pair):
    '''
        index: a list with a pair of indices 
    '''
    index_A = index_pair[0]; label_A = deepcopy(A.label); label_A.pop(index_A)
    index_B = index_pair[1]; label_B = deepcopy(B.label); label_B.pop(index_B)
    newlabel = label_A + label_B
    C = tensordot(A, B, index_pair)
#    pdb.set_trace()
    C = tensor(C, newlabel)
    # C.checklabeldic()
    return C

def contract_labelpair(A, B, labelpair):
    '''
        labelspair: [a1, b1]
    '''
    label_A = deepcopy(A.label); labeldic_A = A.labeldic; index_A = labeldic_A[labelpair[0]]
    label_B = deepcopy(B.label); labeldic_B = B.labeldic; index_B = labeldic_B[labelpair[1]]
    label_A.remove(labelpair[0]); label_B.remove(labelpair[1])
    newlabel = label_A + label_B
    C = tensordot(A, B, (index_A, index_B))
#    pdb.set_trace()
    C = tensor(C, newlabel)
    # C.checklabeldic()
    return C


def contract_labelpairs(A, B, labelpairs):
    '''
        labels pairs: [[a1, a2, ...], [b1, b2, ...]]
    '''
    label_A = deepcopy(A.label); labeldic_A = A.labeldic
    label_B = deepcopy(B.label); labeldic_B = B.labeldic
    contract_index_A = [labeldic_A[l] for l in labelpairs[0]]
    contract_index_B = [labeldic_B[l] for l in labelpairs[1]]
    newlabel_A = [l for l in label_A if l not in labelpairs[0]]
    newlabel_B = [l for l in label_B if l not in labelpairs[1]]
    newlabel = newlabel_A + newlabel_B
    C = tensordot(A, B, [contract_index_A, contract_index_B])
    C = tensor(C, newlabel)
    # C.checklabeldic()
    return C


if __name__ == '__main__':
    LA = ['1','2','3','4']; L_row = ['2','3']; Lolumn = ['4','1']
    A = random.random([2,3,4,5])
    A = tensor(A, LA)
    A1 = A[...,1]
    a = random.random([4])
    Aa = A.mul_vector(a, '3')
    pdb.set_trace()
    index = A.getindex(['2','3','4'])
    U, S, VT = A.svd(L_row, L_column)

    LB = ['1','2','3','5']
    B = random.random([2,3,4,5])
    B = tensor(B, LB)
    C = contract_label(A, B, '1')
#    C = contract_labels(A, B, ['1','2'])
    D = contract_index(A, B, [0, 0])
#    D = einsum('ijkl,ijkl->', A, B)
#    pdb.set_trace()
    C = A.contract_label(B, '1')
    D = A.contract_index(B, [0, 0])
#    pdb.set_trace()

    E = A + B; F = A - B; G = A*2.; H = A/2.
#    pdb.set_trace()
    J = A[1]
    pdb.set_trace()

    I = A[0,0,0,1]
    pdb.set_trace()

    A -= B
#    pdb.set_trace()
    A += B

    #A.reset_label('1','5')
    #A.reset_labels(['2','5'],['5','1'])

    pdb.set_trace()











