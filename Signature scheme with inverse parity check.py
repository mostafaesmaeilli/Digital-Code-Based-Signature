# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:49:28 2020

@author: esmos
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:28:24 2020

@author: Mostafa
"""
import numpy as np
import copy
import time

#this function sorts matrix m (m must be in row format) in descending order and matrix B according to it
def sortindesorder(m,B):
    num_columns=len(m)
    for i in range(num_columns-1):
        for j in range (i+1, num_columns):
            if (m[j] > m[i] ):
                x=m[i]
                m[i]=m[j]
                m[j]=x
                B[[i,j]] = B[[j,i]]
    return m , B 

#this function returns TRUE if k is in the set A (A has to be in the form of a row matrix)
def iskinset(A,k): 
    num_columns=len(A)
    for i in range(num_columns):
        if (A[i] == k):
            return True
    return False

#this function finds the intraset difference of elements in set A (A has to be in the form of a row matrix)
#with a number m
def findintrasetdiff(A,m):
    num_columns=len(A)
    intradiff=np.zeros(num_columns)
    for i in range(num_columns):
        intradiff[i]=abs(A[i]-m)
    intradiff=intradiff.astype(int)
    return intradiff

#this function returns TRUE if the intersect of two sets A & B (given in row matrix format) 
#is null
def isintersectnull(A, B):
    num_col_A=len(A)
    num_col_B=len(B)
    for i in range(num_col_A):
        for j in range(num_col_B):
            if (B[j] == A[i]):
                return False
    return True

#this function does Gauss-Jordan row operations on a binary rectangular matrix to  
#put it into RREF. That is, a matrix with the unit matrix on the left hand side. The output
#is two matrices. The first is the input matrix in RREF. The second is a hard copy of the input
# for purposes where the non-RREF is needed.
def rref(A):
    num_rows=len(A)
    num_columns=len(A[0])
    copy_of_A=copy.deepcopy(A)
    for i in range(num_rows):
        if (A[i][i] != 1): #check if the element on the diagonal is one
            for k in range(i+1,num_columns):#if it isnt change that column with a column that has a one in that row
                if (A[i][k] == 1):
                    A[:,[i, k]]=A[:,[k, i]]
                    copy_of_A[:,[i, k]]=copy_of_A[:,[k,i]]
                    break
        for j in range(num_rows):
            if (j==i):
                continue
            if (A[j][i] == 1):
                A[j]=(A[i]+A[j])%2
    return A, copy_of_A

#this function finds the inverse of  a square binary matrix. This is done by concatenating it with
#an identity matrix on the right hand side and performing Gauss-Jordan elimination
#to put it into RREF. The output is the binary inverse of the input.
def inverse(A):
    num_rows=len(A)
    inv=np.concatenate((A, np.identity(num_rows, dtype=int)), axis=1)
    for i in range(num_rows):
        if (inv[i][i] != 1):
            for k in range(i+1, num_rows):
                if (inv[k][i] == 1):
                    inv[[i,k]]=inv[[k,i]]
                    break
        for j in range(num_rows):
            if (i == j):
                continue
            else:
                if (inv[j][i] == 1):
                    inv[j]=(inv[j]+inv[i])%2
    X=inv[:,num_rows:2*num_rows]
    return X

#this function finds the right inverse of a parity check matrix in systematic form.
def rightinverse(systematic_parity_check):
    num_rows=len(systematic_parity_check)
    num_columns=len(systematic_parity_check[0])
    B_prime=systematic_parity_check[:,num_rows:num_columns]
    right_inverse_matrix=np.zeros((num_columns, num_rows), dtype=int)
    A_prime=np.random.randint(2, size=(num_columns-num_rows, num_rows))
    right_inverse_matrix[num_rows:num_columns,:]=A_prime
    A=(np.identity(num_rows, dtype=int)+B_prime.dot(A_prime))%2
    right_inverse=np.vstack((A, A_prime))
    return right_inverse
    

#this function signs a document and returns the signature along with the syndrome used in
#signing the document. This signature scheme is based on a code-based PKC.
def signature(document, generator_matrix_systematic, parity_check, inverse_parity_check):
    num_rows=len(generator_matrix_systematic)
    num_columns=len(generator_matrix_systematic[0])
    syndrome=document.dot(parity_check.transpose())%2
    sign_generator_matrix=(document+syndrome.dot(inverse_parity_check.transpose()))%2
    sign=sign_generator_matrix[:,num_columns-num_rows:num_columns]
    return sign, syndrome

#this function verifies the signature of a document. If the signature is valid, the function returns TRUE
#otherwise it returns FALSE
def verify(sign, syndrome, document, generator_matrix, inverse_parity_check):
    if (document.all() == ((sign.dot(generator_matrix) + syndrome.dot(inverse_parity_check.transpose()))%2).all()):
        return True
    else:
        return False
    
#This is for generating etended difference families. The families are given in
#a mtrix called ext_diff_fam where each row is a set and each column is a member
#of the corresponding set.  
t=input('Enter the number of sets in EDF ') #t is the number of sets B_t
omega=input('Enter the number of elements in each set ') #omega is the number of elements in each B_t
# start_time_1=time.process_time_ns()
start_time_1=time.process_time()
t=int(t)
omega=int(omega)
ext_diff_fam = np.zeros((t,omega), dtype=int) #the rows are B_t's and the columns are the elements in each set
distance = np.zeros(t, dtype=int) #this is the set of distances
m=np.arange(1,t+1, dtype=int)
for i in range(1,t+1):
    ext_diff_fam[i-1][1]=i
    distance[i-1]=i
m,ext_diff_fam = sortindesorder(m,ext_diff_fam)
count=2 #count counts the number of elements in B_t
while (count<omega):
    for w in range(t):
        j=t+1 #finding smallest positive integer not in distance starting from t+1 (since distance already has 1 to t)
        while (j>0):
            if (iskinset(distance,j) == False): #check if new distance is within the distance set
                candidate=m[w]+j #find the new candidate=m+j
                new_distances=abs(candidate-ext_diff_fam[w]) #compute the new distances of candidate and the corresponding set B_w
                if (isintersectnull(new_distances,distance) == True): #check if the intesect of the new distances does not overlap the previous ones
                    ext_diff_fam[w][count]=candidate #set the new element in B_w to j
                    distance=np.hstack((distance, new_distances)) #update the distance set
                    distance=np.unique(distance) #remove repetitive numbers from distance set
                    m[w]=candidate #update m_l=c
                    break #get out of loop and move onto next set
                else:
                    j=j+1 #if distances overlap find next distance
            else:
                j=j+1
    m,ext_diff_fam = sortindesorder(m,ext_diff_fam)
    count=count+1
#This part generates the first column of each circulant matrix based on the extended difference families. 
#The output is a matrix called 
#first_column where each column is the first column of the corresponding circulant
print("The maximum intraset distance is", distance[-1])
round_1_time=time.process_time()-start_time_1
# round_1_time=time.process_time_ns()-start_time_1
prime_field=input('Enter the order of the prime field to generate parity check matrix ') #enter order of prime field Z_prime
# start_time_2=time.process_time_ns()
start_time_2=time.process_time()
prime_field=int(prime_field)
circulants_first_row=np.zeros((1,prime_field), dtype=int) #first row of circulant from each row of ext_diff_fam
parity_check=np.zeros((1,prime_field), dtype=int) #allocating space for the first row of the parity check matrix
for j in range(omega):
    circulants_first_row[0][ext_diff_fam[t-1][j]]=1 #generate the first row of the parity check matrix from B_1
parity_check=circulants_first_row #enter the first row into the parity check matrix
for k in range(1,prime_field):
    parity_check=np.vstack((parity_check,np.roll(circulants_first_row,k)))# add rows from the first row by circular shifting
for i in range(t-1): #generate the first row from each extended diff family B_i
    circulants_first_row=np.zeros((1,prime_field), dtype=int) #reset the first row of circulants for corresponding B_i
    for j in range(omega):
        circulants_first_row[0][ext_diff_fam[t-2-i][j]]=1 #generate the first row of the circulant corresponding to B_i
    for k in range(prime_field):
        parity_check=np.vstack((parity_check,np.roll(circulants_first_row,k))) #add the circulant and shifted versions of it to the previous parity check matrix
parity_check=parity_check.transpose() #transpose the parity check matrix to put it in conventional format

#This part finds a generator matrix from the parity check matrix by first
#changing the obtained parity check matrix to a systematic parity check matrix

parity_check, official_parity_check = rref(parity_check)
P=parity_check[:,prime_field:prime_field*t] #take the parity check sub-matrix of the obtained systematic parity check matrix
#concatenate the transpose of the parity check submatrix with an identity matrix to obtain a systematic generator matrix
generator_matrix=np.concatenate((P.transpose(), np.identity(prime_field*(t-1), dtype=int)), axis=1)    

#this part finds a right inverse of the systematic parity check matrix using the
#rightinverse function.
right_inverse=rightinverse(parity_check)
round_2_time=time.process_time()-start_time_2
# round_2_time=time.process_time_ns()-start_time_2
print('Total key generation time is', round_1_time + round_2_time, 'seconds')

#this part signs a random document and verifies the signature.
rounds=input("Enter how many times you would like to run the signing and verifying process ")
rounds=int(rounds)
sign_and_verification_time=time.process_time()
for i in range(rounds):
    document=np.random.randint(2, size=(1,prime_field*t))
    sign, syndrome = signature(document, generator_matrix, parity_check, right_inverse)
    ver=verify(sign, syndrome, document, generator_matrix, right_inverse)
    if (ver == False):
        print("Signature verification failed")
        break
sign_and_verification_time=time.process_time()-sign_and_verification_time
print("The signing and verification time is", sign_and_verification_time/rounds, "seconds")



























