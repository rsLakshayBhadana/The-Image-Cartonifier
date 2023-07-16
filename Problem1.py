def Sorting_an_array_to_odd_ones(array):
    n = len(array)
    Oddarray = [None]*n
    j=0
    for i in range(0,n,1):
        if array[i]%2 == 1 or array[i]%2 == -1 :
            Oddarray[j]=array[i]
            j=j+1
    
    return Oddarray



a = [1,2,3,4,6,9,10,-1]
b = Sorting_an_array_to_odd_ones(a)
print(b)

