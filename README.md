# Cuda implementation of parallel bitonic mergesort

### Reference
- [https://en.wikipedia.org/wiki/Bitonic_sorter]


### Sample Output
```
# compile
$ nvcc -o bsort bsort.cu

# default: generate 10 random unsigned ints, then sort and print to stdout
$ ./bsort
 5: 424238335
 8: 596516649
 6: 719885386
 1: 846930886
 9: 1189641421
 7: 1649760492
 2: 1681692777
 3: 1714636915
 0: 1804289383
 4: 1957747793

#         vv----- user input 25: gen 25 random uints, then sort, then print to stdout
$ ./bsort 25
20: 35005211
22: 294702567
18: 304089172
24: 336465782
 5: 424238335
21: 521595368
 8: 596516649
 6: 719885386
12: 783368690
 1: 846930886
10: 1025202362
13: 1102520059
 9: 1189641421
19: 1303455736
11: 1350490027
16: 1365180540
17: 1540383426
 7: 1649760492
 2: 1681692777
 3: 1714636915
23: 1726956429
 0: 1804289383
 4: 1957747793
15: 1967513926
14: 2044897763
```
