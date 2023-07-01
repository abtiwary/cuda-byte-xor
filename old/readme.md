#### How to compile:

```
~/Development/CPP/CudaTimedXOR1 ❯ nvcc xortimer.cu -o xor_timer 

~/Development/CPP/CudaTimedXOR1 ❯ nvcc --std c++17  xortimer.cu -o xor_timer 
```


#### Example run for 32 * 1024

```
~/Development/CPP/CudaTimedXOR1 ❯ ./xor_timer                                                                                           base 
❯ ./xor_timer                                                                                                              
bytes generated in: 36910619 nanoseconds                                                                                                                                
bytes generated in: 36578684 nanoseconds                                                                                                                              
CPU vector xor in: 129930 nanoseconds                                                                                                                               
GPU vector xor in: 9172 nanoseconds                                                                                                                               
1B 70 E1 B0 61 2B B8 04 CD BA 67 CB F0 48 CE 82 10 8B 31 FA 14 27 DB 1E D7 3C 5E CD 6C 01 BB 15 74 9A 96 7C 7B D7 C8 14 1B 7E 8F 28 B2 82 9C 0D 45 80 14 70 93 0F 7F AD BF B8 7F 4E 18 2A 23 93 F4 03 84 14 BC 1D 46 C2 9A 82 4B 24 1C 74 10 6B 10 E8 07 6F AA A1 5B 5F 82 CA A9 75 A6 49 9B CC 63 79 19 42 72 9A FE 79 D5 B3 74 D5 CC F8 E2 8D DE 11 66 7B 87 B2 24 4D 81 09 90 5B DE 7D 9E 12 6C 3C 97 8B B9 50 9A 4B EC CD C7 F4 AB FE 
5B 92 33 0E D8 F1 2D 82 E0 4D 22 3F 5F 4B C3 ...
```

#### Example run for 1 * 1024

```
❯ nvcc --std c++17  xortimer.cu -o xor_timer
❯ ./xor_timer

bytes generated in: 1162108 nanoseconds

bytes generated in: 1198621 nanoseconds

CPU vector xor in: 2060 nanoseconds

GPU vector xor in: 8880 nanoseconds

1A D9 33 A6 B8 01 98 B1 1D 78 33 8B F3 91 27 14 2F A8 E4 5B 35 5C 20 D0 16 B6 26 40 01 EA 08 91 F8 48 0D 46 51 21 30 4D B6 B7 B1 46 8B 73 C5 A0 5F 74 0D 02 3F C3 59 9C D3 34 EF C1 C9 71 6B 20 79 F2 C8 9C 9B E0 D9 57 63 DD 4C 7C C0 6F 8B 55 B5 2B 7B 65 E7 EB DC 72 EB 07 97 91 01 9A CB 60 62 54 88 1D 98 3A 8C E6 3D 2C 69 C1 1E 97 59 C1 3E D6 3B 09 8E F1 A4 A4 3A 07 C8 5F ...
```

#### Example run for 8 * 1024 

```
❯ nvcc --std c++17  xortimer.cu -o xor_timer                                                                                                  
❯ ./xor_timer                                                                                                                            
bytes generated in: 8819596 nanoseconds                                                                                                                                
bytes generated in: 9176473 nanoseconds                                                                                                                               
CPU vector xor in: 15860 nanoseconds                                                                                                                              
GPU vector xor in: 10191 nanoseconds                                                                                                                       
DE FE 3C 9D 7B E6 2B DE 6E 30 53 4B FB 03 7E E6 97 74 C1 20 55 3B D7 49 D2 15 A5 77 6E 2E 6C 2F D8 00 6B A9 88 ED 75 A3 76 29 DE 41 29 C3 4C 8
C 2D 03 9B C4 F2 0A B0 7F E2 BD EF A1 E1 72 0F 57 49 B7 F1 87 A5 BF FE D3 94 60 A9 A7 24 6B C6 D7 ED 72 3B 31 96 D4 11 86 63 EA 21 A7 1A 46 EB
 42 D5 A4 25 44 3F B6 1E 8A CC E7 3D A4 B7 1C C9 22 DE B3 23 95 2B 9C 69 70 DA 58 DB 8F 44 3A B9 43 D8 06 56 75 65 77 DD DF 41 1C 17 C4 42 69 
82 50 46 D7 14 BA C3 37 AB 88 F9 44 ...
```


