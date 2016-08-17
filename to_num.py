# Data Preprocessing
# modify all data containing characters to numbers

def str_to_numstr(ss):
  new_ss=""
  for c in ss:
    new_ss = new_ss+ str(ord(c)-ord('A')+1).zfill(2)
  return new_ss

def mixstr_to_numstr(ss):
  summ = 0
  for c in ss:
    summ = summ+ord(c)
  return str(summ)


fout = open("training_data_1987_2007.txt","a")
for num in range(1987, 2008):
  print num
  fp = open(str(num)+".csv")
  fp.next()
  for line in fp:
    line.replace("NA", "0") # change all 'NA' to 0
    lisst = line.split(',')
    lisst[8] = str_to_numstr(lisst[8]) #UniqueCarrier
    lisst[10] = mixstr_to_numstr(lisst[10]) #TailNum
    lisst[16] = str_to_numstr(lisst[16]) #Origin
    lisst[17] = str_to_numstr(lisst[17]) #Dest
    fout.write(",".join(lisst))
  fp.close()
fout.close()
