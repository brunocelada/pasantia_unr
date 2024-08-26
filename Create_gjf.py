import glob
import os

os.chdir ("C:\linux")

listing = glob.glob("C:\linux\*.out")

for file in listing:
    os.rename(file,( file.rsplit( ".", 1 )[ 0 ] )+".log"); #to rename file

counter = len(glob.glob("*.log"))

for file in glob.glob("*.log"): #iterates over all files in the directory ending in .jpg        
        #for x in range(counter):             
               new = open(( file.rsplit( ".", 1 )[ 0 ] ) + ".gjf", "w")
               old = open(( file.rsplit( ".", 1 )[ 0 ] ) + ".log", "r")
               rline = old.readlines()
               print("0 1", file=new)  
               #new.write("0 1")
               start = 0
               end = 0
               for i in range (len(rline)):
                       if "Standard orientation:" in rline[i]:
                               start = i

               for n in range (start + 5, len(rline)):
                       if "---" in rline[n]:
                               end = n
                               break

               for line in rline[start+5 : end] :

                words = line.split()
       
                print(words[1]+ " " + words[3]+ " " + words[4]+ " " + words[5], file=new)

print("", file=new)
print("", file=new)
print("", file=new)
print("", file=new)
print("", file=new)
print("", file=new)

new.close()
