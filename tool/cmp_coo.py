import sys
filename = sys.argv[1]

with open(filename) as f:
    with open(filename+"cmp", "w") as f2:
        f2.write("belltocoo, coo\n")
        belltocoo = []
        orcoo = []
        for line in f:
            line =line.strip()
            if "after bell to coo" in line:
                belltocoo = line.split(": ")[1].strip()                
            if "orignal COOval" in line:
                orcoo = line.split(": ")[1].strip()
        print(belltocoo==orcoo)

